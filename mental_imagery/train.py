#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Train a VLM on expert puzzle trajectories.

    uv run torchrun --nproc-per-node 8 -m mental_imagery.train experiment=qwen3_5_9b/visual_cot_64

Each step teacher-forces expert trajectories from the engine. The stream holds
only actions (Action Supervision), actions interleaved with a JSON description of every
board (State CoT) or with the positions of its objects (Boxes CoT), actions
plus a visual prediction head on the hidden states
(Visual Supervision), or actions interleaved with the visual tokens of every
board (Visual CoT). A relaunched run resumes from ``<run_dir>/resume``.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from mental_imagery.model.families import (
    patch_flash_attention_for_mrope,
    resolve_family,
    visual_token,
)
from mental_imagery.model.loading import (
    DTYPES,
    load_base_model,
    load_resume_model,
    rank0_first,
    wrap_fsdp,
)
from mental_imagery.model.visual_head import HiddenStateCollector, build_visual_head
from mental_imagery.puzzles import build_engine, build_game_configs, excluded_seeds, text_boards_of
from mental_imagery.utils.callbacks import build_callbacks
from mental_imagery.utils.checkpoint import (
    find_resume_checkpoint,
    load_training_state,
    save_checkpoint,
    save_resume_checkpoint,
    step_dir_name,
)
from mental_imagery.utils.data import (
    IGNORE_INDEX,
    board_content_mask,
    prepare_batch,
    trim_to_longest,
)
from mental_imagery.utils.packing import (
    assign_rows,
    build_plan,
    linear_attention_boundaries,
    row_lengths,
)
from mental_imagery.visual_tokenizers.codec import build_visual_tokenizer


def build_model(config, resume_dir, device):
    """Model (with the visual head attached if enabled), processor, and the head itself."""
    if resume_dir:
        model, processor = load_resume_model(resume_dir, config, device)
    else:
        model, processor = load_base_model(config, device)
    if model.config.text_config.vocab_size != len(processor.tokenizer):
        raise RuntimeError("model and tokenizer vocabulary sizes differ")
    text_config = model.config.text_config
    head = build_visual_head(
        config, text_config.hidden_size, text_config.num_hidden_layers, DTYPES[config.model.dtype]
    )
    if head is not None:
        if resume_dir:
            head.load_trainable_state_dict(
                torch.load(os.path.join(resume_dir, "visual_head.pt"), weights_only=True)
            )
        model.visual_prediction_head = head.to(device)
    return model, processor, head


def build_optimizer(model, head, config):
    """AdamW (the visual head at ``lr * lr_multiplier``) with linear warmup and cosine decay."""
    head_ids = {id(p) for p in head.parameters()} if head is not None else set()
    groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids]}]
    if head is not None:
        groups.append(
            {
                "params": list(head.parameters()),
                "lr": config.optimizer.lr * config.visual_head.lr_multiplier,
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
    )
    warmup = config.scheduler.warmup_steps
    decay = CosineAnnealingLR(
        optimizer,
        T_max=max(config.training.total_steps - warmup, 1),
        eta_min=config.optimizer.lr * config.scheduler.min_lr_ratio,
    )
    if warmup == 0:
        return optimizer, decay
    ramp = LinearLR(optimizer, start_factor=1e-8 / config.optimizer.lr, total_iters=warmup)
    return optimizer, SequentialLR(optimizer, schedulers=[ramp, decay], milestones=[warmup])


def language_model_loss(
    logits, labels, board_token_mask, board_loss_weight, board_metric="loss/visual_ce"
):
    """Next-token cross-entropy.

    With boards in the stream (Visual CoT, State CoT):
    ``w * CE(board tokens) + (1 - w) * CE(other tokens)``, each averaged over
    its own tokens so long boards do not swamp the actions. ``board_token_mask``
    marks the label positions (same shape as ``labels``) that hold board content.
    """
    logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    targets = labels[:, 1:].reshape(-1)
    if board_token_mask is None:
        return F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX), {}
    valid = targets != IGNORE_INDEX
    board = valid & board_token_mask[:, 1:].reshape(-1)
    text = valid & ~board

    def mean_ce(mask):
        return F.cross_entropy(logits[mask], targets[mask]) if mask.any() else logits.new_zeros(())

    board_ce, text_ce = mean_ce(board), mean_ce(text)
    loss = board_loss_weight * board_ce + (1 - board_loss_weight) * text_ce
    return loss, {board_metric: board_ce.item(), "loss/text_ce": text_ce.item()}


def visual_head_loss(
    model, hidden, positions_per_episode, target_images, frame_counts, max_frames, stop_gradient
):
    """Head loss on the board after each ``</action>`` (up to ``max_frames`` per episode).

    ``hidden`` holds the hidden states at the ``</action>`` positions, episode after episode, and
    ``positions_per_episode`` how many of them each episode has. With ``stop_gradient`` the head reads
    detached hidden states, so it decodes what the model represents without shaping it (a probe).
    """
    if hidden is None or target_images is None:
        return None
    hidden_rows, target_rows = [], []
    position_offset = target_offset = 0
    for num_positions, num_frames in zip(positions_per_episode, frame_counts, strict=True):
        num_frames -= 1  # the initial board has no </action>
        usable = min(num_frames, num_positions)
        frames = range(usable)
        if 0 < max_frames < usable:
            frames = torch.randperm(usable)[:max_frames].sort().values.tolist()
        hidden_rows += [position_offset + f for f in frames]
        target_rows += [target_offset + f for f in frames]
        position_offset += num_positions
        target_offset += num_frames
    if not hidden_rows:
        return None
    hidden = hidden[:, hidden_rows]
    if stop_gradient:
        hidden = hidden.detach()
    return model.visual_prediction_head(
        hidden.transpose(0, 1).float(), target_images[target_rows].float()
    )


def board_token_mask_for(labels, is_visual_token, config, special_tokens):
    """Label positions with board content: visual tokens, or text boards (State CoT, Boxes CoT)."""
    if is_visual_token is not None:
        return is_visual_token[labels.clamp(min=0)]
    if config.text_cot.enabled:
        return board_content_mask(
            labels, special_tokens["board_open"], special_tokens["board_close"]
        )
    return None


def pack_micro_batch(
    model,
    family,
    prepared,
    config,
    is_visual_token,
    special_tokens,
    use_visual_head,
    group,
    linear_attention: bool,
) -> dict:
    """A prepared (padded) micro-batch with its episodes packed (``mental_imagery/utils/packing.py``).

    Runs on the batch-preparation thread, so the GPU does not wait for it. Packed shapes are
    equalized across ranks over ``group`` (a CPU process group: the NCCL one belongs to the training
    thread). Masks are built on the padded micro-batch first, so their per-episode logic (board spans)
    stays per episode. ``linear_attention``: also return the segment boundaries for Gated DeltaNet
    layers, which the training thread sets before the forward pass.
    """
    batch, labels = prepared["batch"], prepared["labels"]
    align_mask = positions_per_episode = None
    if use_visual_head:
        align_mask = batch["input_ids"] == special_tokens["action_close"]
        align_mask[:, : prepared["gen_start"]] = False
        positions_per_episode = align_mask.sum(dim=1).tolist()
    board_token_mask = board_token_mask_for(labels, is_visual_token, config, special_tokens)
    lengths = batch["attention_mask"].sum(dim=1).tolist()
    rows = assign_rows(lengths, family.packed_row_tokens(config.training.max_tokens))
    shape = torch.tensor([len(rows), max(row_lengths(lengths, rows))])
    dist.all_reduce(shape, op=dist.ReduceOp.MAX, group=group)
    plan = build_plan(
        batch["attention_mask"].bool(), rows, num_rows=int(shape[0]), row_length=int(shape[1])
    )
    full_shape = batch["input_ids"].shape
    inputs = {
        key: plan.pack(value, fill_value=0) if value.shape == full_shape else value
        for key, value in batch.items()
        if key != "attention_mask"
    }  # None: masks from the positions
    inputs["position_ids"] = family.packed_position_ids(model, batch, plan)
    return {
        "inputs": inputs,
        "labels": plan.pack(labels, fill_value=IGNORE_INDEX),
        "align_mask": plan.pack(align_mask, fill_value=False) if align_mask is not None else None,
        "positions_per_episode": positions_per_episode,
        "board_token_mask": plan.pack(board_token_mask, fill_value=False)
        if board_token_mask is not None
        else None,
        "boundaries": linear_attention_boundaries(plan) if linear_attention else None,
    }


def peak_memory_gib(device) -> dict:
    """Peak GPU memory since the last logged step, the maximum over ranks; resets the peak counters."""
    peaks = (
        torch.tensor(
            [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
            dtype=torch.float64,
            device=device,
        )
        / 2**30
    )
    dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "memory/peak_allocated_gib": peaks[0].item(),
        "memory/peak_reserved_gib": peaks[1].item(),
    }


def train(config) -> None:
    enabled_methods = [
        name for name in ("visual_tokens", "visual_head", "text_cot") if config[name].enabled
    ]
    if len(enabled_methods) > 1:
        raise ValueError(f"{' and '.join(enabled_methods)} are mutually exclusive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # The engine's thread pool (rayon) would start one thread per CPU in every rank; share the CPUs between the
    # ranks of a node instead. Read when the pool first starts.
    os.environ.setdefault(
        "RAYON_NUM_THREADS",
        str(max(1, os.cpu_count() // int(os.environ.get("LOCAL_WORLD_SIZE", 1)))),
    )
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    packing = config.training.sequence_packing
    # Packed shapes are exchanged on the batch-preparation thread, over CPU (gloo): NCCL is the training thread's.
    packing_group = dist.new_group(backend="gloo") if packing else None
    family = resolve_family(config.model.family)
    if family.uses_mrope:
        patch_flash_attention_for_mrope()
    preempted = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: preempted.set())

    run_dir = os.path.join(config.paths.output_dir, config.run_name)
    callbacks = build_callbacks(config.callbacks, rank)
    accumulation = config.training.gradient_accumulation_steps
    final_step = config.training.stop_at_step or config.training.total_steps
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config, resolve=True))
        print(
            f"[train] {run_dir}: {config.training.per_gpu_batch_size * accumulation * world_size} "
            f"episodes per optimizer step on {world_size} GPUs",
            flush=True,
        )
        callbacks.on_run_start(config, run_dir)

    resume_dir = [find_resume_checkpoint(run_dir)]
    dist.broadcast_object_list(resume_dir, src=0)
    resume_dir = resume_dir[0]
    model, processor, head = build_model(config, resume_dir, device)
    unwrapped_model = model
    packing_segments = family.prepare_packing(unwrapped_model) if packing else None
    model = wrap_fsdp(model, config)
    optimizer, scheduler = build_optimizer(model, head, config)
    start_iteration = 0
    if resume_dir:
        state = load_training_state(model, optimizer, scheduler, resume_dir)
        start_iteration = state["iteration"] + 1
        if rank == 0:
            print(f"[train] resumed from {resume_dir}", flush=True)
    collector = (
        HiddenStateCollector(unwrapped_model.model.language_model.layers)
        if head is not None
        else None
    )

    # Episode seeds are seed + (batch index << 16) + episode, so this offset continues the
    # puzzle stream exactly where the interrupted run stopped.
    engine = build_engine(
        processor.tokenizer,
        family=family,
        engine_config=config.engine,
        visual_tokens_config=config.visual_tokens,
        seed=config.engine.seed + (start_iteration << 16),
        rank=rank,
        text_boards=text_boards_of(config),
    )
    loader = engine.loader(
        games=build_game_configs(config.games),
        n=config.training.per_gpu_batch_size,
        prefetch=config.engine.prefetch,
        excluded_seeds=excluded_seeds(config.games),
    )
    special_tokens = engine.special_tokens

    visual_tokenizer = visual_token_ids = is_visual_token = None
    if config.visual_tokens.enabled:
        grid_size = config.visual_tokens.grid_size
        visual_tokenizer = rank0_first(
            lambda: build_visual_tokenizer(config.visual_tokens, device, grid_size)
        )
        vocab = processor.tokenizer.get_vocab()
        codes = range(config.visual_tokens.codebook_size)
        visual_token_ids = torch.tensor([vocab[visual_token(c)] for c in codes], device=device)
        is_visual_token = torch.zeros(len(processor.tokenizer), dtype=torch.bool, device=device)
        is_visual_token[visual_token_ids] = True
    if config.text_cot.enabled:
        board_loss_weight, board_metric = config.text_cot.loss_weight, "loss/board_text_ce"
    else:
        board_loss_weight, board_metric = config.visual_tokens.loss_weight, "loss/visual_ce"

    callbacks.on_train_start(start_iteration)

    # Batches are prepared on a background thread and CUDA stream while the GPU trains.
    executor = ThreadPoolExecutor(
        max_workers=1, initializer=torch.cuda.set_device, initargs=(device,)
    )
    prep_stream = torch.cuda.Stream(device=device)

    def prepare_micro_batch():
        prepared = prepare_batch(
            loader=loader,
            processor=processor,
            family=family,
            config=config,
            device=device,
            visual_tokenizer=visual_tokenizer,
            visual_token_ids=visual_token_ids,
            special_tokens=special_tokens,
            use_visual_head=head is not None,
            stream=prep_stream,
        )
        if packing:
            with torch.cuda.stream(prep_stream):
                prepared["packed"] = pack_micro_batch(
                    unwrapped_model,
                    family,
                    prepared,
                    config,
                    is_visual_token,
                    special_tokens,
                    use_visual_head=head is not None,
                    group=packing_group,
                    linear_attention=packing_segments is not None,
                )
                prepared["ready"] = torch.cuda.Event()
                prepared["ready"].record(prep_stream)
        return prepared

    def next_batch():
        return executor.submit(prepare_micro_batch)

    def checkpoint_kwargs(step, iteration):
        return dict(
            model=model,
            unwrapped_model=unwrapped_model,
            processor=processor,
            config=config,
            state={"step": step, "iteration": iteration},
        )

    last_iteration = final_step * accumulation - 1
    step_metrics = StepMetrics()
    pending = next_batch()
    previous_iteration_start = None
    for iteration in range(start_iteration, last_iteration + 1):
        iteration_start = time.perf_counter()
        prepared = pending.result()
        batch_wait_seconds = (
            time.perf_counter() - iteration_start
        )  # the GPU idles while the next batch is late
        torch.cuda.current_stream(device).wait_event(prepared["ready"])
        if iteration < last_iteration:
            pending = next_batch()

        if packing:  # packed on the preparation thread
            packed = prepared["packed"]
            model_inputs, labels, board_token_mask = (
                packed["inputs"],
                packed["labels"],
                packed["board_token_mask"],
            )
            align_mask, positions_per_episode = (
                packed["align_mask"],
                packed["positions_per_episode"],
            )
            if packing_segments is not None:
                packing_segments.set(*packed["boundaries"])
        else:
            model_inputs, labels = trim_to_longest(prepared["batch"], prepared["labels"])
            align_mask = positions_per_episode = None
            if collector is not None:
                align_mask = model_inputs["input_ids"] == special_tokens["action_close"]
                align_mask[:, : prepared["gen_start"]] = False
                positions_per_episode = align_mask.sum(dim=1).tolist()
            board_token_mask = board_token_mask_for(labels, is_visual_token, config, special_tokens)
        if collector is not None:
            collector.prepare(align_mask)
        model.train()
        logits = model(**model_inputs, use_cache=False).logits
        hidden = collector.collect() if collector is not None else None

        loss, metrics = language_model_loss(
            logits, labels, board_token_mask, board_loss_weight, board_metric
        )
        metrics["time/batch_wait_s"] = batch_wait_seconds
        if previous_iteration_start is not None:
            metrics["time/iteration_s"] = (
                iteration_start - previous_iteration_start
            )  # the previous micro-batch
        previous_iteration_start = iteration_start
        if head is not None:
            head_loss = visual_head_loss(
                model,
                hidden,
                positions_per_episode,
                prepared["target_images"],
                prepared["frame_counts"],
                config.visual_head.max_frames_per_episode,
                stop_gradient=config.visual_head.stop_gradient,
            )
            if head_loss is not None:
                metrics["visual_head_loss"] = head_loss.item()
                loss = loss + config.visual_head.loss_weight * head_loss
        (loss / accumulation).backward()
        if packing_segments is not None:
            packing_segments.clear()  # after backward: activation checkpointing recomputes with them

        step = (iteration + 1) // accumulation
        at_step_boundary = (iteration + 1) % accumulation == 0
        current_step = (
            iteration // accumulation + 1
        )  # the optimizer step this micro-batch contributes to
        log_this_step = current_step == 1 or current_step % config.logging.log_every == 0
        if log_this_step:
            step_metrics.add(
                loss=loss, logits=logits, labels=labels, prepared=prepared, metrics=metrics
            )
        if at_step_boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.max_grad_norm
            ).item()
            step_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if log_this_step:
                memory = peak_memory_gib(device)
                record = step_metrics.reduce(
                    grad_norm=grad_norm, lr=step_lr, step=step, iteration=iteration
                )
                if record is not None:
                    record.update(memory)
                    print(
                        f"[train] step {step}/{config.training.total_steps} loss={record['loss']:.4f} "
                        f"acc={record['token_accuracy']:.3f} grad_norm={record['grad_norm']:.3f} "
                        f"lr={record['lr']:.2e} seq={record['seq_len']:.0f} "
                        f"mem={record['memory/peak_allocated_gib']:.1f}/{record['memory/peak_reserved_gib']:.1f}GiB",
                        flush=True,
                    )
                    callbacks.on_metrics(record, step)
        if iteration == start_iteration:
            torch.cuda.empty_cache()  # release memory fragmented by lazy initialization

        stop = torch.tensor(float(preempted.is_set()), device=device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if at_step_boundary:
            if (
                step % config.checkpoint.save_every_steps == 0
                or step in config.checkpoint.save_steps
                or step == final_step
            ):
                checkpoint_dir = os.path.join(run_dir, "checkpoints", step_dir_name(step))
                save_checkpoint(checkpoint_dir, **checkpoint_kwargs(step, iteration))
                callbacks.on_checkpoint_saved(checkpoint_dir, step, "eval")
            if step % config.checkpoint.resume_every_steps == 0 or stop:
                resume_checkpoint_dir = save_resume_checkpoint(
                    run_dir,
                    step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    **checkpoint_kwargs(step, iteration),
                )
                callbacks.on_checkpoint_saved(resume_checkpoint_dir, step, "resume")
        elif stop:
            # Mid-accumulation: drop the partial gradients and resume from the last full step.
            resume_checkpoint_dir = save_resume_checkpoint(
                run_dir,
                step,
                optimizer=optimizer,
                scheduler=scheduler,
                **checkpoint_kwargs(step, step * accumulation - 1),
            )
            callbacks.on_checkpoint_saved(resume_checkpoint_dir, step, "resume")
        if stop:
            if rank == 0:
                print("[train] SIGTERM: saved resume checkpoint, exiting", flush=True)
            break

    executor.shutdown(cancel_futures=True)
    callbacks.on_run_end()
    dist.destroy_process_group()


class StepMetrics:
    """Collects the micro-batches of one optimizer step, so a logged record covers the whole step on all ranks."""

    def __init__(self):
        self.sums, self.counts = defaultdict(float), defaultdict(int)

    def add(self, loss, logits, labels, prepared, metrics):
        with torch.no_grad():
            target = labels[:, 1:]
            valid = target != IGNORE_INDEX
            correct = (logits[:, :-1].argmax(dim=-1)[valid] == target[valid]).sum().item()
        averaged = {"loss": loss.item(), "seq_len": labels.shape[1], **metrics}
        for name, value in averaged.items():  # averaged over micro-batches
            self.sums[name] += value
            self.counts[name] += 1
        totals = {
            "correct": correct,
            "valid": valid.sum().item(),
            "truncated": prepared["num_truncated"],
            "episodes": len(prepared["action_counts"]),
            "actions": sum(prepared["action_counts"]),
        }
        for name, value in totals.items():  # pooled into ratios
            self.sums[name] += value

    def reduce(self, grad_norm, lr, step, iteration):
        """The step's record on rank 0 (None on other ranks); resets for the next logged step."""
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (dict(self.sums), dict(self.counts)))
        self.sums, self.counts = defaultdict(float), defaultdict(int)
        if dist.get_rank() != 0:
            return None
        sums, counts = defaultdict(float), defaultdict(int)
        for rank_sums, rank_counts in gathered:
            for name, value in rank_sums.items():
                sums[name] += value
            for name, count in rank_counts.items():
                counts[name] += count
        record = {
            "loss": sums["loss"] / counts["loss"],
            "token_accuracy": sums["correct"] / max(sums["valid"], 1),
            "truncation_rate": sums["truncated"] / sums["episodes"],
            "avg_actions": sums["actions"] / sums["episodes"],
            "grad_norm": grad_norm,
            "lr": lr,
            "seq_len": sums["seq_len"] / counts["seq_len"],
            "iteration": iteration,
            "optimizer_step": step,
        }
        record.update(
            {name: sums[name] / counts[name] for name in counts if name not in ("loss", "seq_len")}
        )
        return record


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(config) -> None:
    train(config)


if __name__ == "__main__":
    main()
