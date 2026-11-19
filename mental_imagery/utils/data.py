#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Training batches: prompts, expert token sequences, visual tokens, head targets."""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF

from mental_imagery.model.families import user_message_text

ENGINE_IGNORE_ID = 0xFFFFFFFF  # the engine's "no label"; PyTorch uses -100
IGNORE_INDEX = -100
HEAD_IMAGE_SIZE = 256  # visual-head targets are 256x256 boards


def build_prompts(processor, rollout, max_tokens: int, device, family) -> dict:
    """Left-padded processor inputs for ``family``'s prompts; sequence tensors are zero-extended to ``max_tokens``.

    Also returns ``prompt_lengths``. Generation starts at ``prompt_lengths.max()``.
    """
    return stage_prompts(process_prompts(processor, rollout, family), max_tokens, device)


def process_prompts(processor, rollout, family):
    """The processor's (CPU) inputs for the rollout's prompts: texts and initial images."""
    texts = [
        family.build_prompt_text(processor, user_message_text(prompt, actions))
        for prompt, actions in zip(rollout.prompts, rollout.actions, strict=True)
    ]
    images = rollout.initial_images
    inputs = processor(
        text=texts, images=family.processor_images(images), padding=True, return_tensors="pt"
    )
    family.check_image_inputs(inputs, num_images=len(images))
    return inputs


def reprocess_prompt_images(processor, rollout, family, inputs):
    """``inputs`` with the image tensors of the rollout's current initial images.

    Random truncation replaces the prompt image of a cut episode by its board at the cut (same size)
    and leaves the prompt texts unchanged; so the token ids stay and only the image tensors (the
    image processor's outputs) are recomputed.
    """
    images = rollout.initial_images
    image_inputs = processor.image_processor(
        images=family.processor_images(images), return_tensors="pt"
    )
    replaced = [
        key for key in image_inputs if key in inputs
    ]  # others (InternVL's num_patches) only shape the text
    if not replaced:
        raise RuntimeError(
            f"none of the image processor's outputs {sorted(image_inputs)} is a prompt input"
        )
    for key in replaced:
        if inputs[key].shape != image_inputs[key].shape:
            raise RuntimeError(
                f"the image processor's {key} has shape {tuple(image_inputs[key].shape)}, "
                f"the prompt inputs' {tuple(inputs[key].shape)}"
            )
        inputs[key] = image_inputs[key]
    family.check_image_inputs(inputs, num_images=len(images))
    return inputs


def stage_prompts(inputs, max_tokens: int, device) -> dict:
    """Processor inputs on ``device``, sequence tensors zero-extended to ``max_tokens``, plus ``prompt_lengths``."""
    batch_size, prompt_len = inputs["input_ids"].shape
    if prompt_len >= max_tokens:
        raise ValueError(f"prompt length {prompt_len} leaves no room in max_tokens={max_tokens}")
    batch = {}
    for key, tensor in inputs.items():
        if tensor.shape == (batch_size, prompt_len):
            dtype = torch.bool if key == "attention_mask" else tensor.dtype
            batch[key] = torch.zeros(batch_size, max_tokens, dtype=dtype, device=device)
            batch[key][:, :prompt_len] = tensor
        else:
            batch[key] = tensor.to(device)
    batch["prompt_lengths"] = inputs["attention_mask"].sum(dim=1)
    return batch


def stitch_visual_tokens(
    seq, labels, token_ids, frame_counts, gen_start, board_open_id, sentinel_id
) -> None:
    """Replace the placeholders after the k-th ``<board>`` of each episode with its k-th frame's codes."""
    board_mask = seq[:, gen_start:] == board_open_id
    rows, cols, values = [], [], []
    frame = 0
    for episode, count in enumerate(frame_counts):
        boards = board_mask[episode].nonzero(as_tuple=True)[0] + gen_start
        if len(boards) != count:
            raise RuntimeError(
                f"episode {episode}: {len(boards)} <board> tokens but {count} rendered frames"
            )
        for k in range(count):
            start = boards[k].item() + 1
            end = min(start + len(token_ids[frame + k]), seq.shape[1])
            if start < end:
                cols.append(torch.arange(start, end, device=seq.device))
                rows.append(torch.full_like(cols[-1], episode))
                values.append(token_ids[frame + k][: end - start].to(seq.device))
        frame += count
    if not cols:
        return
    rows, cols, values = torch.cat(rows), torch.cat(cols), torch.cat(values)
    labels[rows, cols] = values
    is_placeholder = seq[rows, cols] == sentinel_id
    seq[rows[is_placeholder], cols[is_placeholder]] = values[is_placeholder]
    if (seq == sentinel_id).any() or (labels == sentinel_id).any():
        raise RuntimeError("some visual-token placeholders were not replaced")


def board_content_mask(token_ids, board_open_id: int, board_close_id: int):
    """True at positions strictly inside ``<board> ... </board>`` (delimiters excluded).

    State CoT descriptions use ordinary vocabulary tokens, so they are
    identified by position rather than by token id.
    """
    is_open = token_ids == board_open_id
    depth = is_open.long().cumsum(dim=-1) - (token_ids == board_close_id).long().cumsum(dim=-1)
    return (depth > 0) & ~is_open


def collect_visual_targets(rollout, device):
    """Boards after each action as (N, 3, 256, 256) in [0, 1], plus frames per episode.

    The engine records [initial board, after action 1, ...]; the initial board is dropped. The
    frames come as one uint8 array and are converted on ``device`` (a quarter of the float bytes).
    """
    frames, frame_counts = rollout.take_frames_array()
    starts = np.cumsum([0, *frame_counts[:-1]])
    kept = [
        index
        for start, count in zip(starts, frame_counts, strict=False)  # starts is [0] with no frames
        for index in range(start + 1, start + count)
    ]
    if not kept:
        return None, frame_counts
    frames = torch.from_numpy(frames[kept])
    if frames.shape[1:3] != (HEAD_IMAGE_SIZE, HEAD_IMAGE_SIZE):  # resized as PIL images
        frames = torch.stack(
            [
                TF.pil_to_tensor(
                    TF.to_pil_image(frame.numpy()).resize((HEAD_IMAGE_SIZE, HEAD_IMAGE_SIZE))
                ).permute(1, 2, 0)
                for frame in frames
            ]
        )
    # Byte -> [0, 1] lookup table computed on the CPU, matching torchvision's to_tensor exactly (a GPU
    # division by 255 rounds some values differently).
    pixel_values = (
        torch.arange(256, dtype=torch.uint8).to(torch.get_default_dtype()).div(255).to(device)
    )
    return pixel_values[frames.to(device).permute(0, 3, 1, 2).int()].contiguous(), frame_counts


def trim_to_longest(batch: dict, labels):
    """Trim sequence tensors to the longest sequence across ranks (FSDP needs equal shapes)."""
    full_shape = batch["input_ids"].shape
    length = torch.tensor(int(batch["attention_mask"].any(dim=0).sum()), device=labels.device)
    dist.all_reduce(length, op=dist.ReduceOp.MAX)
    length = int(length)
    batch = {k: v[:, :length] if v.shape == full_shape else v for k, v in batch.items()}
    return batch, labels[:, :length]


def prepare_batch(
    loader,
    processor,
    family,
    config,
    device,
    visual_tokenizer,
    visual_token_ids,
    special_tokens,
    use_visual_head,
    stream,
) -> dict:
    """Build one micro-batch of expert trajectories on a background thread and CUDA ``stream``.

    Episodes longer than ``training.max_tokens`` are cut at a random step; the
    prompt image becomes the board at that step. ``visual_tokenizer`` is None
    unless boards are visual tokens.
    """
    rollout = loader.next()
    max_tokens = config.training.max_tokens
    with torch.cuda.stream(stream):
        inputs = process_prompts(processor, rollout, family)
        probe_length = int(inputs["attention_mask"].sum(dim=1).max())
        truncation = rollout.teacher_force_labels_randomly_truncated(max_tokens - probe_length)
        if any(episode["original_length"] for episode in truncation):  # some prompt images changed
            inputs = reprocess_prompt_images(processor, rollout, family, inputs)

        batch = stage_prompts(inputs, max_tokens, device)
        gen_start = int(batch.pop("prompt_lengths").max())
        seq, attn_mask = batch["input_ids"], batch["attention_mask"]
        labels = torch.full_like(seq, ENGINE_IGNORE_ID)
        inputs_np, labels_np, lengths, num_truncated = rollout.teacher_force_labels(
            max_tokens - gen_start
        )
        for episode, length in enumerate(lengths.tolist()):
            end = gen_start + length
            seq[episode, gen_start:end] = torch.from_numpy(
                inputs_np[episode, :length].astype(np.int64)
            )
            labels[episode, gen_start:end] = torch.from_numpy(
                labels_np[episode, :length].astype(np.int64)
            )
            attn_mask[episode, gen_start:end] = True

        if visual_tokenizer is not None:
            images, frame_counts = rollout.take_frames()
            codes = visual_tokenizer.encode(images)
            stitch_visual_tokens(
                seq,
                labels,
                [visual_token_ids[c] for c in codes],
                frame_counts,
                gen_start,
                special_tokens["board_open"],
                special_tokens["visual_token_placeholder"],
            )
        labels[labels == ENGINE_IGNORE_ID] = IGNORE_INDEX
        target_images, frame_counts = (
            collect_visual_targets(rollout, device) if use_visual_head else (None, None)
        )

        ready = torch.cuda.Event()
        ready.record(stream)
    return {
        "batch": batch,
        "labels": labels,
        "gen_start": gen_start,
        "target_images": target_images,
        "frame_counts": frame_counts,
        "num_truncated": num_truncated,
        "action_counts": list(rollout.action_counts),
        "ready": ready,
    }
