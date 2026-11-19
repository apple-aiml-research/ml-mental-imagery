#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Train the FSQ board autoencoder on rendered puzzle states (single GPU).

    uv run python -m mental_imagery.prepare.train_fsq fsq=board_256        # target of the visual prediction head
    uv run python -m mental_imagery.prepare.train_fsq fsq=board_multires   # tokenizer of the FSQ visual-token runs

Each batch draws fresh puzzles, advances each by a random number of expert
moves, and renders the board (resized to one of ``resolutions``).
"""

from __future__ import annotations

import math
import os
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.utils import save_image

from mental_imagery.model.families import resolve_family
from mental_imagery.model.loading import load_processor
from mental_imagery.puzzles import build_engine, build_game_configs, excluded_seeds
from mental_imagery.utils.callbacks import build_callbacks
from mental_imagery.visual_tokenizers.fsq import FSQAutoencoder, fsq_entropy_loss


def render_batch(loader, resolutions, render_resolution) -> torch.Tensor:
    """(B, 3, R, R) pinned tensor in [-1, 1] of boards at random points of their expert solutions."""
    rollout = loader.next()
    lengths = rollout.solution_length()
    rollout.advance_expert([random.randint(0, n) for n in lengths])
    raw, width, height = rollout.render_bytes(render_resolution)
    batch = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
        len(lengths), height, width, 3
    )
    batch = batch.permute(0, 3, 1, 2).float() / 127.5 - 1
    resolution = random.choice(resolutions)
    if resolution != render_resolution:
        batch = F.interpolate(batch, size=resolution, mode="bilinear", align_corners=False)
    return batch.pin_memory()


def board_batches(loader, resolutions, render_resolution, workers: int = 3):
    """Endless batches, rendered ahead of time on a thread pool."""
    with ThreadPoolExecutor(workers) as pool:
        pending = deque(
            pool.submit(render_batch, loader, resolutions, render_resolution)
            for _ in range(2 * workers)
        )
        while True:
            yield pending.popleft().result()
            pending.append(pool.submit(render_batch, loader, resolutions, render_resolution))


def save_checkpoint(model, optimizer, scheduler, iteration, path, config) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iteration": iteration,
            "levels": model.fsq.levels.long().tolist(),
            "resolutions": list(config.resolutions),
            "games": OmegaConf.to_container(config.games, resolve=True),
        },
        path,
    )
    print(f"[fsq] saved {path}", flush=True)


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return step / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@hydra.main(config_path="../configs", config_name="train_fsq", version_base=None)
def main(config) -> None:
    if any(resolution % 16 for resolution in config.resolutions):
        raise ValueError("resolutions must be multiples of 16")
    render_resolution = max(config.engine.render_resolution, *config.resolutions)
    model = FSQAutoencoder(levels=list(config.levels)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    warmup = min(1000, config.iterations // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup, config.iterations)
    )

    # The engine needs a tokenizer; only its rendering is used here.
    no_visual_tokens = OmegaConf.create({"enabled": False})
    processor = load_processor(config.model.name, config.model.revision, no_visual_tokens)
    engine_config = OmegaConf.merge(config.engine, {"render_resolution": render_resolution})
    engine = build_engine(
        processor.tokenizer,
        family=resolve_family(config.model.family),
        engine_config=engine_config,
        visual_tokens_config=no_visual_tokens,
        seed=config.seed,
    )
    loader = engine.loader(
        games=build_game_configs(config.games),
        n=config.batch_size,
        prefetch=32,
        excluded_seeds=excluded_seeds(config.games),
    )
    batches = board_batches(loader, list(config.resolutions), render_resolution)

    checkpoint_dir = os.path.splitext(config.output_path)[0] + "_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = build_callbacks(config.callbacks, rank=0)
    callbacks.on_run_start(config, checkpoint_dir)
    callbacks.on_train_start(0)
    running = torch.zeros(3, device="cuda")  # loss, mse, entropy
    for iteration in range(1, config.iterations + 1):
        batch = next(batches).cuda(non_blocking=True)
        reconstruction, _, z_bounded = model(batch)
        mse = F.mse_loss(reconstruction, batch)
        entropy = fsq_entropy_loss(z_bounded, model.fsq.levels)
        loss = mse + config.entropy_weight * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running += torch.stack([loss, mse, entropy]).detach()
        if iteration % config.log_every == 0:
            loss_avg, mse_avg, entropy_avg = (running / config.log_every).tolist()
            running.zero_()
            print(
                f"[fsq] iter {iteration}/{config.iterations} loss={loss_avg:.4f} mse={mse_avg:.4f} "
                f"entropy={entropy_avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
            callbacks.on_metrics(
                {
                    "loss": loss_avg,
                    "mse": mse_avg,
                    "entropy": entropy_avg,
                    "lr": scheduler.get_last_lr()[0],
                    "iteration": iteration,
                },
                step=iteration,
            )
        if iteration % config.save_every == 0:
            path = os.path.join(checkpoint_dir, f"iter_{iteration:06d}.pt")
            save_checkpoint(model, optimizer, scheduler, iteration, path, config)
            callbacks.on_checkpoint_saved(path, iteration, "artifact")
            with torch.no_grad():  # originals (top row) and reconstructions (bottom row)
                samples = torch.cat([batch[:8], model(batch[:8])[0].clamp(-1, 1)]) * 0.5 + 0.5
                save_image(
                    samples, path.replace(".pt", "_reconstructions.png"), nrow=min(8, len(batch))
                )

    save_checkpoint(model, optimizer, scheduler, config.iterations, config.output_path, config)
    callbacks.on_checkpoint_saved(config.output_path, config.iterations, "artifact")
    batches.close()
    callbacks.on_run_end()


if __name__ == "__main__":
    main()
