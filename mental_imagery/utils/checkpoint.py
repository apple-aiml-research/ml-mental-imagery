#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Checkpoints of a training run, under ``<output_dir>/<run_name>/``.

    checkpoints/step_XXXXXX/  weights in Hugging Face format (servable by vLLM),
                              processor, visual_head.pt (runs with a visual head),
                              config.yaml, training_state.json
    resume/step_XXXXXX/       the same plus optimizer, scheduler, and RNG state;
                              only the newest is kept

A directory is valid only once its ``COMPLETE`` marker exists.
"""

from __future__ import annotations

import json
import os
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dcs
from omegaconf import OmegaConf

from mental_imagery.model.families import resolve_family

COMPLETE = "COMPLETE"
FULL_STATE = dcs.StateDictOptions(full_state_dict=True, cpu_offload=True)
HEAD_PREFIX = "visual_prediction_head."


def step_dir_name(step: int) -> str:
    return f"step_{step:06d}"


def save_checkpoint(
    directory, model, unwrapped_model, processor, config, state, optimizer=None, scheduler=None
):
    """Gather full weights on rank 0 and write them; with an optimizer, also the training state."""
    rank = dist.get_rank()
    model_state = dcs.get_model_state_dict(model, options=FULL_STATE)
    optimizer_state = (
        dcs.get_optimizer_state_dict(model, optimizer, options=FULL_STATE) if optimizer else None
    )
    if rank == 0:
        if os.path.exists(os.path.join(directory, COMPLETE)):
            os.remove(os.path.join(directory, COMPLETE))
        os.makedirs(directory, exist_ok=True)
        weights = {k: v for k, v in model_state.items() if not k.startswith(HEAD_PREFIX)}
        head = {
            k[len(HEAD_PREFIX) :]: v
            for k, v in model_state.items()
            if k.startswith(HEAD_PREFIX)
            and not k.startswith(HEAD_PREFIX + "fsq_model.")  # frozen codec
        }
        resolve_family(config.model.family).prepare_checkpoint(unwrapped_model, weights)
        unwrapped_model.save_pretrained(directory, state_dict=weights)
        processor.save_pretrained(directory)
        if head:
            torch.save(head, os.path.join(directory, "visual_head.pt"))
        with open(os.path.join(directory, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config, resolve=True))
        with open(os.path.join(directory, "training_state.json"), "w") as f:
            json.dump(state, f)
        if optimizer is not None:
            torch.save(optimizer_state, os.path.join(directory, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(directory, "scheduler.pt"))
    dist.barrier()
    if optimizer is not None:
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        }
        torch.save(rng, os.path.join(directory, f"rng_rank_{rank:04d}.pt"))
        dist.barrier()
    if rank == 0:
        open(os.path.join(directory, COMPLETE), "w").close()
        print(f"[checkpoint] saved {directory}", flush=True)


def save_resume_checkpoint(run_dir, step, **kwargs) -> str:
    """Full training state at ``resume/step_XXXXXX`` (returned); older resume checkpoints are deleted."""
    resume_root = os.path.join(run_dir, "resume")
    directory = os.path.join(resume_root, step_dir_name(step))
    save_checkpoint(directory, **kwargs)
    if dist.get_rank() == 0:
        for name in os.listdir(resume_root):
            if name != step_dir_name(step):
                shutil.rmtree(os.path.join(resume_root, name))
    return directory


def find_resume_checkpoint(run_dir: str) -> str | None:
    resume_root = os.path.join(run_dir, "resume")
    if not os.path.isdir(resume_root):
        return None
    complete = sorted(
        d for d in os.listdir(resume_root) if os.path.exists(os.path.join(resume_root, d, COMPLETE))
    )
    return os.path.join(resume_root, complete[-1]) if complete else None


def load_training_state(model, optimizer, scheduler, directory: str) -> dict:
    """Restore optimizer, scheduler, and RNG state (the weights are loaded with the model)."""
    optimizer_state = torch.load(
        os.path.join(directory, "optimizer.pt"), map_location="cpu", weights_only=False
    )
    dcs.set_optimizer_state_dict(
        model, optimizers=optimizer, optim_state_dict=optimizer_state, options=FULL_STATE
    )
    scheduler.load_state_dict(
        torch.load(os.path.join(directory, "scheduler.pt"), weights_only=False)
    )
    rng = torch.load(
        os.path.join(directory, f"rng_rank_{dist.get_rank():04d}.pt"), weights_only=False
    )
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    with open(os.path.join(directory, "training_state.json")) as f:
        return json.load(f)
