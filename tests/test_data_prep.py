#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Batch preparation: the vectorized visual-head targets equal a per-image reference.

The visual-head targets come from one uint8 frame array converted at once; they must equal the
frames converted one PIL image at a time (and resized as PIL images when the engine renders at
another resolution), for every benchmark game.
"""

from __future__ import annotations

import pytest
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf

from mental_imagery.model.families import FAMILIES
from mental_imagery.puzzles import build_engine, build_game_configs
from mental_imagery.utils.data import HEAD_IMAGE_SIZE, collect_visual_targets
from tests.test_engine import FULL_EPISODE, load_games

SEEDS = [0, 1, 2]


def per_image_visual_targets(rollout):
    """Reference visual-head targets: each frame converted as a PIL image, one at a time."""
    images, frame_counts = rollout.take_frames()
    targets, offset = [], 0
    for count in frame_counts:
        for image in images[offset + 1 : offset + count]:
            if image.size != (HEAD_IMAGE_SIZE, HEAD_IMAGE_SIZE):
                image = image.resize((HEAD_IMAGE_SIZE, HEAD_IMAGE_SIZE))
            targets.append(TF.to_tensor(image))
        offset += count
    return (torch.stack(targets) if targets else None), frame_counts


def teacher_forced_rollout(tokenizer, entry, render_resolution: int):
    engine_config = OmegaConf.create(
        {"render_resolution": render_resolution, "action_limit_multiplier": 1.5}
    )
    puzzle_engine = build_engine(
        tokenizer,
        family=FAMILIES["qwen35"],
        engine_config=engine_config,
        visual_tokens_config=OmegaConf.create({"enabled": False}),
        seed=0,
        parallel=False,
    )
    rollout = puzzle_engine.loader(
        games=build_game_configs([entry]), n=len(SEEDS), prefetch=0
    ).next_with_seeds(seeds=SEEDS)
    rollout.teacher_force_labels(FULL_EPISODE)
    return rollout


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("render_resolution", [256, 128], ids=["head_size", "resized"])
@pytest.mark.parametrize("entry_index", range(12))
def test_visual_targets_from_the_frame_array_match_the_per_image_conversion(
    mock_tokenizer, entry_index, render_resolution, device
):
    entry = load_games("benchmark")[entry_index]
    expected, expected_counts = per_image_visual_targets(
        teacher_forced_rollout(mock_tokenizer, entry, render_resolution)
    )
    targets, frame_counts = collect_visual_targets(
        teacher_forced_rollout(mock_tokenizer, entry, render_resolution), device
    )
    assert frame_counts == expected_counts
    assert (
        targets.shape
        == expected.shape
        == (sum(expected_counts) - len(SEEDS), 3, HEAD_IMAGE_SIZE, HEAD_IMAGE_SIZE)
    )
    assert targets.dtype == expected.dtype and targets.is_contiguous()
    assert torch.equal(targets.cpu(), expected)  # bit for bit, also when converted on the GPU
