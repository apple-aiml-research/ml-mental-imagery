#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Puzzle engine construction and the benchmark task list."""

from __future__ import annotations

import os
import tempfile

from omegaconf import OmegaConf

from mental_imagery import engine

# The twelve benchmark tasks in table order: (group, [(game key, display name)]).
TASK_GROUPS = [
    (
        "Spatial rotation",
        [
            ("mental_rotation_2d", "2D Shape Matching"),
            ("mental_rotation_2d_certificate", "2D Mental Rotation"),
            ("mental_rotation", "3D Shape Matching"),
            ("mental_rotation_certificate", "3D Mental Rotation"),
        ],
    ),
    (
        "Reconstruction from pieces",
        [
            ("jigsaw", "Jigsaw"),
            ("character_recognition", "Character Recognition"),
            ("tangram", "Tangram"),
            ("tangram_certificate", "Tangram (yes/no)"),
        ],
    ),
    (
        "Sliding and pushing",
        [
            ("bloxorz", "Bloxorz"),
            ("rush_hour", "Rush Hour"),
            ("sokoban", "Sokoban"),
        ],
    ),
    ("Text", [("anagram", "Anagram")]),
]
TASK_LABELS = {key: label for _, tasks in TASK_GROUPS for key, label in tasks}


def game_key(entry) -> str:
    """Result key of a game entry: its label if set, else its engine name."""
    return entry.get("label") or entry.name


def build_game_configs(games) -> list:
    return [
        engine.GameConfig(entry.name, **OmegaConf.to_container(entry.params)) for entry in games
    ]


def excluded_seeds(games) -> list[list[int]]:
    """Per game, the seeds whose puzzles a training loader must never generate.

    A game entry with ``exclude_evaluation_seeds: N`` excludes the puzzles of seeds
    0..N-1, i.e. its evaluation puzzles, so they cannot occur in training.
    """
    return [list(range(entry.get("exclude_evaluation_seeds", 0))) for entry in games]


def text_boards_of(config) -> str | None:
    """Text boards of a training config: ``"state"`` (State CoT), ``"boxes"`` (Boxes CoT) or None.

    A config without ``text_cot.boards`` uses State CoT.
    """
    if not OmegaConf.select(config, "text_cot.enabled", default=False):
        return None
    return OmegaConf.select(config, "text_cot.boards", default="state")


def build_engine(
    tokenizer,
    family,
    engine_config,
    visual_tokens_config,
    seed: int,
    rank: int = 0,
    parallel: bool = True,
    text_boards: str | None = None,
):
    """Puzzle engine using ``tokenizer`` and the special tokens of model ``family``.

    Boards are written as visual-token placeholders if visual tokens are
    enabled, as text if ``text_boards`` is set (``"state"``: JSON state
    descriptions for State CoT, ``"boxes"``: object positions for Boxes CoT),
    and not at all otherwise.
    """
    if text_boards not in (None, "state", "boxes"):
        raise ValueError(f"text_boards must be 'state', 'boxes' or None, got {text_boards!r}")
    visual_token_config = None
    if visual_tokens_config.enabled:
        visual_token_config = engine.VisualTokenConfig(
            codebook_size=visual_tokens_config.codebook_size,
            short_edge_res=engine_config.render_resolution,
            token_short_edge=visual_tokens_config.grid_size,
        )
    with tempfile.TemporaryDirectory() as directory:  # the engine reads tokenizer.json once
        tokenizer_path = os.path.join(directory, "tokenizer.json")
        tokenizer.backend_tokenizer.save(tokenizer_path)
        return engine.PuzzleEngine(
            tokenizer_path,
            rank=rank,
            seed=seed,
            visual_tokens=visual_token_config,
            board_text=text_boards,
            render_resolution=engine_config.render_resolution,
            action_limit_multiplier=engine_config.action_limit_multiplier,
            parallel=parallel,
            **family.engine_token_kwargs(tokenizer),
        )
