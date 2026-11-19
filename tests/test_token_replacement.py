#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Token-replacement test: board blocks are rewritten with the chosen substitute boards."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from mental_imagery.ablation.token_replacement import BOARD_SLOT_NONE, BoardReplacer
from mental_imagery.puzzles import build_game_configs
from mental_imagery.visual_tokenizers.codec import code_grid_size
from tests.test_engine import load_games, make_engine

GRID_SIZE = 4
CODEBOOK_SIZE = 16384


class ImageHashTokenizer:
    """Stand-in visual tokenizer: every code of a board is a hash of its pixels."""

    def encode(self, images):
        codes = []
        for image in images:
            grid_w, grid_h = code_grid_size(image.width, image.height, GRID_SIZE)
            code = int(hashlib.sha256(image.tobytes()).hexdigest(), 16) % CODEBOOK_SIZE
            codes.append(torch.full((grid_w * grid_h,), code))
        return codes


def board_blocks(stream, special):
    """Contents of each <board> ... </board> block of a written stream."""
    blocks, position = [], 0
    while True:
        opens = np.flatnonzero(stream[position:] == special["board_open"])
        if len(opens) == 0:
            return blocks
        start = position + opens[0] + 1
        end = start + np.flatnonzero(stream[start:] == special["board_close"])[0]
        blocks.append(stream[start:end])
        position = end + 1


def rewrite_boards(mock_tokenizer, mode, num_positions=600):
    """Writes garbage actions (``<action>x</action>``) through a BoardReplacer; returns the streams and helpers."""
    puzzle_engine = make_engine(
        mock_tokenizer, {"enabled": True, "codebook_size": CODEBOOK_SIZE, "grid_size": GRID_SIZE}
    )
    special = puzzle_engine.special_tokens
    sokoban = [entry for entry in load_games("benchmark") if entry.name == "sokoban"]
    loader = puzzle_engine.loader(games=build_game_configs(sokoban), n=2, prefetch=0)
    rollout = loader.next_with_seeds(seeds=[0, 1])
    visual_token_ids = torch.tensor(
        [mock_tokenizer.convert_tokens_to_ids(f"<v{code}>") for code in range(CODEBOOK_SIZE)]
    )
    replacer = BoardReplacer(
        mode=mode,
        seed=0,
        rollout=rollout,
        visual_tokenizer=ImageHashTokenizer(),
        visual_token_ids=visual_token_ids,
        special_tokens=special,
        render_resolution=256,
        episode_seeds=[0, 1],
    )
    garbage_action = [
        special["action_open"],
        mock_tokenizer.backend_tokenizer.token_to_id("x"),
        special["action_close"],
    ]
    done = torch.zeros(2, dtype=torch.bool)
    written = []
    position_in_action = [0, 0]
    for _ in range(num_positions):
        proposed = torch.zeros(2, dtype=torch.long)
        for episode, slot in enumerate(rollout.board_slots):
            if slot == BOARD_SLOT_NONE:  # outside a board the model writes an (invalid) action
                proposed[episode] = garbage_action[
                    position_in_action[episode] % len(garbage_action)
                ]
                position_in_action[episode] += 1
            else:
                position_in_action[episode] = 0
        tokens = replacer.override(proposed, done)
        written.append(tokens.clone())
        replacer.advance(tokens)
    streams = torch.stack(written, dim=1).numpy()
    return streams, special, visual_token_ids, loader


@pytest.mark.parametrize(
    "mode", ["ground_truth", "black_frame", "uniform", "random_history", "initial_frame"]
)
def test_every_board_block_has_the_board_length(mock_tokenizer, mode):
    streams, special, visual_token_ids, _ = rewrite_boards(mock_tokenizer, mode)
    is_visual = np.isin(streams, visual_token_ids.numpy())
    for stream, visual in zip(streams, is_visual, strict=True):
        blocks = board_blocks(stream, special)
        assert len(blocks) > 1
        assert all(len(block) == GRID_SIZE**2 for block in blocks[:-1])
        assert visual.sum() == sum(
            len(block) for block in blocks
        )  # visual tokens appear only inside boards


def test_initial_frame_repeats_the_first_board(mock_tokenizer):
    streams, special, _, loader = rewrite_boards(mock_tokenizer, "initial_frame")
    expected = ImageHashTokenizer().encode(loader.next_with_seeds(seeds=[0, 1]).initial_images)
    for episode, stream in enumerate(streams):
        for block in board_blocks(stream, special)[:-1]:
            assert len(set(block.tolist())) == 1
            assert block[0] == mock_tokenizer.convert_tokens_to_ids(
                f"<v{int(expected[episode][0])}>"
            )


def test_ground_truth_starts_from_the_true_initial_board(mock_tokenizer):
    streams, special, _, loader = rewrite_boards(mock_tokenizer, "ground_truth")
    initial_codes = ImageHashTokenizer().encode(loader.next_with_seeds(seeds=[0, 1]).initial_images)
    for episode, stream in enumerate(streams):
        first_block = board_blocks(stream, special)[0]
        assert first_block[0] == mock_tokenizer.convert_tokens_to_ids(
            f"<v{int(initial_codes[episode][0])}>"
        )
