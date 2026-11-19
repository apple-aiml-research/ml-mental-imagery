#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Engine tests: the benchmark builds, and evaluation scores the expert as perfect.

If the evaluation replay mis-parsed actions or episode ends, or the expert
disagreed with the engine's success criterion, the expert would not score 100%
and every reported solve rate would be wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from mental_imagery import engine
from mental_imagery.eval import score_token_streams
from mental_imagery.model.families import FAMILIES
from mental_imagery.puzzles import TASK_LABELS, build_engine, build_game_configs, game_key
from tests.conftest import compose_config

BENCHMARK_GAMES = {
    "anagram",
    "bloxorz",
    "character_recognition",
    "jigsaw",
    "mental_rotation",
    "mental_rotation_2d",
    "rush_hour",
    "sokoban",
    "tangram",
}
FULL_EPISODE = 1_000_000  # token budget large enough for any episode
SEEDS = [0, 1, 2]


def load_games(suite: str):
    return compose_config(
        "eval", [f"games={suite}", "checkpoint=unused", "output_dir=unused"]
    ).games


def make_engine(
    tokenizer, visual_tokens: dict, text_boards: str | None = None, family=FAMILIES["qwen35"]
):
    engine_config = OmegaConf.create({"render_resolution": 256, "action_limit_multiplier": 1.5})
    return build_engine(
        tokenizer,
        family=family,
        engine_config=engine_config,
        visual_tokens_config=OmegaConf.create(visual_tokens),
        seed=0,
        parallel=False,
        text_boards=text_boards,
    )


def test_engine_registers_exactly_the_benchmark_games():
    assert set(engine.available_games()) == BENCHMARK_GAMES


def test_benchmark_builds_and_has_display_labels():
    games = load_games("benchmark")
    assert len(build_game_configs(games)) == 12
    assert {game_key(entry) for entry in games} == set(TASK_LABELS)


@pytest.mark.parametrize(
    "text_boards", [None, "state", "boxes"], ids=["no_boards", "state_boards", "box_boards"]
)
@pytest.mark.parametrize("entry_index", range(12))
def test_expert_trajectory_scores_as_solved(mock_tokenizer, entry_index, text_boards):
    entry = load_games("benchmark")[entry_index]
    puzzle_engine = make_engine(mock_tokenizer, {"enabled": False}, text_boards=text_boards)
    loader = puzzle_engine.loader(games=build_game_configs([entry]), n=len(SEEDS), prefetch=0)
    expert_inputs, _, lengths, _ = loader.next_with_seeds(seeds=SEEDS).teacher_force_labels(
        FULL_EPISODE
    )
    tokens = np.zeros((len(SEEDS), int(lengths.max()) + 1), dtype=np.uint32)
    for episode, length in enumerate(lengths):
        tokens[episode, :length] = expert_inputs[episode, :length]

    outcomes = score_token_streams(
        loader.next_with_seeds(seeds=SEEDS), tokens, puzzle_engine.special_tokens
    )
    for seed, outcome in zip(SEEDS, outcomes, strict=True):
        assert outcome["is_solved"], f"{game_key(entry)} seed {seed}: expert not scored as solved"
        assert not outcome["is_truncated"] and not outcome["is_timed_out"]


def board_spans(sequence, special):
    """(start, end) of the content of every board in ``sequence``."""
    opens = np.flatnonzero(sequence == special["board_open"])
    closes = np.flatnonzero(sequence == special["board_close"])
    assert len(opens) == len(closes)
    return [(start + 1, end) for start, end in zip(opens, closes, strict=True)]


@pytest.mark.parametrize("entry_index", range(12))
def test_text_boards_are_json_state_descriptions_with_one_frame_each(mock_tokenizer, entry_index):
    entry = load_games("benchmark")[entry_index]
    puzzle_engine = make_engine(mock_tokenizer, {"enabled": False}, text_boards="state")
    special = puzzle_engine.special_tokens
    rollout = puzzle_engine.loader(
        games=build_game_configs([entry]), n=2, prefetch=0
    ).next_with_seeds(seeds=[0, 1])
    inputs, labels, lengths, _ = rollout.teacher_force_labels(FULL_EPISODE)
    _, frame_counts = rollout.take_frames()
    for episode, length in enumerate(lengths):
        sequence = inputs[episode, :length]
        assert (labels[episode, :length] == sequence).all()
        spans = board_spans(sequence, special)
        assert len(spans) == frame_counts[episode] > 1
        assert sequence[0] == special["board_open"]
        for step, (start, end) in enumerate(spans):
            description = json.loads(
                mock_tokenizer.backend_tokenizer.decode(sequence[start:end].tolist())
            )
            assert description["moves"] == step
            assert sequence[end + 1] == special["action_open"]


def unique_keys(pairs):
    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate keys in {keys}"
    return dict(pairs)


@pytest.mark.parametrize("entry_index", range(12))
def test_box_boards_list_objects_with_one_frame_each(mock_tokenizer, entry_index):
    entry = load_games("benchmark")[entry_index]
    puzzle_engine = make_engine(mock_tokenizer, {"enabled": False}, text_boards="boxes")
    special = puzzle_engine.special_tokens
    rollout = puzzle_engine.loader(
        games=build_game_configs([entry]), n=2, prefetch=0
    ).next_with_seeds(seeds=[0, 1])
    inputs, labels, lengths, _ = rollout.teacher_force_labels(FULL_EPISODE)
    _, frame_counts = rollout.take_frames()
    for episode, length in enumerate(lengths):
        sequence = inputs[episode, :length]
        assert (labels[episode, :length] == sequence).all()
        spans = board_spans(sequence, special)
        assert len(spans) == frame_counts[episode] > 1
        boards = [
            json.loads(
                mock_tokenizer.backend_tokenizer.decode(sequence[start:end].tolist()),
                object_pairs_hook=unique_keys,
            )
            for start, end in spans
        ]
        for board, (_, end) in zip(boards, spans, strict=True):
            assert board, "a board lists at least one object"
            assert len({len(coordinates) for coordinates in board.values()}) == 1, board
            assert len(next(iter(board.values()))) in (1, 2, 4, 6), board
            assert all(
                isinstance(value, int) and 0 <= value <= 100
                for coordinates in board.values()
                for value in coordinates
            ), board
            for coordinates in board.values():
                if len(coordinates) in (4, 6):  # boxes: minimum corner, then maximum corner
                    half = len(coordinates) // 2
                    assert all(
                        coordinates[axis] <= coordinates[axis + half] for axis in range(half)
                    ), board
            assert sequence[end + 1] == special["action_open"]
    repeated_inputs, _, _, _ = (
        puzzle_engine.loader(games=build_game_configs([entry]), n=2, prefetch=0)
        .next_with_seeds(seeds=[0, 1])
        .teacher_force_labels(FULL_EPISODE)
    )
    assert (repeated_inputs == inputs).all(), "box boards are not deterministic"


def test_truncated_text_board_episodes_start_at_a_board(mock_tokenizer):
    puzzle_engine = make_engine(mock_tokenizer, {"enabled": False}, text_boards="state")
    special = puzzle_engine.special_tokens
    jigsaw = [entry for entry in load_games("benchmark") if entry.name == "jigsaw"]
    rollout = puzzle_engine.loader(
        games=build_game_configs(jigsaw), n=4, prefetch=0
    ).next_with_seeds(seeds=[0, 1, 2, 3])
    max_tokens = 200
    stats = rollout.teacher_force_labels_randomly_truncated(max_tokens)
    inputs, _, lengths, _ = rollout.teacher_force_labels(max_tokens)
    assert any(episode["original_length"] > 0 for episode in stats), (
        "no episode was long enough to truncate"
    )
    for episode, length in enumerate(lengths):
        assert 0 < length <= max_tokens
        assert inputs[episode, 0] == special["board_open"]


@pytest.mark.parametrize("grid_size", [4, 8])
def test_each_board_has_grid_size_squared_placeholders_and_one_frame(mock_tokenizer, grid_size):
    puzzle_engine = make_engine(
        mock_tokenizer, {"enabled": True, "codebook_size": 16384, "grid_size": grid_size}
    )
    special = puzzle_engine.special_tokens
    sokoban = [entry for entry in load_games("benchmark") if entry.name == "sokoban"]
    rollout = puzzle_engine.loader(
        games=build_game_configs(sokoban), n=2, prefetch=0
    ).next_with_seeds(seeds=[0, 1])
    inputs, _, lengths, _ = rollout.teacher_force_labels(FULL_EPISODE)
    _, frame_counts = rollout.take_frames()
    for episode, length in enumerate(lengths):
        sequence = inputs[episode, :length]
        boards = np.flatnonzero(sequence == special["board_open"])
        assert len(boards) == frame_counts[episode] > 1
        for position in boards:
            assert (
                sequence[position + 1 : position + 1 + grid_size**2]
                == special["visual_token_placeholder"]
            ).all()
            assert sequence[position + 1 + grid_size**2] == special["board_close"]
