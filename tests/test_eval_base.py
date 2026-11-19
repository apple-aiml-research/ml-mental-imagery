#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Base-model gameplay evaluation: parsing free-text answers into replayable actions."""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest
from omegaconf import OmegaConf

from mental_imagery.eval import score_token_streams
from mental_imagery.eval_base import (
    INVALID_ACTION,
    action_stream,
    answer_text,
    coin_flip,
    evaluate_game,
    parse_actions,
)
from mental_imagery.model.families import FAMILIES
from mental_imagery.puzzles import build_game_configs, game_key
from tests.conftest import MockTokenizer, make_mock_tokenizer
from tests.test_engine import load_games, make_engine

TANGRAM_ACTIONS = ["place('P0', 1, 2)", "rotatecw('P0')", "rotateccw('P0')", "done"]


@pytest.mark.parametrize("family", FAMILIES.values(), ids=FAMILIES.keys())
def test_answer_is_the_text_after_the_family_reasoning_span(family):
    reasoning_end = family.think_close_token
    assert (
        answer_text(f"let me think\n{reasoning_end}\n\nup\nleft", reasoning_end) == "\n\nup\nleft"
    )
    assert answer_text("thinking until the budget runs out", reasoning_end) == ""


def test_actions_are_mapped_to_the_engine_spelling():
    text = "1. place(\"P0\",1,2)\n- `rotatecw(P0)`\nRotateCCW('P0')"
    assert parse_actions(text, valid_actions=TANGRAM_ACTIONS, is_yes_no_task=False) == [
        "place('P0', 1, 2)",
        "rotatecw('P0')",
        "rotateccw('P0')",
        "done",
    ]


def test_commas_inside_calls_do_not_split_actions():
    text = "slide('A', 2), slide('B', -1)"
    valid_actions = ["slide('A', 2)", "slide('B', -1)", "done"]
    assert parse_actions(text, valid_actions=valid_actions, is_yes_no_task=False) == [
        "slide('A', 2)",
        "slide('B', -1)",
        "done",
    ]


def test_unknown_calls_are_kept_as_malformed_actions_and_prose_is_ignored():
    text = "I will move up, then place('P9', 0, 0)."
    valid_actions = TANGRAM_ACTIONS + ["up"]
    assert parse_actions(text, valid_actions=valid_actions, is_yes_no_task=False) == [
        "up",
        INVALID_ACTION,
        "done",
    ]


def test_yes_no_tasks_end_with_the_last_answer():
    text = "rotate(90)\nno wait\nyes"
    valid_actions = ["rotate(90)", "yes", "no"]
    assert parse_actions(text, valid_actions=valid_actions, is_yes_no_task=True) == [
        "rotate(90)",
        "yes",
    ]


def test_case_is_only_relaxed_when_unambiguous():
    valid_actions = ["identify('A')", "identify('a')", "identify('psi')", "done"]
    parsed = parse_actions(
        "identify('A')\nidentify('Psi')\nidentify('B')",
        valid_actions=valid_actions,
        is_yes_no_task=False,
    )
    assert parsed == ["identify('A')", "identify('psi')", INVALID_ACTION, "done"]


mock_tokenizer_for = functools.cache(lambda family_name: make_mock_tokenizer(FAMILIES[family_name]))


@pytest.mark.parametrize("family", FAMILIES.values(), ids=FAMILIES.keys())
@pytest.mark.parametrize("entry_index", range(12))
def test_expert_plan_written_as_text_scores_as_solved(family, entry_index):
    """A reply listing the expert's actions is scored as solved by the shared replay, for every family."""
    mock_tokenizer = mock_tokenizer_for(family.name)
    entry = load_games("benchmark")[entry_index]
    puzzle_engine = make_engine(mock_tokenizer, {"enabled": False}, family=family)
    special = puzzle_engine.special_tokens
    loader = puzzle_engine.loader(games=build_game_configs([entry]), n=2, prefetch=0)
    expert_inputs, _, lengths, _ = loader.next_with_seeds(seeds=[0, 1]).teacher_force_labels(
        1_000_000
    )
    rollout = loader.next_with_seeds(seeds=[0, 1])
    is_yes_no_task = bool(entry.params.get("certificate_game"))
    streams = []
    for episode, valid_actions in enumerate(rollout.valid_actions):
        sequence = expert_inputs[episode, : lengths[episode]].tolist()
        opens = [i for i, token in enumerate(sequence) if token == special["action_open"]]
        closes = [i for i, token in enumerate(sequence) if token == special["action_close"]]
        expert_actions = [
            mock_tokenizer.backend_tokenizer.decode(sequence[start + 1 : end])
            for start, end in zip(opens, closes, strict=True)
        ]
        reply = f"reasoning{family.think_close_token}\n" + "\n".join(
            f"{index + 1}. {action}" for index, action in enumerate(expert_actions)
        )
        parsed = parse_actions(
            answer_text(reply, family.think_close_token),
            valid_actions=valid_actions,
            is_yes_no_task=is_yes_no_task,
        )
        streams.append(
            mock_tokenizer.backend_tokenizer.encode(
                action_stream(parsed, family.turn_end_token)
            ).ids
        )
    tokens = np.zeros((2, max(map(len, streams))), dtype=np.uint32)
    for row, stream in enumerate(streams):
        tokens[row, : len(stream)] = stream
    for outcome in score_token_streams(rollout, tokens, special):
        assert outcome["is_solved"], game_key(entry)
        assert not outcome["is_truncated"]


QWEN = FAMILIES["qwen35"]
UNFINISHED = "still reasoning when the response budget ran out"


class EncodingMockTokenizer(MockTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.backend_tokenizer.encode(text).ids


class ScriptedClient:
    """Replies to the requests in order, cycling through ``replies``."""

    def __init__(self, *replies: str):
        self.replies = itertools.cycle(replies)

    def generate_batch(self, messages, max_new_tokens):
        return [next(self.replies) for _ in messages]


@functools.cache
def base_eval_engine():
    tokenizer = EncodingMockTokenizer(make_mock_tokenizer(QWEN).backend_tokenizer)
    return make_engine(tokenizer, {"enabled": False}), tokenizer


def entry_named(label: str):
    return next(entry for entry in load_games("benchmark") if game_key(entry) == label)


def outcomes(label: str, client, unfinished_yes_no: str = "coin_flip") -> list[dict]:
    engine, tokenizer = base_eval_engine()
    config = OmegaConf.create(
        {
            "num_puzzles": 12,
            "batch_size": 5,
            "max_new_tokens": 1,
            "unfinished_yes_no": unfinished_yes_no,
        }
    )
    return evaluate_game(entry_named(label), engine, tokenizer, QWEN, client, config, None)


def finished(answer: str) -> str:
    return f"reasoning{QWEN.think_close_token}\n{answer}"


def test_coin_flip_is_reproducible_per_puzzle():
    flips = [coin_flip("tangram_certificate", seed) for seed in range(20)]
    assert flips == [coin_flip("tangram_certificate", seed) for seed in range(20)]
    assert set(flips) == {"yes", "no"}
    assert flips != [coin_flip("mental_rotation_certificate", seed) for seed in range(20)]


@pytest.mark.parametrize("label", ["tangram_certificate", "mental_rotation_certificate"])
def test_unfinished_yes_no_answer_is_a_coin_flip(label):
    solved_by_yes = [p["is_solved"] for p in outcomes(label, ScriptedClient(finished("yes")))]
    solved_by_no = [p["is_solved"] for p in outcomes(label, ScriptedClient(finished("no")))]
    assert all(yes != no for yes, no in zip(solved_by_yes, solved_by_no, strict=True))
    flips = [coin_flip(label, seed) for seed in range(12)]
    expected = [
        yes if flip == "yes" else no
        for flip, yes, no in zip(flips, solved_by_yes, solved_by_no, strict=True)
    ]
    assert any(expected)
    assert [
        p["is_solved"] for p in outcomes(label, ScriptedClient(UNFINISHED), "coin_flip")
    ] == expected


def test_unfinished_yes_no_answer_fails_without_the_coin_flip():
    assert not any(
        p["is_solved"] for p in outcomes("tangram_certificate", ScriptedClient(UNFINISHED), "fail")
    )


@pytest.mark.parametrize(
    ("label", "reply"),
    [
        ("tangram_certificate", finished("yes")),
        ("tangram_certificate", finished("I cannot tell")),
        ("tangram", UNFINISHED),
        ("anagram", UNFINISHED),
        ("mental_rotation", finished("rotate(90)\ndone")),
    ],
)
def test_finished_answers_and_other_tasks_ignore_the_setting(label, reply):
    assert outcomes(label, ScriptedClient(reply), "coin_flip") == outcomes(
        label, ScriptedClient(reply), "fail"
    )
