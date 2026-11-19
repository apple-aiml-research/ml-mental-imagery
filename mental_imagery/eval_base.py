#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Zero-shot gameplay evaluation of an untrained base model.

    uv run python -m mental_imagery.eval_base model=qwen3_5_9b

Unlike ``eval``, which replays a trained model's raw ``<action>`` tokens, this
evaluates a model served from the Hub at its pinned revision (no run config):

- prompt: the training prompt plus a per-task instruction on how to answer;
- output: the text after the reasoning span is parsed into actions and
  rewritten as an ``<action>`` stream (unrecognized actions stay as no-ops).

The rewritten stream is scored by ``eval.score_token_streams`` with the same
puzzles (seed i for puzzle i), action limit and success criteria. The prompt,
reasoning delimiter and stop tokens come from ``model.family``. Writes
``<output_dir>/results.json``.

A yes/no answer whose reasoning never ended (the response budget ran out) is a
reproducible coin flip with ``unfinished_yes_no=coin_flip`` (the default) and a
fail with ``unfinished_yes_no=fail``.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re

import hydra
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoConfig

from mental_imagery.eval import score_token_streams, summarize
from mental_imagery.model.families import resolve_family, resolve_token_id, user_message_text
from mental_imagery.model.loading import load_processor
from mental_imagery.puzzles import build_engine, build_game_configs, game_key
from mental_imagery.utils.inference import VLLMClient, VLLMServer, chat_messages, visible_device_ids

DEFAULT_INSTRUCTION = (
    "Provide the complete sequence of actions needed to solve the puzzle, one action per line. "
    "Each action must be in the action grammar above. Output only the actions, no explanation."
)
YES_NO_INSTRUCTION = (
    "This puzzle is a yes/no question about whether it can be solved. You may list any helper "
    "actions from the action grammar above as intermediate steps, but the FINAL line of your output "
    "must be exactly `yes` or `no`, with no other text on that line."
)
ANAGRAM_INSTRUCTION = (
    "The letters shown are a scrambled English word. Work out what the word is. You may list "
    "swap(i, j) actions as intermediate steps if helpful, but the FINAL line of your output must be "
    "exactly `identify('word')` — where `word` is the unscrambled English word. Do not put anything "
    "else on that final line."
)
CHARACTER_INSTRUCTION = (
    "The pieces shown form a single character when assembled (a letter, digit, or Greek letter). "
    "Work out what the character is. You may list rotatecw / rotateccw / place / undo actions as "
    "intermediate steps if helpful, but the FINAL line of your output must be exactly "
    "`identify('NAME')` — where NAME is the English name of the character, e.g. `identify('A')`, "
    "`identify('7')`, or `identify('psi')` for ψ. Do not put anything else on that final line."
)
TANGRAM_INSTRUCTION = (
    "Fill the gray silhouette using the colored pieces, labeled `P0`, `P1`, `P2`, .... Use "
    "`place('P0', x, y)` (substituting the piece label you want to place) where x is the column and "
    "y is the row, both 0-indexed from the top-left of the canvas. Use `rotatecw('P0')` or "
    "`rotateccw('P0')` to rotate a piece 90° before placing it. Output one action per line; the full "
    "sequence should cover every silhouette cell with exactly one piece each. Output only the "
    "actions, no explanation."
)

# Function-call actions such as place('P0', 1, 2), or single words such as up.
CALL_OR_WORD = re.compile(r"[A-Za-z_]\w*\s*\([^()]*\)|[A-Za-z_]\w*")
WORD_ACTIONS = {"up", "down", "left", "right", "back", "reset", "done", "yes", "no"}
TERMINAL_ACTIONS = {"done", "yes", "no"}
# Written in place of an action that matches no valid action: it is replayed as
# a malformed action (a no-op that still counts towards the action limit).
INVALID_ACTION = "?"


def instruction_for(entry) -> str:
    if entry.name == "anagram":
        return ANAGRAM_INSTRUCTION
    if entry.name == "character_recognition":
        return CHARACTER_INSTRUCTION
    if entry.params.get("certificate_game"):
        return YES_NO_INSTRUCTION
    if entry.name == "tangram":
        return TANGRAM_INSTRUCTION
    return DEFAULT_INSTRUCTION


def answer_text(response: str, reasoning_end_token: str) -> str:
    """The reply after the reasoning span; empty if the reasoning never ended (budget exhausted)."""
    return response.rsplit(reasoning_end_token, 1)[1] if reasoning_end_token in response else ""


def coin_flip(game: str, seed: int) -> str:
    """A reproducible random yes/no answer for one puzzle."""
    return random.Random(f"{game}:{seed}").choice(["yes", "no"])


def action_key(text: str) -> str:
    """Spelling-insensitive form of an action: no spaces, quotes or other punctuation."""
    return re.sub(r"[^A-Za-z0-9_(),\-]", "", text)


def parse_actions(text: str, valid_actions: list[str], is_yes_no_task: bool) -> list[str]:
    """Actions in ``text``, in order, mapped to their exact engine spelling.

    Call-form actions that match no valid action become ``INVALID_ACTION``.
    Terminal answers are removed from the sequence; the episode ends with the
    last ``yes``/``no`` in the answer for yes/no tasks, and with ``done`` otherwise.
    """
    by_key = {action_key(action): action for action in valid_actions}
    by_folded_key: dict[str, list[str]] = {}
    for action in valid_actions:
        by_folded_key.setdefault(action_key(action).lower(), []).append(action)

    actions, last_answer = [], None
    for match in CALL_OR_WORD.finditer(text):
        candidate = match.group(0)
        if "(" not in candidate:
            word = candidate.lower()
            if word in ("yes", "no"):
                last_answer = word
            if word in WORD_ACTIONS and word not in TERMINAL_ACTIONS:
                actions.append(word)
            continue
        key = action_key(candidate)
        folded_matches = by_folded_key.get(key.lower(), [])
        if key in by_key:
            actions.append(by_key[key])
        elif len(folded_matches) == 1:
            actions.append(folded_matches[0])  # e.g. identify('Church') for identify('church')
        else:
            actions.append(INVALID_ACTION)
    if is_yes_no_task:
        if last_answer is not None:
            actions.append(last_answer)
    else:
        actions.append("done")
    return actions


def action_stream(actions: list[str], turn_end_token: str) -> str:
    """The actions in the trained models' output format, ending the turn."""
    return "".join(f"<action>{action}</action>" for action in actions) + turn_end_token


def evaluate_game(entry, engine, tokenizer, family, client, config, generations_file) -> list[dict]:
    loader = engine.loader(games=build_game_configs([entry]), n=config.batch_size, prefetch=0)
    is_yes_no_task = bool(entry.params.get("certificate_game"))
    flip_unfinished = is_yes_no_task and config.unfinished_yes_no == "coin_flip"
    instruction = instruction_for(entry)
    puzzles = []
    for start in range(0, config.num_puzzles, config.batch_size):
        seeds = list(range(start, min(start + config.batch_size, config.num_puzzles)))
        rollout = loader.next_with_seeds(seeds=seeds)
        optimal_lengths = rollout.solution_length()
        valid_actions = rollout.valid_actions
        messages = [
            chat_messages(image, user_message_text(prompt, actions) + "\n\n" + instruction)
            for image, prompt, actions in zip(
                rollout.initial_images, rollout.prompts, rollout.actions, strict=True
            )
        ]
        responses = client.generate_batch(messages, config.max_new_tokens)

        streams = []
        for seed, response, episode_actions in zip(seeds, responses, valid_actions, strict=True):
            if flip_unfinished and family.think_close_token not in response:
                parsed = [
                    coin_flip(game_key(entry), seed)
                ]  # the reasoning never ended: no answer to read
            else:
                parsed = parse_actions(
                    answer_text(response, family.think_close_token), episode_actions, is_yes_no_task
                )
            streams.append(
                tokenizer.encode(
                    action_stream(parsed, family.turn_end_token), add_special_tokens=False
                )
            )
        tokens = np.zeros((len(seeds), max(len(stream) for stream in streams)), dtype=np.uint32)
        for row, stream in enumerate(streams):
            tokens[row, : len(stream)] = stream
        outcomes = score_token_streams(rollout, tokens, engine.special_tokens)
        for seed, optimal_length, outcome, response in zip(
            seeds, optimal_lengths, outcomes, responses, strict=True
        ):
            puzzles.append({"seed": seed, "optimal_length": optimal_length, **outcome})
            if generations_file:
                generations_file.write(
                    json.dumps({"game": game_key(entry), "seed": seed, "response": response}) + "\n"
                )
        print(
            f"[eval_base] {game_key(entry)}: {len(puzzles)}/{config.num_puzzles} done, "
            f"{sum(p['is_solved'] for p in puzzles)} solved",
            flush=True,
        )
    return puzzles


@hydra.main(config_path="configs", config_name="eval_base", version_base=None)
def main(config) -> None:
    keys = [game_key(entry) for entry in config.games]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate game keys in {keys}; give repeated games distinct labels")
    if config.unfinished_yes_no not in ("coin_flip", "fail"):
        raise ValueError(
            f"unfinished_yes_no must be coin_flip or fail, got {config.unfinished_yes_no!r}"
        )
    family = resolve_family(config.model.family)
    family.check_model_type(
        AutoConfig.from_pretrained(config.model.name, revision=config.model.revision)
    )
    os.makedirs(config.output_dir, exist_ok=True)

    # The engine replays the parsed actions with the trained models' structural tokens;
    # the base model itself is served unchanged.
    processor = load_processor(
        config.model.name, config.model.revision, OmegaConf.create({"enabled": False})
    )
    tokenizer = processor.tokenizer
    engine = build_engine(
        tokenizer,
        family=family,
        engine_config=config.engine,
        visual_tokens_config=OmegaConf.create({"enabled": False}),
        seed=0,
        parallel=False,
    )
    server = VLLMServer(
        config.model.name,
        visible_device_ids(config.vllm.gpus),
        config.vllm.port,
        config.max_tokens,
        config.vllm.gpu_memory_utilization,
        config.vllm.enforce_eager,
        revision=config.model.revision,
        chat_template=family.vllm_chat_template_path(),
        host=config.vllm.host,
    )
    generations_path = os.path.join(config.output_dir, "generations.jsonl")
    games = {}
    try:
        server.start()
        stop_token_ids = [
            resolve_token_id(tokenizer, token)
            for token in (family.eos_token, family.turn_end_token)
        ]
        client = VLLMClient(config.vllm.port, config.model.name, stop_token_ids)
        with (
            open(generations_path, "w")
            if config.save_generations
            else contextlib.nullcontext() as generations_file
        ):
            for entry in config.games:
                games[game_key(entry)] = summarize(
                    evaluate_game(
                        entry, engine, tokenizer, family, client, config, generations_file
                    )
                )
    finally:
        server.stop()

    results = {
        "model": config.model.name,
        "run_name": f"{config.model.short_name}/base",
        "eval_config": OmegaConf.to_container(config, resolve=True),
        "avg_solve_rate": sum(g["solve_rate"] for g in games.values()) / len(games),
        "games": games,
    }
    with open(os.path.join(config.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{config.model.name} (base, zero-shot)")
    for key, summary in games.items():
        print(f"  {key:32s} solve_rate={summary['solve_rate']:.3f}")
    print(f"  {'average':32s} solve_rate={results['avg_solve_rate']:.3f}")


if __name__ == "__main__":
    main()
