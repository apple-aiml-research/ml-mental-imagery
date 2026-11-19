#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Evaluate a checkpoint on the benchmark with vLLM.

    uv run python -m mental_imagery.eval checkpoint=outputs/qwen3_5_9b/action_supervision/checkpoints/step_025000

Puzzle i of every task uses seed i (training uses seeds >= 10000). Each greedy
response is replayed token by token in the engine, which executes its actions
and decides whether the puzzle was solved; a question-answering task counts as
solved when its final answer is right. Method settings come from the
checkpoint's config.yaml. Writes ``<output_dir>/results.json``.
"""

from __future__ import annotations

import contextlib
import json
import os

import hydra
import numpy as np
from omegaconf import OmegaConf

from mental_imagery.model.families import resolve_family, user_message_text
from mental_imagery.model.loading import load_checkpoint_processor
from mental_imagery.puzzles import build_engine, build_game_configs, game_key, text_boards_of
from mental_imagery.utils.data import build_prompts
from mental_imagery.utils.inference import VLLMClient, VLLMServer, chat_messages, visible_device_ids


def score_token_streams(rollout, tokens: np.ndarray, special_tokens) -> list[dict]:
    """Replay (B, T) zero-padded token streams in the engine, executing every token.

    An episode ends at its terminal answer, at the end of the turn, or at the
    action limit (``is_timed_out``); ``is_truncated`` means it never ended.
    """
    num_actions = np.zeros(len(tokens), dtype=int)
    done = np.zeros(len(tokens), dtype=bool)
    for position in range(tokens.shape[1]):
        column = np.ascontiguousarray(tokens[:, position])
        num_actions += (column == special_tokens["action_close"]) & ~done
        done = rollout.step(column)
        if done.all():
            break
    return [
        {
            "is_solved": solved,
            "is_timed_out": timed_out,
            "is_truncated": not finished,
            "num_actions": int(actions),
        }
        for solved, timed_out, finished, actions in zip(
            rollout.is_solved, rollout.timed_out, done, num_actions, strict=True
        )
    ]


def evaluate_game(entry, engine, processor, family, client, config, generations_file) -> list[dict]:
    loader = engine.loader(games=build_game_configs([entry]), n=config.batch_size, prefetch=0)
    puzzles = []
    for start in range(0, config.num_puzzles, config.batch_size):
        seeds = list(range(start, min(start + config.batch_size, config.num_puzzles)))
        rollout = loader.next_with_seeds(seeds=seeds)
        optimal_lengths = rollout.solution_length()
        prompt_length = int(
            build_prompts(processor, rollout, config.max_tokens, "cpu", family)[
                "prompt_lengths"
            ].max()
        )
        max_new_tokens = config.max_tokens - prompt_length
        messages = [
            chat_messages(image, user_message_text(prompt, actions))
            for image, prompt, actions in zip(
                rollout.initial_images, rollout.prompts, rollout.actions, strict=True
            )
        ]
        responses = client.generate_batch(messages, max_new_tokens)

        tokens = np.zeros((len(seeds), max_new_tokens), dtype=np.uint32)
        for row, text in enumerate(responses):
            ids = processor.tokenizer.encode(text, add_special_tokens=False)[:max_new_tokens]
            tokens[row, : len(ids)] = ids
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
            f"[eval] {game_key(entry)}: {len(puzzles)}/{config.num_puzzles} done, "
            f"{sum(p['is_solved'] for p in puzzles)} solved",
            flush=True,
        )
    return puzzles


def summarize(puzzles: list[dict]) -> dict:
    n = len(puzzles)
    return {
        "solve_rate": sum(p["is_solved"] for p in puzzles) / n,
        "timeout_rate": sum(p["is_timed_out"] for p in puzzles) / n,
        "truncation_rate": sum(p["is_truncated"] for p in puzzles) / n,
        "avg_actions": sum(p["num_actions"] for p in puzzles) / n,
        "num_puzzles": n,
        "puzzles": puzzles,
    }


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(config) -> None:
    checkpoint = config.checkpoint
    train_config = OmegaConf.load(os.path.join(checkpoint, "config.yaml"))
    with open(os.path.join(checkpoint, "training_state.json")) as f:
        step = json.load(f)["step"]
    keys = [game_key(entry) for entry in config.games]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate game keys in {keys}; give repeated games distinct labels")
    os.makedirs(config.output_dir, exist_ok=True)

    family = resolve_family(train_config.model.family)
    processor = load_checkpoint_processor(checkpoint)
    text_boards = text_boards_of(train_config)
    engine = build_engine(
        processor.tokenizer,
        family=family,
        engine_config=train_config.engine,
        visual_tokens_config=train_config.visual_tokens,
        seed=0,
        parallel=False,
        text_boards=text_boards,
    )
    server = VLLMServer(
        checkpoint,
        visible_device_ids(config.vllm.gpus),
        config.vllm.port,
        config.max_tokens,
        config.vllm.gpu_memory_utilization,
        config.vllm.enforce_eager,
        chat_template=family.vllm_chat_template_path(),
        host=config.vllm.host,
    )
    generations_path = os.path.join(config.output_dir, "generations.jsonl")
    games = {}
    try:
        server.start()
        client = VLLMClient(
            config.vllm.port,
            checkpoint,
            [engine.special_tokens["eos"], engine.special_tokens["im_end"]],
        )
        with (
            open(generations_path, "w")
            if config.save_generations
            else contextlib.nullcontext() as generations_file
        ):
            for entry in config.games:
                games[game_key(entry)] = summarize(
                    evaluate_game(
                        entry, engine, processor, family, client, config, generations_file
                    )
                )
    finally:
        server.stop()

    results = {
        "checkpoint": os.path.abspath(checkpoint),
        "run_name": train_config.run_name,
        "step": step,
        "eval_config": OmegaConf.to_container(config, resolve=True),
        "avg_solve_rate": sum(g["solve_rate"] for g in games.values()) / len(games),
        "games": games,
    }
    with open(os.path.join(config.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{train_config.run_name} @ step {step}")
    for key, summary in games.items():
        print(f"  {key:32s} solve_rate={summary['solve_rate']:.3f}")
    print(f"  {'average':32s} solve_rate={results['avg_solve_rate']:.3f}")


if __name__ == "__main__":
    main()
