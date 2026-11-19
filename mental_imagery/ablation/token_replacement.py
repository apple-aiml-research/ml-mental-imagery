#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Token-replacement test for the Visual CoT method.

    uv run python -m mental_imagery.ablation.token_replacement checkpoint=outputs/qwen3_5_9b/visual_cot_128/checkpoints/step_025000 mode=ground_truth

Decodes greedily with Hugging Face and, before each generated board token is
committed, overwrites the board with substitute visual tokens; the model's own
actions are kept. A second rollout from the same seeds, stepped with the
written tokens, tracks the true state and which position of a board block is
being generated. The written streams are then scored by the usual replay.

Modes: ``none`` (the model's own boards; control), ``ground_truth`` (the true
board), ``black_frame``, ``uniform`` (uniformly random codes), ``random_history``
(a random earlier true board), ``initial_frame`` (the puzzle's starting board).
Writes ``<output_dir>/results.json``.
"""

from __future__ import annotations

import json
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoModelForImageTextToText

from mental_imagery.eval import score_token_streams, summarize
from mental_imagery.model.families import (
    patch_flash_attention_for_mrope,
    resolve_family,
    resolve_token_id,
    visual_token,
)
from mental_imagery.model.loading import load_checkpoint_processor
from mental_imagery.puzzles import build_engine, build_game_configs, game_key
from mental_imagery.utils.data import build_prompts
from mental_imagery.visual_tokenizers.codec import build_visual_tokenizer

MODES = ("none", "ground_truth", "black_frame", "uniform", "random_history", "initial_frame")
# Values of Rollout.board_slots: the position of each episode within a board block.
BOARD_SLOT_NONE, BOARD_SLOT_OPEN, BOARD_SLOT_CLOSE = -1, -2, -3


class BoardReplacer:
    """Overwrites every generated board block with substitute visual tokens.

    Driven by the board phase of a generation rollout: after each action the
    model closes, the rollout dictates the next block token by token, so every
    block is exactly ``<board>``, N codes, ``</board>``. Call ``override`` after
    sampling each position and ``advance`` with the tokens actually written.
    """

    def __init__(
        self,
        mode,
        seed,
        rollout,
        visual_tokenizer,
        visual_token_ids,
        special_tokens,
        render_resolution,
        episode_seeds,
    ):
        self.mode = mode
        self.rollout = rollout
        self.visual_tokenizer = visual_tokenizer
        self.visual_token_ids = visual_token_ids.cpu()
        self.board_open_id = special_tokens["board_open"]
        self.board_close_id = special_tokens["board_close"]
        self.render_resolution = render_resolution
        self.num_tokens = list(rollout.board_num_tokens)
        self.rngs = [random.Random(seed + episode_seed) for episode_seed in episode_seeds]
        self.history = [[] for _ in episode_seeds]  # true boards of earlier blocks
        self.board_ids = [None] * len(episode_seeds)  # tokens of the block each episode is writing
        self.constant_ids = None
        if mode == "initial_frame":
            self.constant_ids = self.encode(rollout.initial_images, list(range(len(episode_seeds))))
        elif mode == "black_frame":
            black_images = [
                Image.new("RGB", image.size, (0, 0, 0)) for image in rollout.initial_images
            ]
            self.constant_ids = self.encode(black_images, list(range(len(episode_seeds))))

    def encode(self, images, episodes) -> list[torch.Tensor]:
        codes = self.visual_tokenizer.encode(images)
        token_ids = [self.visual_token_ids[episode_codes.cpu()] for episode_codes in codes]
        for episode, ids in zip(episodes, token_ids, strict=True):
            if len(ids) != self.num_tokens[episode]:
                raise ValueError(
                    f"episode {episode}: {len(ids)} codes for a board of {self.num_tokens[episode]} tokens"
                )
        return token_ids

    def start_blocks(self, episodes) -> None:
        """Choose the substitute board of each episode that starts a new block."""
        if self.constant_ids is not None:
            for episode in episodes:
                self.board_ids[episode] = self.constant_ids[episode]
            return
        if self.mode == "uniform":
            for episode in episodes:
                codes = [
                    self.rngs[episode].randrange(len(self.visual_token_ids))
                    for _ in range(self.num_tokens[episode])
                ]
                self.board_ids[episode] = self.visual_token_ids[torch.tensor(codes)]
            return
        current_boards = self.rollout.render(self.render_resolution)
        chosen = []
        for episode in episodes:
            current = current_boards[episode]
            if self.mode == "random_history" and self.history[episode]:
                chosen.append(self.rngs[episode].choice(self.history[episode]))
            else:  # ground truth, or the first block of random_history
                chosen.append(current)
            self.history[episode].append(current)
        for episode, ids in zip(episodes, self.encode(chosen, episodes), strict=True):
            self.board_ids[episode] = ids

    def override(self, next_tokens: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        slots = self.rollout.board_slots
        is_done = done.tolist()
        starting = [
            episode for episode, slot in enumerate(slots) if slot == 0 and not is_done[episode]
        ]
        if starting:
            self.start_blocks(starting)
        for episode, slot in enumerate(slots):
            if slot == BOARD_SLOT_NONE or is_done[episode]:
                continue
            if slot == BOARD_SLOT_OPEN:
                next_tokens[episode] = self.board_open_id
            elif slot == BOARD_SLOT_CLOSE:
                next_tokens[episode] = self.board_close_id
            else:
                next_tokens[episode] = int(self.board_ids[episode][slot])
        return next_tokens

    def advance(self, written_tokens: torch.Tensor) -> None:
        self.rollout.step(written_tokens.cpu().numpy().astype(np.uint32))


@torch.no_grad()
def generate(
    model, batch, gen_start, max_tokens, stop_token_ids, pad_token_id, replacer=None
) -> torch.Tensor:
    """Greedy decoding with a KV cache; returns the (B, max_tokens - gen_start) generated tokens."""
    seq, attention_mask = batch["input_ids"], batch["attention_mask"]
    done = torch.zeros(seq.shape[0], dtype=torch.bool, device=seq.device)
    stop_token_ids = torch.tensor(stop_token_ids, device=seq.device)
    past_key_values, rope_deltas = None, None
    for position in range(gen_start, max_tokens):
        if past_key_values is None:
            prompt_inputs = {
                key: tensor[:, :gen_start] if tensor.shape == seq.shape else tensor
                for key, tensor in batch.items()
            }
            outputs = model(**prompt_inputs, use_cache=True)
            rope_deltas = (
                model.model.rope_deltas
            )  # Qwen 3.5 mRoPE offsets of the text after the image
        else:
            text_positions = attention_mask[:, :position].long().sum(dim=1) - 1
            position_ids = text_positions.view(1, -1, 1).expand(3, -1, 1) + rope_deltas
            outputs = model(
                input_ids=seq[:, position - 1 : position],
                attention_mask=attention_mask[:, :position],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1].argmax(dim=-1)
        if replacer is not None:
            next_tokens = replacer.override(next_tokens, done)
        written = torch.where(done, torch.full_like(next_tokens, pad_token_id), next_tokens)
        seq[:, position] = written
        attention_mask[:, position] = True
        if replacer is not None:
            replacer.advance(written)
        done |= torch.isin(next_tokens, stop_token_ids)
        if done.all():
            break
    return seq[:, gen_start:]


def evaluate_game(
    entry,
    engine,
    model,
    processor,
    family,
    visual_tokenizer,
    visual_token_ids,
    config,
    render_resolution,
):
    loader = engine.loader(games=build_game_configs([entry]), n=config.batch_size, prefetch=0)
    special = engine.special_tokens
    stop_token_ids = [special["eos"], special["im_end"]]
    puzzles = []
    for start in range(0, config.num_puzzles, config.batch_size):
        seeds = list(range(start, min(start + config.batch_size, config.num_puzzles)))
        rollout = loader.next_with_seeds(seeds=seeds)
        optimal_lengths = rollout.solution_length()
        batch = build_prompts(processor, rollout, config.max_tokens, model.device, family)
        gen_start = int(batch.pop("prompt_lengths").max())
        replacer = None
        if config.mode != "none":
            replacer = BoardReplacer(
                mode=config.mode,
                seed=config.seed,
                rollout=loader.next_with_seeds(seeds=seeds),
                visual_tokenizer=visual_tokenizer,
                visual_token_ids=visual_token_ids,
                special_tokens=special,
                render_resolution=render_resolution,
                episode_seeds=seeds,
            )
        generated = generate(
            model,
            batch,
            gen_start,
            config.max_tokens,
            stop_token_ids,
            pad_token_id=resolve_token_id(processor.tokenizer, family.pad_token),
            replacer=replacer,
        )
        outcomes = score_token_streams(rollout, generated.cpu().numpy().astype(np.uint32), special)
        for seed, optimal_length, outcome in zip(seeds, optimal_lengths, outcomes, strict=True):
            puzzles.append({"seed": seed, "optimal_length": optimal_length, **outcome})
        print(
            f"[token_replacement] {game_key(entry)} ({config.mode}): {len(puzzles)}/{config.num_puzzles} done, "
            f"{sum(p['is_solved'] for p in puzzles)} solved",
            flush=True,
        )
    return puzzles


@hydra.main(config_path="../configs", config_name="token_replacement", version_base=None)
def main(config) -> None:
    if config.mode not in MODES:
        raise ValueError(f"mode={config.mode!r} is not one of {MODES}")
    checkpoint = config.checkpoint
    train_config = OmegaConf.load(os.path.join(checkpoint, "config.yaml"))
    if not train_config.visual_tokens.enabled:
        raise ValueError("the token-replacement test needs a Visual CoT checkpoint")
    family = resolve_family(train_config.model.family)
    if not family.uses_mrope:
        raise ValueError(
            f"the token-replacement decoding passes Qwen 3.5 mRoPE positions; "
            f"{family.name} models are not supported"
        )
    with open(os.path.join(checkpoint, "training_state.json")) as f:
        step = json.load(f)["step"]
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda")
    patch_flash_attention_for_mrope()
    processor = load_checkpoint_processor(checkpoint)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            attn_implementation=train_config.model.attn_implementation,
        )
        .to(device)
        .eval()
    )
    engine = build_engine(
        processor.tokenizer,
        family=family,
        engine_config=train_config.engine,
        visual_tokens_config=train_config.visual_tokens,
        seed=0,
        parallel=False,
    )
    visual_tokenizer = build_visual_tokenizer(
        train_config.visual_tokens, device, train_config.visual_tokens.grid_size
    )
    vocab = processor.tokenizer.get_vocab()
    visual_token_ids = torch.tensor(
        [vocab[visual_token(code)] for code in range(train_config.visual_tokens.codebook_size)]
    )

    games = {
        game_key(entry): summarize(
            evaluate_game(
                entry,
                engine,
                model,
                processor,
                family,
                visual_tokenizer,
                visual_token_ids,
                config,
                train_config.engine.render_resolution,
            )
        )
        for entry in config.games
    }
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
    print(f"\n{train_config.run_name} @ step {step}, boards: {config.mode}")
    for key, summary in games.items():
        print(f"  {key:32s} solve_rate={summary['solve_rate']:.3f}")
    print(f"  {'average':32s} solve_rate={results['avg_solve_rate']:.3f}")


if __name__ == "__main__":
    main()
