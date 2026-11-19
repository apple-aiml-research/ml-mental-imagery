#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Render the boards a checkpoint imagines in its evaluation episodes.

    uv run python -m mental_imagery.visualize checkpoint=outputs/qwen3_5_9b/visual_cot_64/checkpoints/step_025000

Needs the responses saved by ``mental_imagery.eval`` with ``save_generations=true``.
Each response is replayed in the engine, and every action is drawn as the true
board after it (engine render) above the board the model imagined:

- Visual CoT: the true board encoded and decoded by the run's board tokenizer
  (LlamaGen VQ-16, or the FSQ autoencoder), then the board the model wrote next
  as visual tokens, decoded by the same tokenizer.
- Visual head (Visual Supervision, and Action Supervision with a stop-gradient
  probe): the head's prediction from the hidden states at the action's
  ``</action>``, read by one teacher-forced pass over the prompt and the
  model's response, as in training, and decoded by the FSQ decoder.

Writes ``<output_dir>/<game>_seed<seed>.png`` per episode and ``index.html``.
"""

from __future__ import annotations

import html
import json
import os
import textwrap
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForImageTextToText

from mental_imagery.model.families import (
    patch_flash_attention_for_mrope,
    resolve_family,
    resolve_token_id,
    visual_token,
)
from mental_imagery.model.loading import load_checkpoint_processor
from mental_imagery.model.visual_head import HiddenStateCollector, build_visual_head
from mental_imagery.puzzles import build_engine, build_game_configs, game_key, text_boards_of
from mental_imagery.utils.data import process_prompts, stage_prompts
from mental_imagery.visual_tokenizers.codec import build_visual_tokenizer, code_grid_size

TILE = 160  # pixel height of a board in the figure
CAPTION = 20  # pixel height of the action caption under each column
LABEL_WIDTH = 120  # pixel width of the row labels left of the boards
COLUMNS = 10  # columns per figure row
FONT_SIZE = 13


@dataclass
class Step:
    """The start of an episode, or one action and the board after it."""

    true_board: Image.Image  # engine render after the action (the start: the initial board)
    action: str | None = None  # None for the start
    position: int | None = None  # index of the action's </action> in the response tokens
    board_tokens: list[int] | None = (
        None  # Visual CoT: the board block written next (possibly incomplete)
    )


@dataclass
class Column:
    boards: list[Image.Image | str]  # one per figure row: an image, or why there is none
    caption: str


def imagination_method(train_config) -> str:
    """``"visual_cot"`` or ``"visual_head"``: how the checkpoint's imagined boards are read."""
    if train_config.visual_tokens.enabled:
        return "visual_cot"
    if train_config.visual_head.enabled:
        return "visual_head"
    raise ValueError(
        f"{train_config.run_name} neither writes visual tokens (Visual CoT) nor has a visual head "
        "(Visual Supervision, visual probe): it has no imagined boards to render"
    )


def load_episodes(eval_dir: str, num_episodes: int | None = None) -> list[dict]:
    """Evaluated episodes (the lowest ``num_episodes`` seeds per game, or all): game entry, seed, response, outcome."""
    generations_path = os.path.join(eval_dir, "generations.jsonl")
    if not os.path.exists(generations_path):
        raise FileNotFoundError(
            f"{generations_path} is missing: evaluate the checkpoint with save_generations=true"
        )
    with open(os.path.join(eval_dir, "results.json")) as f:
        results = json.load(f)
    responses = {}
    with open(generations_path) as f:
        for line in f:
            row = json.loads(line)
            responses[(row["game"], row["seed"])] = row["response"]
    episodes = []
    for entry in OmegaConf.create(results["eval_config"]["games"]):
        game = game_key(entry)
        puzzles = sorted(results["games"][game]["puzzles"], key=lambda puzzle: puzzle["seed"])
        for puzzle in puzzles[:num_episodes]:
            episodes.append(
                {
                    "entry": entry,
                    "seed": puzzle["seed"],
                    "response": responses[(game, puzzle["seed"])],
                    "outcome": puzzle,
                }
            )
    return episodes


def replay(
    rollout, token_ids, special_tokens, render_resolution: int, decode
) -> tuple[list[Step], bool]:
    """Step a one-episode rollout through a response; returns its steps and whether it was solved.

    An action takes effect when its ``</action>`` is executed, so the board rendered right after
    is the board after the action: the visual head's target at that position. ``decode`` maps an
    action's token ids to its text.
    """
    steps = [Step(true_board=rollout.render(render_resolution)[0])]
    action_ids, in_action, board = [], False, None
    for position, token in enumerate(token_ids):
        if token == special_tokens["board_open"]:
            board = steps[-1].board_tokens = []
        elif token == special_tokens["board_close"]:
            board = None
        elif board is not None:
            board.append(token)
        elif token == special_tokens["action_open"]:
            action_ids, in_action = [], True
        elif in_action and token != special_tokens["action_close"]:
            action_ids.append(token)
        done = rollout.step(np.array([token], dtype=np.uint32))[0]
        if token == special_tokens["action_close"]:
            steps.append(
                Step(
                    true_board=rollout.render(render_resolution)[0],
                    action=decode(action_ids),
                    position=position,
                )
            )
            action_ids, in_action = [], False
        if done:
            break
    return steps, bool(rollout.is_solved[0])


def decode_written_boards(
    steps: list[Step], visual_tokenizer, code_of_token: dict, grid_size: int
) -> list:
    """Visual CoT: each step's written board as an image, or why there is none."""
    imagined = []
    for step in steps:
        grid_w, grid_h = code_grid_size(*step.true_board.size, grid_size)
        tokens = step.board_tokens
        if tokens is None:
            imagined.append("no board written")
        elif len(tokens) != grid_w * grid_h or any(token not in code_of_token for token in tokens):
            num_visual = sum(token in code_of_token for token in tokens)
            imagined.append(
                f"malformed board:\n{len(tokens)} tokens, {num_visual} visual\n"
                f"(expected {grid_w * grid_h} visual)"
            )
        else:
            codes = [code_of_token[token] for token in tokens]
            imagined.append(to_pil_image(visual_tokenizer.decode(codes, grid_w, grid_h)))
    return imagined


def tokenized_true_boards(steps: list[Step], visual_tokenizer, grid_size: int) -> list:
    """Visual CoT: each true board encoded and decoded by the board tokenizer (the best board the model can write)."""
    tokenized = []
    for step in steps:
        grid_w, grid_h = code_grid_size(*step.true_board.size, grid_size)
        codes = visual_tokenizer.encode([step.true_board.convert("RGB")])[0]
        tokenized.append(to_pil_image(visual_tokenizer.decode(codes, grid_w, grid_h)))
    return tokenized


@torch.no_grad()
def predict_boards(
    model, head, collector, processor, family, rollout, token_ids, steps: list[Step], device
) -> list:
    """Visual head: its board prediction at each action's ``</action>``, from one teacher-forced pass.

    The sequence is built as in training (``mental_imagery/utils/data.py``): the prompt, then the response tokens.
    """
    imagined = ["no prediction\nbefore the first action"]
    positions = [step.position for step in steps[1:]]
    if not positions:
        return imagined
    inputs = process_prompts(processor, rollout, family)
    prompt_length = inputs["input_ids"].shape[1]
    response = torch.tensor(token_ids[: positions[-1] + 1], dtype=torch.long)
    batch = stage_prompts(inputs, prompt_length + len(response), device)
    batch.pop("prompt_lengths")
    batch["input_ids"][0, prompt_length:] = response.to(device)
    batch["attention_mask"][0, prompt_length:] = True
    mask = torch.zeros_like(batch["attention_mask"])
    mask[0, [prompt_length + position for position in positions]] = True
    collector.prepare(mask)
    model(**batch, use_cache=False)
    hidden = collector.collect()  # (layers, actions, hidden)
    predictions = head.predict(hidden.transpose(0, 1).float())
    return imagined + [to_pil_image(image.cpu()) for image in predictions]


def placeholder(text: str, size: tuple[int, int], font) -> Image.Image:
    image = Image.new("RGB", size, (235, 235, 235))
    ImageDraw.Draw(image).multiline_text((8, size[1] // 2 - 24), text, fill=(150, 0, 0), font=font)
    return image


def fit_text(draw, text: str, width: int, font) -> str:
    """``text`` shortened with an ellipsis to fit ``width`` pixels."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def episode_figure(
    title_lines: list[str], row_labels: list[str], columns: list[Column]
) -> Image.Image:
    """Columns left to right, wrapped every ``COLUMNS``: one board per row label, then the caption."""
    font = ImageFont.load_default(size=FONT_SIZE)
    first = columns[0].boards[0]
    tile = (max(1, round(TILE * first.width / first.height)), TILE)
    header = 8 + (FONT_SIZE + 6) * len(title_lines)
    row_height = len(row_labels) * TILE + CAPTION
    num_rows = (len(columns) + COLUMNS - 1) // COLUMNS
    width = LABEL_WIDTH + min(len(columns), COLUMNS) * (tile[0] + 4)
    figure = Image.new("RGB", (width, header + num_rows * row_height), "white")
    draw = ImageDraw.Draw(figure)
    for line, text in enumerate(title_lines):
        draw.text((6, 4 + line * (FONT_SIZE + 6)), text, fill="black", font=font)
    for row in range(num_rows):
        for board_row, label in enumerate(row_labels):
            y = header + row * row_height + board_row * TILE + TILE // 2
            text = "\n".join(textwrap.wrap(label, width=16))
            draw.multiline_text((6, y), text, fill="black", font=font, anchor="lm")
    for index, column in enumerate(columns):
        x = LABEL_WIDTH + (index % COLUMNS) * (tile[0] + 4)
        y = header + (index // COLUMNS) * row_height
        for board_row, board in enumerate(column.boards):
            if isinstance(board, str):
                image = placeholder(board, tile, font)
            else:  # small decodes (e.g. 64x64 pixels) are enlarged without smoothing
                resample = Image.NEAREST if board.height < TILE else Image.BILINEAR
                image = board.convert("RGB").resize(tile, resample)
            figure.paste(image, (x, y + board_row * TILE))
        caption_y = y + len(row_labels) * TILE + 3
        draw.text(
            (x + 3, caption_y),
            fit_text(draw, column.caption, tile[0] - 6, font),
            fill="black",
            font=font,
        )
    return figure


def episode_status(outcome: dict) -> str:
    if outcome["is_solved"]:
        return "solved"
    if outcome["is_timed_out"]:
        return "failed (action limit)"
    if outcome["is_truncated"]:
        return "failed (no answer)"
    return "failed"


def write_index(path: str, title: str, legend: str, pages: list[dict]) -> None:
    """``index.html``: per game, a link and the figure of each episode with its outcome."""
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title></head>",
        "<body style='font-family: -apple-system, Helvetica, sans-serif'>",
        f"<h2>{html.escape(title)}</h2>",
        f"<p>{html.escape(legend)}</p>",
    ]
    game = None
    for page in pages:
        if page["game"] != game:
            game = page["game"]
            lines.append(f"<h3>{html.escape(game)}</h3>")
        color = "green" if page["status"] == "solved" else "firebrick"
        lines.append(
            f"<p><a href='{html.escape(page['file'])}'>seed {page['seed']}</a>: "
            f"<span style='color:{color}'>{html.escape(page['status'])}</span></p>"
        )
        lines.append(
            f"<img src='{html.escape(page['file'])}' loading='lazy' style='max-width:100%'>"
        )
    lines.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


@hydra.main(config_path="configs", config_name="visualize", version_base=None)
def main(config) -> None:
    checkpoint = config.checkpoint
    train_config = OmegaConf.load(os.path.join(checkpoint, "config.yaml"))
    method = imagination_method(train_config)
    with open(os.path.join(checkpoint, "training_state.json")) as f:
        step = json.load(f)["step"]
    episodes = load_episodes(os.path.join(checkpoint, "eval", "benchmark"), config.num_episodes)
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda")
    family = resolve_family(train_config.model.family)
    processor = load_checkpoint_processor(checkpoint)
    tokenizer = processor.tokenizer
    engine = build_engine(
        tokenizer,
        family=family,
        engine_config=train_config.engine,
        visual_tokens_config=train_config.visual_tokens,
        seed=0,
        parallel=False,
        text_boards=text_boards_of(train_config),
    )
    special_tokens = engine.special_tokens
    render_resolution = train_config.engine.render_resolution
    if method == "visual_cot":
        visual_tokens = train_config.visual_tokens
        visual_tokenizer = build_visual_tokenizer(visual_tokens, device, visual_tokens.grid_size)
        code_of_token = {
            resolve_token_id(tokenizer, visual_token(code)): code
            for code in range(visual_tokens.codebook_size)
        }
        tokenizer_name = "LlamaGen VQ-16" if visual_tokens.tokenizer == "llamagen" else "FSQ"
        row_labels = ["True board", f"True board through {tokenizer_name}", "Model's board"]
        legend = (
            f"True board: the board after each action (engine). True board through {tokenizer_name}: that "
            f"board encoded and decoded with {visual_tokens.grid_size} visual tokens per short side. "
            f"Model's board: the board the model wrote next, decoded with {tokenizer_name}."
        )
    else:
        if family.uses_mrope:
            patch_flash_attention_for_mrope()
        model = (
            AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                dtype=torch.bfloat16,
                attn_implementation=train_config.model.attn_implementation,
            )
            .to(device)
            .eval()
        )
        text_config = model.config.text_config
        head = build_visual_head(
            train_config, text_config.hidden_size, text_config.num_hidden_layers, torch.float32
        )
        head.load_trainable_state_dict(
            torch.load(os.path.join(checkpoint, "visual_head.pt"), weights_only=True)
        )
        head.to(device).eval()
        collector = HiddenStateCollector(model.model.language_model.layers)
        row_labels = ["True board", "Visual head prediction"]
        legend = (
            "True board: the board after each action (engine). Visual head prediction: the head's prediction "
            "of that board from the hidden states at the action's </action>, decoded with the FSQ decoder."
        )

    loaders = {}
    pages = []
    for episode in episodes:
        entry, seed, outcome = episode["entry"], episode["seed"], episode["outcome"]
        game = game_key(entry)
        if game not in loaders:
            loaders[game] = engine.loader(games=build_game_configs([entry]), n=1, prefetch=0)
        rollout = loaders[game].next_with_seeds(seeds=[seed])
        token_ids = tokenizer.encode(episode["response"], add_special_tokens=False)
        steps, solved = replay(
            rollout, token_ids, special_tokens, render_resolution, tokenizer.decode
        )
        if solved != outcome["is_solved"]:
            print(
                f"[visualize] warning: {game} seed {seed} replays as solved={solved}, "
                f"results.json has solved={outcome['is_solved']}",
                flush=True,
            )
        rows = [[step.true_board for step in steps]]
        if method == "visual_cot":
            rows.append(tokenized_true_boards(steps, visual_tokenizer, visual_tokens.grid_size))
            rows.append(
                decode_written_boards(
                    steps, visual_tokenizer, code_of_token, visual_tokens.grid_size
                )
            )
        else:  # the replay does not change the rollout's prompt
            rows.append(
                predict_boards(
                    model, head, collector, processor, family, rollout, token_ids, steps, device
                )
            )
        columns = [
            Column(list(boards), "start" if step.action is None else f"{index}: {step.action}")
            for index, (step, *boards) in enumerate(zip(steps, *rows, strict=True))
        ]
        status = episode_status(outcome)
        title_lines = [
            f"{game} seed {seed}: {status}, {outcome['num_actions']} actions "
            f"(optimal {outcome['optimal_length']}). {train_config.run_name} step {step}",
            legend,
        ]
        file_name = f"{game}_seed{seed:03d}.png"
        episode_figure(title_lines, row_labels, columns).save(
            os.path.join(config.output_dir, file_name)
        )
        pages.append({"game": game, "seed": seed, "status": status, "file": file_name})
        print(f"[visualize] {file_name}: {status}", flush=True)

    write_index(
        os.path.join(config.output_dir, "index.html"),
        f"{train_config.run_name} step {step}",
        legend,
        pages,
    )
    print(
        f"[visualize] wrote {len(pages)} episodes to {os.path.join(config.output_dir, 'index.html')}"
    )


if __name__ == "__main__":
    main()
