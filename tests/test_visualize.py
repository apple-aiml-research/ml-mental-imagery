#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Visualization: the replay pairs every action with the true board after it, and imagined boards are
decoded in the layout the visual head is trained on (no GPU, no downloaded weights)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from mental_imagery.model.visual_head import VisualPredictionHead
from mental_imagery.puzzles import build_game_configs
from mental_imagery.visual_tokenizers.fsq import FSQ, FSQAutoencoder
from mental_imagery.visual_tokenizers.llamagen import VQModel
from mental_imagery.visualize import (
    Step,
    decode_written_boards,
    replay,
    tokenized_true_boards,
)
from tests.test_engine import FULL_EPISODE, load_games, make_engine

GAMES = ["sokoban", "jigsaw", "mental_rotation_certificate"]


def game_entry(key: str):
    return next(
        entry for entry in load_games("benchmark") if (entry.get("label") or entry.name) == key
    )


def expert_episode(puzzle_engine, entry, seed: int):
    """The expert's token stream of one episode, and the frames teacher forcing rendered."""
    loader = puzzle_engine.loader(games=build_game_configs([entry]), n=1, prefetch=0)
    rollout = loader.next_with_seeds(seeds=[seed])
    inputs, _, lengths, _ = rollout.teacher_force_labels(FULL_EPISODE)
    frames, _ = rollout.take_frames_array()
    return inputs[0, : lengths[0]].astype(np.int64), frames, loader


@pytest.mark.parametrize("game", GAMES)
def test_true_boards_are_the_visual_head_targets(mock_tokenizer, game):
    """Without boards in the stream, teacher forcing renders a frame after every action: the head's targets."""
    puzzle_engine = make_engine(mock_tokenizer, {"enabled": False})
    special = puzzle_engine.special_tokens
    stream, frames, loader = expert_episode(puzzle_engine, game_entry(game), seed=0)
    decode = mock_tokenizer.backend_tokenizer.decode
    rollout = loader.next_with_seeds(seeds=[0])
    steps, solved = replay(rollout, stream.tolist(), special, 256, decode)
    assert solved
    assert len(steps) == len(frames) + 1 > 2  # the start, then one step per action
    assert np.array_equal(np.asarray(steps[0].true_board), np.asarray(rollout.initial_images[0]))
    for step, frame in zip(steps[1:], frames, strict=True):
        assert np.array_equal(np.asarray(step.true_board), frame)
    assert [step.position for step in steps[1:]] == np.flatnonzero(
        stream == special["action_close"]
    ).tolist()
    assert steps[0].action is None and all(step.action for step in steps[1:])


def test_written_boards_follow_their_actions(mock_tokenizer):
    grid_size = 4
    puzzle_engine = make_engine(
        mock_tokenizer, {"enabled": True, "codebook_size": 16384, "grid_size": grid_size}
    )
    special = puzzle_engine.special_tokens
    stream, frames, loader = expert_episode(puzzle_engine, game_entry("sokoban"), seed=1)
    visual_id = mock_tokenizer.convert_tokens_to_ids("<v7>")
    stream[stream == special["visual_token_placeholder"]] = visual_id
    decode = mock_tokenizer.backend_tokenizer.decode
    steps, solved = replay(loader.next_with_seeds(seeds=[1]), stream.tolist(), special, 256, decode)
    assert solved
    for step, frame in zip(
        steps[:-1], frames, strict=True
    ):  # each written board shows its step's true board
        assert np.array_equal(np.asarray(step.true_board), frame)
    assert all(step.board_tokens == [visual_id] * grid_size**2 for step in steps[:-1])
    assert steps[-1].board_tokens is None  # nothing is written after the final action
    steps[1].board_tokens = steps[1].board_tokens[:-1]  # as if the model wrote one token too few

    class FakeTokenizer:
        def decode(self, codes, grid_w, grid_h):
            assert codes == [7] * grid_w * grid_h
            return torch.zeros(3, 16 * grid_h, 16 * grid_w)

    imagined = decode_written_boards(steps, FakeTokenizer(), {visual_id: 7}, grid_size)
    assert isinstance(imagined[0], Image.Image) and imagined[0].size == (64, 64)
    assert imagined[1].startswith("malformed board")
    assert imagined[-1] == "no board written"


def test_tokenized_true_boards_encode_and_decode_every_true_board():
    steps = [Step(true_board=Image.new("RGB", (256, 128), (0, 128, 0))) for _ in range(3)]

    class FakeTokenizer:
        def encode(self, images):
            assert [image.size for image in images] == [(256, 128)]
            return [torch.zeros(8 * 4, dtype=torch.long)]

        def decode(self, codes, grid_w, grid_h):
            assert (len(codes), grid_w, grid_h) == (32, 8, 4)
            return torch.ones(3, 16 * grid_h, 16 * grid_w)

    tokenized = tokenized_true_boards(steps, FakeTokenizer(), grid_size=4)
    assert [image.size for image in tokenized] == [(128, 64)] * 3


def test_fsq_codes_round_trip():
    fsq = FSQ(levels=[8, 5, 5, 5])
    codes = torch.stack([torch.randint(0, levels, (2, 3, 4)) for levels in (8, 5, 5, 5)], dim=1)
    assert torch.equal(fsq.codes_from_indices(fsq.indices(codes)), codes)
    assert torch.equal(fsq.codes_from_quantized(fsq.quantized_from_codes(codes)), codes)
    autoencoder = FSQAutoencoder(levels=[5, 5, 5]).eval()
    with torch.no_grad():
        z_q, indices, _ = autoencoder.encode(torch.rand(1, 3, 32, 32) * 2 - 1)
        torch.testing.assert_close(autoencoder.decode_indices(indices), autoencoder.decoder(z_q))


@pytest.fixture(scope="module")
def tiny_head(tmp_path_factory):
    """A randomly initialized head over 2 layers of 8-dim hidden states."""
    path = tmp_path_factory.mktemp("fsq") / "fsq.pt"
    torch.save({"levels": [5, 5, 5], "model": FSQAutoencoder(levels=[5, 5, 5]).state_dict()}, path)
    return VisualPredictionHead(
        hidden_size=8,
        num_model_layers=2,
        fsq_checkpoint=str(path),
        decoder_dim=16,
        num_layers=1,
        num_heads=2,
        fsq_dtype=torch.float32,
    ).eval()


def test_head_predicts_one_board_per_position(tiny_head):
    boards = tiny_head.predict(torch.randn(3, 2, 8))
    assert boards.shape == (3, 3, 256, 256)
    assert boards.min() >= 0 and boards.max() <= 1


def test_head_predictions_use_the_training_target_layout(tiny_head, monkeypatch):
    """Logits that put all mass on the target's codes: zero loss, and the prediction is the target's FSQ decode."""
    images = torch.rand(2, 3, 256, 256)
    fsq_model = tiny_head.fsq_model
    with torch.no_grad():
        codes = fsq_model.fsq.codes_from_quantized(
            fsq_model.fsq(fsq_model.encoder(images * 2 - 1))[0]
        )
    targets = codes.view(2, tiny_head.num_dims, -1).permute(0, 2, 1)
    monkeypatch.setattr(
        tiny_head, "logits", lambda hidden: F.one_hot(targets, tiny_head.num_levels).float() * 100
    )
    hidden = torch.randn(2, 2, 8)
    assert tiny_head(hidden, images).item() < 1e-6
    with torch.no_grad():
        expected = (fsq_model.decode_codes(codes).clamp(-1, 1) + 1) / 2
    torch.testing.assert_close(tiny_head.predict(hidden), expected)


def test_llamagen_decodes_a_grid_of_codes():
    model = VQModel().eval()
    with torch.no_grad():
        images = model.decode_indices(torch.randint(0, 16384, (2, 4, 6)))
    assert images.shape == (2, 3, 64, 96) and torch.isfinite(images).all()
