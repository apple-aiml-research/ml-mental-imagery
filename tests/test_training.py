#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Training-path unit tests: visual-token stitching, board masks, and the decoupled loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mental_imagery.train import language_model_loss, visual_head_loss
from mental_imagery.utils.data import IGNORE_INDEX, board_content_mask, stitch_visual_tokens

BOARD_OPEN, BOARD_CLOSE, SENTINEL = 1, 2, 99


class SumHeadModel(torch.nn.Module):
    """Stand-in for a model with a visual head whose loss is the sum of its inputs."""

    def visual_prediction_head(self, hidden, target_images):
        return hidden.sum() + target_images.sum()


@pytest.mark.parametrize("stop_gradient", [False, True])
def test_stop_gradient_keeps_the_visual_loss_out_of_the_model(stop_gradient):
    hidden = torch.randn(3, 4, 8, requires_grad=True)  # (layers, </action> positions, hidden size)
    loss = visual_head_loss(
        SumHeadModel(),
        hidden,
        positions_per_episode=[4],
        target_images=torch.zeros(4, 3, 2, 2),
        frame_counts=[5],
        max_frames=32,
        stop_gradient=stop_gradient,
    )
    assert loss.requires_grad != stop_gradient
    if not stop_gradient:
        loss.backward()
        assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_stitch_fills_the_kth_board_with_the_kth_frame():
    # One episode, two boards of two visual tokens each, after a 1-token prompt.
    seq = torch.tensor(
        [
            [
                7,
                BOARD_OPEN,
                SENTINEL,
                SENTINEL,
                BOARD_CLOSE,
                5,
                BOARD_OPEN,
                SENTINEL,
                SENTINEL,
                BOARD_CLOSE,
            ]
        ]
    )
    labels = seq.clone()
    frames = [torch.tensor([40, 41]), torch.tensor([50, 51])]
    stitch_visual_tokens(
        seq=seq,
        labels=labels,
        token_ids=frames,
        frame_counts=[2],
        gen_start=1,
        board_open_id=BOARD_OPEN,
        sentinel_id=SENTINEL,
    )
    expected = torch.tensor(
        [[7, BOARD_OPEN, 40, 41, BOARD_CLOSE, 5, BOARD_OPEN, 50, 51, BOARD_CLOSE]]
    )
    assert torch.equal(seq, expected) and torch.equal(labels, expected)


def test_stitch_rejects_frame_board_mismatch():
    seq = torch.tensor([[BOARD_OPEN, SENTINEL, BOARD_CLOSE]])
    with pytest.raises(RuntimeError, match="rendered frames"):
        stitch_visual_tokens(
            seq=seq,
            labels=seq.clone(),
            token_ids=[torch.tensor([3]), torch.tensor([4])],
            frame_counts=[2],
            gen_start=0,
            board_open_id=BOARD_OPEN,
            sentinel_id=SENTINEL,
        )


def test_board_content_mask_covers_only_text_between_delimiters():
    labels = torch.tensor(
        [[IGNORE_INDEX, BOARD_OPEN, 7, 8, BOARD_CLOSE, 5, 6, BOARD_OPEN, 9, BOARD_CLOSE, 4]]
    )
    mask = board_content_mask(labels, board_open_id=BOARD_OPEN, board_close_id=BOARD_CLOSE)
    expected = torch.tensor(
        [[False, False, True, True, False, False, False, False, True, False, False]]
    )
    assert torch.equal(mask, expected)


def test_decoupled_loss_weights_board_and_text_means():
    """loss = w * mean CE(board tokens) + (1 - w) * mean CE(other tokens), regardless of token counts."""
    torch.manual_seed(0)
    vocab_size = 12
    logits = torch.randn(1, 8, vocab_size)
    labels = torch.tensor([[IGNORE_INDEX, 3, 10, 11, 10, 4, IGNORE_INDEX, 5]])
    is_board = torch.tensor([[False, False, True, True, True, False, False, False]])

    loss, metrics = language_model_loss(logits, labels, is_board, board_loss_weight=0.25)

    targets = labels[0, 1:]
    per_token = F.cross_entropy(logits[0, :-1], targets.clamp(min=0), reduction="none")
    valid = targets != IGNORE_INDEX
    board = valid & is_board[0, 1:]
    text = valid & ~board
    expected = 0.25 * per_token[board].mean() + 0.75 * per_token[text].mean()
    assert torch.allclose(loss, expected)
    assert metrics["loss/visual_ce"] == pytest.approx(per_token[board].mean().item())


def test_plain_loss_ignores_unsupervised_positions():
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[IGNORE_INDEX, 1, 2, IGNORE_INDEX, 3], [IGNORE_INDEX] * 4 + [6]])
    loss, _ = language_model_loss(logits, labels, None, board_loss_weight=0.0)
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX
    )
    assert torch.allclose(loss, expected)
