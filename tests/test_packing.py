#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Sequence packing layout: every episode keeps its tokens, in order, and its positions restart at 0."""

from __future__ import annotations

import pytest
import torch

from mental_imagery.utils.packing import (
    LinearAttentionSegments,
    assign_rows,
    build_plan,
    linear_attention_boundaries,
    row_lengths,
)


def padded_micro_batch():
    """Three episodes of 3, 5 and 2 tokens, left- and right-padded to 7 columns (as build_prompts + the engine)."""
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    tokens = (
        torch.arange(1, 22).reshape(3, 7) * attention_mask
    )  # token value = its position in the padded layout
    return attention_mask, tokens


def test_rows_take_consecutive_episodes_up_to_their_capacity():
    assert assign_rows([3, 5, 2], max_row_tokens=None) == [[0, 1, 2]]
    assert assign_rows([3, 5, 2], max_row_tokens=8) == [[0, 1], [2]]
    assert assign_rows([3, 5, 2], max_row_tokens=7) == [[0], [1, 2]]
    assert assign_rows([9, 2], max_row_tokens=7) == [
        [0],
        [1],
    ]  # too long for a row: a row of its own
    assert row_lengths([3, 5, 2], [[0, 1], [2]]) == [8, 2]


def test_single_row_concatenates_the_episodes_in_order_without_padding():
    attention_mask, tokens = padded_micro_batch()
    plan = build_plan(attention_mask, assign_rows([3, 5, 2], None), num_rows=1, row_length=10)
    assert plan.pack(tokens, fill_value=0).tolist() == [[2, 3, 4, 8, 9, 10, 11, 12, 17, 18]]
    assert plan.positions.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]
    assert plan.cu_seqlens.tolist() == [0, 3, 8, 10]
    assert plan.segment_ids().tolist() == [[0, 0, 0, 1, 1, 1, 1, 1, 2, 2]]


def test_row_padding_and_extra_rows_are_segments_of_their_own():
    attention_mask, tokens = padded_micro_batch()
    plan = build_plan(
        attention_mask, [[0, 1], [2]], num_rows=3, row_length=9
    )  # equalized to other ranks
    assert plan.pack(tokens, fill_value=0).tolist() == [
        [2, 3, 4, 8, 9, 10, 11, 12, 0],
        [17, 18, 0, 0, 0, 0, 0, 0, 0],
        [0] * 9,
    ]
    assert plan.positions.tolist() == [
        [0, 1, 2, 0, 1, 2, 3, 4, 0],
        [0, 1, 0, 1, 2, 3, 4, 5, 6],
        list(range(9)),
    ]
    assert plan.cu_seqlens.tolist() == [0, 3, 8, 9, 11, 18, 27]


def test_masks_and_labels_pack_like_the_tokens():
    attention_mask, tokens = padded_micro_batch()
    plan = build_plan(attention_mask, [[0, 1, 2]], num_rows=1, row_length=11)
    labels = torch.where(tokens % 2 == 0, tokens, torch.full_like(tokens, -100))
    assert plan.pack(labels, fill_value=-100).tolist() == [
        [2, -100, 4, 8, -100, 10, -100, 12, -100, 18, -100]
    ]
    is_board = tokens >= 10
    assert plan.pack(is_board, fill_value=False).tolist() == [
        [False, False, False, False, False, True, True, True, True, True, False]
    ]


def test_a_plan_rejects_shapes_too_small_for_its_rows():
    attention_mask, _ = padded_micro_batch()
    with pytest.raises(ValueError, match="do not fit"):
        build_plan(attention_mask, [[0, 1, 2]], num_rows=1, row_length=9)


def test_linear_attention_boundaries_need_a_single_row():
    attention_mask, _ = padded_micro_batch()
    cu_seqlens, seq_idx = linear_attention_boundaries(
        build_plan(attention_mask, [[0, 1, 2]], num_rows=1, row_length=10)
    )
    assert cu_seqlens.dtype == torch.int64 and cu_seqlens.tolist() == [0, 3, 8, 10]
    assert seq_idx.dtype == torch.int32 and seq_idx.tolist() == [[0, 0, 0, 1, 1, 1, 1, 1, 2, 2]]
    segments = LinearAttentionSegments()
    segments.set(cu_seqlens, seq_idx)
    assert segments.cu_seqlens is cu_seqlens and segments.seq_idx is seq_idx
    segments.clear()
    assert segments.cu_seqlens is None and segments.seq_idx is None
    with pytest.raises(ValueError, match="single packed row"):
        linear_attention_boundaries(
            build_plan(attention_mask, [[0, 1], [2]], num_rows=2, row_length=8)
        )


@pytest.fixture
def gloo_group(tmp_path):
    """A single-process gloo group, like the CPU group that packs micro-batches during training."""
    import torch.distributed as dist

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
        )
    yield dist.new_group(backend="gloo")
    if initialized_here:
        dist.destroy_process_group()


def test_pack_micro_batch_packs_inputs_labels_and_per_episode_masks(gloo_group):
    from omegaconf import OmegaConf

    from mental_imagery.model.families import FAMILIES
    from mental_imagery.train import pack_micro_batch

    board_open, board_close, action_close = 90, 91, 92
    attention_mask, _ = padded_micro_batch()
    input_ids = torch.tensor(
        [
            [0, 5, 6, action_close, 0, 0, 0],  # attended: columns 1-3
            [4, board_open, 7, board_close, action_close, 0, 0],  # columns 0-4
            [0, 0, 8, action_close, 0, 0, 0],  # columns 2-3
        ]
    )
    supervised = attention_mask & (torch.arange(7) >= 1)  # generation starts at column 1
    labels = torch.where(supervised, input_ids, torch.full_like(input_ids, -100))
    prepared = {
        "batch": {"input_ids": input_ids, "attention_mask": attention_mask},
        "labels": labels,
        "gen_start": 1,
    }
    config = OmegaConf.create(
        {
            "training": {"max_tokens": 7},
            "text_cot": {"enabled": True, "loss_weight": 0.3},
            "visual_tokens": {"enabled": False},
        }
    )
    special_tokens = {
        "board_open": board_open,
        "board_close": board_close,
        "action_close": action_close,
    }

    packed = pack_micro_batch(
        None,
        FAMILIES["internvl"],
        prepared,
        config,
        None,
        special_tokens,
        use_visual_head=True,
        group=gloo_group,
        linear_attention=False,
    )

    assert packed["inputs"]["input_ids"].tolist() == [
        [5, 6, action_close, 4, board_open, 7, board_close, action_close, 8, action_close]
    ]
    assert "attention_mask" not in packed["inputs"]  # episodes are kept apart by their positions
    assert packed["inputs"]["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]
    assert packed["labels"].tolist() == [
        [5, 6, action_close, -100, board_open, 7, board_close, action_close, 8, action_close]
    ]
    assert packed["positions_per_episode"] == [
        1,
        1,
        1,
    ]  # one </action> each, counted in the padded micro-batch
    assert packed["align_mask"].tolist() == [
        [False, False, True, False, False, False, False, True, False, True]
    ]
    assert packed["board_token_mask"].tolist() == [
        [False, False, False, False, False, True, False, False, False, False]
    ]
    assert packed["boundaries"] is None


def test_a_plan_rejects_episodes_with_gaps():
    attention_mask = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="not contiguous"):
        build_plan(attention_mask, [[0]], num_rows=1, row_length=5)
