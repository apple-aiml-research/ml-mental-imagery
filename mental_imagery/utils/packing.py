#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Sequence packing: a micro-batch's episodes concatenated in a few rows instead of padded to the longest.

Packing concatenates a micro-batch's episodes, in order, into rows. Each episode's positions
restart at 0, which keeps the episodes apart:

- FlashAttention-2 reads the restarts from 2-D position ids and runs its variable-length kernel;
- SDPA and flex attention get a block-diagonal ("document") causal mask from transformers;
- Qwen 3.5's Gated DeltaNet layers get the segment boundaries (``LinearAttentionSegments``).

Trailing padding of a row forms its own segment (unsupervised). A micro-batch holds the same episodes
as without packing, so its loss is the same up to kernel numerics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def assign_rows(lengths: list[int], max_row_tokens: int | None) -> list[list[int]]:
    """Episodes of each row, in order: as many consecutive episodes as fit in ``max_row_tokens``.

    ``max_row_tokens=None`` puts every episode in one row. An episode longer than the row
    capacity gets a row of its own.
    """
    if max_row_tokens is None:
        return [list(range(len(lengths)))]
    rows, used = [[]], 0
    for episode, length in enumerate(lengths):
        if rows[-1] and used + length > max_row_tokens:
            rows.append([])
            used = 0
        rows[-1].append(episode)
        used += length
    return rows if rows[-1] else rows[:-1]


def row_lengths(lengths: list[int], rows: list[list[int]]) -> list[int]:
    return [sum(lengths[episode] for episode in row) for row in rows]


@dataclass(frozen=True)
class PackingPlan:
    """Where each kept token of a padded (B, T) micro-batch goes in the packed (num_rows, row_length) one."""

    source_rows: Tensor  # (N,) padded row (episode) of each kept token
    source_columns: Tensor  # (N,) its padded column
    target_rows: Tensor  # (N,) its packed row
    target_columns: Tensor  # (N,) its packed column
    num_rows: int
    row_length: int
    positions: (
        Tensor  # (num_rows, row_length) position within its segment (episode or trailing padding)
    )
    cu_seqlens: (
        Tensor  # (num_segments + 1,) segment boundaries in the flattened packed layout (int32)
    )

    def pack(self, tensor: Tensor, fill_value) -> Tensor:
        """A (B, T, ...) padded tensor in the packed (num_rows, row_length, ...) layout."""
        packed = tensor.new_full((self.num_rows, self.row_length, *tensor.shape[2:]), fill_value)
        packed[self.target_rows, self.target_columns] = tensor[
            self.source_rows, self.source_columns
        ]
        return packed

    def segment_ids(self) -> Tensor:
        """(1, num_rows * row_length) int32 segment index of each flattened token (causal-conv1d's ``seq_idx``)."""
        lengths = (self.cu_seqlens[1:] - self.cu_seqlens[:-1]).long()
        return torch.repeat_interleave(
            torch.arange(len(lengths), device=lengths.device, dtype=torch.int32), lengths
        )[None]


def build_plan(
    attention_mask: Tensor, rows: list[list[int]], num_rows: int, row_length: int
) -> PackingPlan:
    """Pack the attended tokens of each episode (row of ``attention_mask``) into ``rows``.

    ``num_rows`` and ``row_length`` may exceed what ``rows`` needs (to equalize shapes across ranks):
    the extra rows and the end of each row are padding segments. Each episode's attended tokens
    must be contiguous (a left-padded prompt followed by its response, as ``prepare_batch`` builds
    them). Vectorized, with one device-to-host copy: it runs once per micro-batch.
    """
    device = attention_mask.device
    mask = attention_mask.bool()
    num_episodes = mask.shape[0]
    starts = mask.int().argmax(dim=1)  # first attended column of each episode
    gaps = (mask[:, 1:] & ~mask[:, :-1]).sum(dim=1) + mask[
        :, 0
    ].int()  # spans of attended tokens (1 if contiguous)
    lengths, gap_counts = torch.stack([mask.sum(dim=1), gaps]).tolist()
    if any(count > 1 for count in gap_counts):
        raise ValueError("an episode's attended tokens are not contiguous")
    if len(rows) > num_rows or max(row_lengths(lengths, rows), default=0) > row_length:
        raise ValueError(
            f"{len(rows)} rows of up to {max(row_lengths(lengths, rows))} tokens do not fit "
            f"{num_rows} x {row_length}"
        )
    episode_row, episode_offset = [0] * num_episodes, [0] * num_episodes
    order, used, boundaries = [], [0] * num_rows, [0]
    for row in range(num_rows):
        for episode in rows[row] if row < len(rows) else []:
            episode_row[episode], episode_offset[episode] = row, used[row]
            order.append(episode)
            used[row] += lengths[episode]
            boundaries.append(row * row_length + used[row])
        if used[row] < row_length:  # trailing padding: a segment of its own
            boundaries.append((row + 1) * row_length)

    order_t = torch.tensor(order, dtype=torch.long, device=device)
    order_lengths = torch.tensor(
        [lengths[episode] for episode in order], dtype=torch.long, device=device
    )
    token_episode = torch.repeat_interleave(
        order_t, order_lengths
    )  # episode of each kept token, in packed order
    first_token = torch.cumsum(order_lengths, dim=0) - order_lengths
    within = torch.arange(len(token_episode), device=device) - torch.repeat_interleave(
        first_token, order_lengths
    )
    target_rows = torch.tensor(episode_row, dtype=torch.long, device=device)[token_episode]
    target_columns = (
        torch.tensor(episode_offset, dtype=torch.long, device=device)[token_episode] + within
    )
    columns = torch.arange(row_length, device=device)
    used_t = torch.tensor(used, dtype=torch.long, device=device)
    positions = (
        columns[None] - used_t[:, None]
    )  # trailing padding: positions from 0 after the last episode
    positions[target_rows, target_columns] = within
    return PackingPlan(
        source_rows=token_episode,
        source_columns=starts[token_episode] + within,
        target_rows=target_rows,
        target_columns=target_columns,
        num_rows=num_rows,
        row_length=row_length,
        positions=positions,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device),
    )


def linear_attention_boundaries(plan: PackingPlan) -> tuple[Tensor, Tensor]:
    """``(cu_seqlens, seq_idx)`` of a single-row plan, as Qwen 3.5's Gated DeltaNet kernels take them."""
    if plan.num_rows != 1:
        raise ValueError("Gated DeltaNet packing needs a single packed row")
    return plan.cu_seqlens.long(), plan.segment_ids().contiguous()


class LinearAttentionSegments:
    """Segment boundaries of the current packed micro-batch, read by Qwen 3.5's Gated DeltaNet layers.

    Set by the training thread before the forward pass and cleared after the backward pass:
    activation checkpointing recomputes the layers during backward, and the recomputation must see
    the same segments. (The boundaries are computed with the micro-batch, on the preparation thread.)
    """

    def __init__(self):
        self.cu_seqlens: Tensor | None = None  # int64, for fla's chunk_gated_delta_rule
        self.seq_idx: Tensor | None = None  # (1, L) int32, for causal_conv1d_fn

    def set(self, cu_seqlens: Tensor, seq_idx: Tensor) -> None:
        self.cu_seqlens, self.seq_idx = cu_seqlens, seq_idx

    def clear(self) -> None:
        self.cu_seqlens = self.seq_idx = None
