#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Visual Supervision: predict the next board from the LLM's hidden states.

At every ``</action>``, the hidden states of all decoder layers are decoded into
the FSQ codes of the board after that action. Trained jointly with the language
model and discarded at inference (``mental_imagery/visualize.py`` decodes its
predictions into images).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mental_imagery.visual_tokenizers.fsq import load_fsq_autoencoder

NUM_PATCHES = 16 * 16  # 256x256 boards, 16x16-pixel patches


class HiddenStateCollector:
    """Captures decoder-layer outputs at masked positions with forward hooks.

    Cheaper than ``output_hidden_states=True``, which keeps every layer at
    every position and defeats activation checkpointing.
    """

    def __init__(self, layers: nn.ModuleList):
        self.mask = None
        self.collected: list[Tensor] = []
        for layer in layers:
            layer.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.mask is not None:
            self.collected.append(output[self.mask])

    def prepare(self, mask: Tensor) -> None:
        """(B, T) positions to capture in the next forward pass."""
        self.mask, self.collected = mask, []

    def collect(self) -> Tensor | None:
        """(L, N, H) hidden states; disables the hooks (before backward recomputation)."""
        self.mask = None
        collected, self.collected = self.collected, []
        return torch.stack(collected) if collected else None


class VisualPredictionHead(nn.Module):
    """Per-layer projections + learned patch queries + self-attention -> FSQ code logits."""

    def __init__(
        self,
        hidden_size,
        num_model_layers,
        fsq_checkpoint,
        decoder_dim,
        num_layers,
        num_heads,
        fsq_dtype,
    ):
        super().__init__()
        self.num_model_layers = num_model_layers
        self.fsq_model = load_fsq_autoencoder(fsq_checkpoint).to(dtype=fsq_dtype)
        levels = set(self.fsq_model.fsq.levels.long().tolist())
        if len(levels) != 1:
            raise ValueError(
                f"the visual head needs the same FSQ level count per dimension, got {levels}"
            )
        self.num_dims = self.fsq_model.fsq.num_dimensions
        self.num_levels = levels.pop()

        self.layer_proj = nn.ModuleList(
            [nn.Linear(hidden_size, decoder_dim) for _ in range(num_model_layers)]
        )
        self.type_embed = nn.Parameter(
            torch.randn(2, decoder_dim) * 0.02
        )  # memory vs. patch tokens
        self.patch_queries = nn.Parameter(torch.randn(NUM_PATCHES, decoder_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(NUM_PATCHES, decoder_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            decoder_dim,
            num_heads,
            dim_feedforward=4 * decoder_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.code_head = nn.Linear(decoder_dim, self.num_dims * self.num_levels)

    def logits(self, all_hidden_states: Tensor) -> Tensor:
        """(N, NUM_PATCHES, num_dims, num_levels) FSQ code logits from (N, layers, hidden) hidden states."""
        n, num_layers, _ = all_hidden_states.shape
        if num_layers != self.num_model_layers:
            raise ValueError(
                f"expected {self.num_model_layers} layers of hidden states, got {num_layers}"
            )
        memory = torch.stack(
            [self.layer_proj[i](all_hidden_states[:, i]) for i in range(num_layers)], dim=1
        )
        patches = (self.patch_queries + self.pos_embed).expand(n, -1, -1)
        tokens = torch.cat([memory + self.type_embed[0], patches + self.type_embed[1]], dim=1)
        patch_out = self.norm(self.transformer(tokens)[:, num_layers:])
        return self.code_head(patch_out).view(n, NUM_PATCHES, self.num_dims, self.num_levels)

    def forward(self, all_hidden_states: Tensor, target_images: Tensor) -> Tensor:
        """Cross-entropy of the predicted FSQ codes of ``target_images`` ((N, 3, 256, 256) in [0, 1])."""
        n = all_hidden_states.shape[0]
        with torch.no_grad():
            z_q, _ = self.fsq_model.fsq(self.fsq_model.encoder(target_images * 2 - 1))
            codes = (
                self.fsq_model.fsq.codes_from_quantized(z_q)
                .view(n, self.num_dims, -1)
                .permute(0, 2, 1)
            )
        logits = self.logits(all_hidden_states)
        return F.cross_entropy(logits.reshape(-1, self.num_levels), codes.reshape(-1))

    @torch.no_grad()
    def predict(self, all_hidden_states: Tensor) -> Tensor:
        """Predicted boards (N, 3, 256, 256) in [0, 1]: the most likely code of every patch, FSQ-decoded."""
        codes = self.logits(all_hidden_states).argmax(
            dim=-1
        )  # (N, patches, dims), patches row-major
        side = math.isqrt(NUM_PATCHES)
        codes = codes.permute(0, 2, 1).reshape(len(codes), self.num_dims, side, side)
        return (self.fsq_model.decode_codes(codes).float().clamp(-1, 1) + 1) / 2

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        """Load weights saved without the frozen FSQ codec."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.startswith("fsq_model.")]
        if missing or unexpected:
            raise ValueError(
                f"visual head state mismatch: missing={missing} unexpected={unexpected}"
            )


def build_visual_head(config, hidden_size: int, num_model_layers: int, dtype):
    if not config.visual_head.enabled:
        return None
    head = config.visual_head
    return VisualPredictionHead(
        hidden_size,
        num_model_layers,
        head.fsq_checkpoint,
        head.decoder_dim,
        head.num_layers,
        head.num_heads,
        dtype,
    )
