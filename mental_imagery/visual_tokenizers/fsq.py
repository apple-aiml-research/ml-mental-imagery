#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""FSQ (finite scalar quantization) autoencoder for board images.

A small convolutional autoencoder whose codes are local: moving one piece only
changes nearby codes. Trained on rendered boards by ``mental_imagery.prepare.train_fsq``
and used frozen, as the target of the visual prediction head or as the
tokenizer of the FSQ visual-token runs. Input (B, 3, H, W) in [-1, 1] gives an
(H/16) x (W/16) grid of codes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fsq_entropy_loss(z_bounded: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Negative normalized entropy of the soft level usage, averaged over dimensions.

    Minimizing it spreads the encoder outputs over all levels (better codebook use).
    """
    total = 0.0
    for dim, num_levels in enumerate(levels.long().tolist()):
        half_width = (num_levels - 1) / 2
        centers = (
            torch.arange(num_levels, device=z_bounded.device, dtype=torch.float32) - half_width
        )
        values = (z_bounded[:, dim] * half_width).reshape(-1, 1)
        usage = (-((values - centers) ** 2)).softmax(dim=1).mean(dim=0)
        total += -(usage * (usage + 1e-8).log()).sum() / math.log(num_levels)
    return -total / len(levels)


class FSQ(nn.Module):
    """Rounds each latent dimension to a fixed number of levels (codebook = prod(levels))."""

    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        self.codebook_size = math.prod(levels)
        self.num_dimensions = len(levels)
        # Fixed random rotation that decorrelates the dimensions before rounding.
        rotation = torch.empty(self.num_dimensions, self.num_dimensions)
        nn.init.orthogonal_(rotation)
        self.register_buffer("rotation", rotation)

    def _half_width(self) -> torch.Tensor:
        return (self.levels.view(1, -1, 1, 1) - 1) / 2

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, D, H, W) latents -> (quantized values in [-1, 1], bounded pre-rounding values)."""
        z = F.conv2d(z, self.rotation.view(self.num_dimensions, self.num_dimensions, 1, 1))
        z_bounded = z / (1 + z.abs())
        half_width = self._half_width()
        z_rounded = (z_bounded * half_width).round().clamp(-half_width, half_width) / half_width
        z_q = z_bounded + (z_rounded - z_bounded).detach()  # straight-through gradient
        return z_q, z_bounded

    def codes_from_quantized(self, z_q: torch.Tensor) -> torch.Tensor:
        """Quantized values -> integer level per dimension, (B, D, H, W) in [0, L-1]."""
        half_width = self._half_width()
        codes = (z_q * half_width + half_width).round().long()
        return codes.clamp(min=0).minimum((self.levels.view(1, -1, 1, 1) - 1).long())

    def quantized_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Levels (B, D, H, W) -> quantized values in [-1, 1]; inverse of ``codes_from_quantized``."""
        half_width = self._half_width()
        return (codes.to(half_width.dtype) - half_width) / half_width

    def indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Per-dimension codes (B, D, H, W) -> flat codebook indices (B, H, W)."""
        levels = self.levels.long()
        index = codes[:, 0]
        for dim in range(1, self.num_dimensions):
            index = index * levels[dim] + codes[:, dim]
        return index

    def codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Flat codebook indices (B, H, W) -> per-dimension codes (B, D, H, W) (inverse of ``indices``)."""
        codes = []
        for num_levels in reversed(self.levels.long().tolist()):
            codes.append(indices % num_levels)
            indices = indices // num_levels
        return torch.stack(codes[::-1], dim=1)


class PixelNorm(nn.Module):
    """Normalizes over channels at each pixel independently (no spatial mixing)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - x.mean(dim=1, keepdim=True)) * torch.rsqrt(
            x.var(dim=1, keepdim=True, correction=0) + self.eps
        )
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def norm_act(channels: int) -> list[nn.Module]:
    return [PixelNorm(channels), nn.SiLU(inplace=True)]


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            *norm_act(channels),
            conv3x3(channels, channels),
            *norm_act(channels),
            conv3x3(channels, channels),
        )
        nn.init.zeros_(self.block[-1].weight)  # each block starts as the identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FSQEncoder(nn.Module):
    """Four stride-2 stages (16x downsampling), residual blocks only in the first."""

    def __init__(self, fsq_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(3, 128, stride=2),
            ResBlock(128),
            ResBlock(128),
            conv3x3(128, 256, stride=2),
            *norm_act(256),
            conv3x3(256, 512, stride=2),
            *norm_act(512),
            conv3x3(512, 512, stride=2),
            *norm_act(512),
            nn.Conv2d(512, fsq_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSQDecoder(nn.Module):
    """Mirror of the encoder with nearest-neighbor upsampling."""

    def __init__(self, fsq_dim: int):
        super().__init__()
        up = lambda: nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
        self.net = nn.Sequential(
            nn.Conv2d(fsq_dim, 512, kernel_size=1),
            *norm_act(512),
            up(),
            conv3x3(512, 512),
            *norm_act(512),
            up(),
            conv3x3(512, 256),
            *norm_act(256),
            up(),
            conv3x3(256, 128),
            ResBlock(128),
            ResBlock(128),
            up(),
            conv3x3(128, 128),
            *norm_act(128),
            nn.Conv2d(128, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSQAutoencoder(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.fsq = FSQ(levels=levels)
        self.encoder = FSQEncoder(fsq_dim=len(levels))
        self.decoder = FSQDecoder(fsq_dim=len(levels))

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Images -> (quantized latents, flat indices (B, h, w), bounded latents)."""
        z_q, z_bounded = self.fsq(self.encoder(x))
        return z_q, self.fsq.indices(self.fsq.codes_from_quantized(z_q)), z_bounded

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Per-dimension codes (B, D, h, w) -> images (B, 3, 16 * h, 16 * w), roughly in [-1, 1]."""
        return self.decoder(self.fsq.quantized_from_codes(codes))

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Flat codebook indices (B, h, w) -> images (B, 3, 16 * h, 16 * w), roughly in [-1, 1]."""
        return self.decode_codes(self.fsq.codes_from_indices(indices))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Images -> (reconstruction, flat indices, bounded latents)."""
        z_q, indices, z_bounded = self.encode(x)
        return self.decoder(z_q), indices, z_bounded


def load_fsq_autoencoder(checkpoint_path: str) -> FSQAutoencoder:
    """Frozen autoencoder from a checkpoint written by ``mental_imagery.prepare.train_fsq``."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FSQAutoencoder(levels=checkpoint["levels"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    return model
