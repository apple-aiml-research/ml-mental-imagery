#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Frozen image tokenizers: board image -> grid of codebook indices (and back, for visualization).

Both backends downsample by 16, so a board resized to (16 * w) x (16 * h)
pixels becomes w x h codes.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms

from mental_imagery.visual_tokenizers.fsq import load_fsq_autoencoder
from mental_imagery.visual_tokenizers.llamagen import load_llamagen_vq16

PATCH_SIZE = 16


def code_grid_size(width: int, height: int, grid_size: int | None) -> tuple[int, int]:
    """(grid_w, grid_h): ``grid_size`` codes on the short edge, or native resolution if None."""
    native_w, native_h = width // PATCH_SIZE, height // PATCH_SIZE
    if grid_size is None:
        return native_w, native_h
    if native_w >= native_h:
        return grid_size * native_w // native_h, grid_size
    return grid_size, grid_size * native_h // native_w


class VisualTokenizer:
    def __init__(
        self,
        tokenizer: str,
        device,
        codebook_size: int,
        grid_size: int | None,
        fsq_checkpoint: str | None,
    ):
        if tokenizer == "llamagen":
            self.model = load_llamagen_vq16()
            model_codebook_size = self.model.quantize.n_e
        elif tokenizer == "fsq":
            self.model = load_fsq_autoencoder(fsq_checkpoint)
            model_codebook_size = self.model.codebook_size
        else:
            raise ValueError(f"unknown visual tokenizer {tokenizer!r}")
        if model_codebook_size != codebook_size:
            raise ValueError(
                f"codebook_size={codebook_size}, but the {tokenizer} tokenizer has {model_codebook_size}"
            )
        self.tokenizer = tokenizer
        self.grid_size = grid_size
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> list[torch.Tensor]:
        """One 1-D tensor of codebook indices (row-major) per image."""
        by_size = defaultdict(list)
        for index, image in enumerate(images):
            by_size[image.size].append(index)
        codes: list[torch.Tensor | None] = [None] * len(images)
        for (width, height), indices in by_size.items():
            grid_w, grid_h = code_grid_size(width, height, self.grid_size)
            preprocess = transforms.Compose(
                [
                    transforms.Resize(
                        (grid_h * PATCH_SIZE, grid_w * PATCH_SIZE),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.ToTensor(),
                ]
            )
            batch = torch.stack([preprocess(images[i]) for i in indices]).to(self.device) * 2 - 1
            if self.tokenizer == "llamagen":
                batch_codes = self.model.encode(batch)
            else:
                batch_codes = self.model.encode(batch)[1]
            batch_codes = batch_codes.long().reshape(len(indices), -1)
            for row, index in enumerate(indices):
                codes[index] = batch_codes[row]
        return codes

    @torch.no_grad()
    def decode(self, codes, grid_w: int, grid_h: int) -> torch.Tensor:
        """(3, 16 * grid_h, 16 * grid_w) image in [0, 1] (on the CPU) from one board's row-major codes."""
        indices = torch.as_tensor(codes, dtype=torch.long, device=self.device).view(
            1, grid_h, grid_w
        )
        image = self.model.decode_indices(indices)[0].float()
        return ((image.clamp(-1, 1) + 1) / 2).cpu()


def build_visual_tokenizer(visual_tokens_config, device, grid_size: int | None) -> VisualTokenizer:
    """Tokenizer for a ``visual_tokens`` config section. ``grid_size=None`` = native resolution."""
    return VisualTokenizer(
        tokenizer=visual_tokens_config.tokenizer,
        device=device,
        codebook_size=visual_tokens_config.codebook_size,
        grid_size=grid_size,
        fsq_checkpoint=visual_tokens_config.fsq_checkpoint,
    )
