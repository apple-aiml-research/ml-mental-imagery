#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# LlamaGen VQ-16 image tokenizer (16x downsampling, 16,384-entry codebook).
#
# Model code adapted from LlamaGen (https://github.com/FoundationVision/LlamaGen,
# MIT License, Copyright (c) 2024 FoundationVision), which builds on
# taming-transformers and MaskGIT. Used here as a frozen image tokenizer; the
# released class-conditional ImageNet weights are downloaded from the
# Hugging Face Hub. Training encodes boards; decoding is only used to
# visualize the boards a Visual CoT model writes (mental_imagery/visualize.py).

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from mental_imagery.visual_tokenizers import (
    LLAMAGEN_REPO_ID,
    LLAMAGEN_REVISION,
    LLAMAGEN_VQ16_FILENAME,
)

LLAMAGEN_CODEBOOK_SIZE = 16384
CODEBOOK_EMBED_DIM = 8
Z_CHANNELS = 256
CHANNEL_MULTIPLIERS = (1, 1, 2, 2, 4)


def _nonlinearity(x):
    return x * torch.sigmoid(x)


def _normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(
            F.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest").to(x.dtype)
        )


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = self.conv1(_nonlinearity(self.norm1(x)))
        h = self.conv2(_nonlinearity(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h_dim, w_dim = q.shape
        q = q.reshape(b, c, h_dim * w_dim).permute(0, 2, 1)  # (b, hw, c)
        k = k.reshape(b, c, h_dim * w_dim)  # (b, c, hw)
        w = torch.bmm(q, k) * (int(c) ** (-0.5))
        w = F.softmax(w, dim=2)

        v = v.reshape(b, c, h_dim * w_dim)
        h_out = torch.bmm(v, w.permute(0, 2, 1))
        h_out = h_out.reshape(b, c, h_dim, w_dim)
        return x + self.proj_out(h_out)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=CHANNEL_MULTIPLIERS,
        num_res_blocks=2,
        z_channels=Z_CHANNELS,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in)
            self.conv_blocks.append(conv_block)

        self.mid = nn.ModuleList(
            [ResnetBlock(block_in, block_in), AttnBlock(block_in), ResnetBlock(block_in, block_in)]
        )

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        for mid_block in self.mid:
            h = mid_block(h)
        h = _nonlinearity(self.norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels=Z_CHANNELS,
        ch=128,
        ch_mult=CHANNEL_MULTIPLIERS,
        num_res_blocks=2,
        out_channels=3,
    ):
        super().__init__()
        num_resolutions = len(ch_mult)
        self.num_resolutions = num_resolutions
        block_in = ch * ch_mult[num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.ModuleList(
            [ResnetBlock(block_in, block_in), AttnBlock(block_in), ResnetBlock(block_in, block_in)]
        )

        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if i_level == num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != 0:
                conv_block.upsample = Upsample(block_in)
            self.conv_blocks.append(conv_block)

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        for mid_block in self.mid:
            h = mid_block(h)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(len(block.res)):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)
        h = _nonlinearity(self.norm_out(h))
        return self.conv_out(h)


class VectorQuantizer(nn.Module):
    """L2-normalized codebook; maps each latent vector to its nearest code."""

    def __init__(self, n_e, e_dim):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))  # in the checkpoint

    def codes(self, z):
        """(B * H * W,) nearest-code indices of (B, C, H, W) latents, row-major per image."""
        z = torch.einsum("b c h w -> b h w c", z).contiguous()
        z_flattened = F.normalize(z.view(-1, self.e_dim), p=2, dim=-1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, embedding.t())
        )
        return torch.argmin(d, dim=1)

    def embed(self, indices):
        """(B, C, H, W) normalized codebook vectors of (B, H, W) indices."""
        return F.normalize(self.embedding.weight, p=2, dim=-1)[indices].permute(0, 3, 1, 2)


class VQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = VectorQuantizer(LLAMAGEN_CODEBOOK_SIZE, CODEBOOK_EMBED_DIM)
        self.quant_conv = nn.Conv2d(Z_CHANNELS, CODEBOOK_EMBED_DIM, 1)
        self.post_quant_conv = nn.Conv2d(CODEBOOK_EMBED_DIM, Z_CHANNELS, 1)

    def encode(self, x):
        """Codebook indices of images in [-1, 1], flattened over (B, H / 16, W / 16)."""
        return self.quantize.codes(self.quant_conv(self.encoder(x)))

    def decode_indices(self, indices):
        """(B, H, W) codebook indices -> images (B, 3, 16 * H, 16 * W), roughly in [-1, 1]."""
        return self.decoder(self.post_quant_conv(self.quantize.embed(indices)))


def load_llamagen_vq16() -> VQModel:
    """Download (once, into the Hugging Face cache) and load the frozen VQ-16 model."""
    weight_path = hf_hub_download(
        repo_id=LLAMAGEN_REPO_ID,
        filename=LLAMAGEN_VQ16_FILENAME,
        revision=LLAMAGEN_REVISION,
    )
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = VQModel()
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
