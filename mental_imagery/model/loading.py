#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Model loading, new-token embedding initialization, and FSDP2 wrapping."""

from __future__ import annotations

import re

import torch
import torch.distributed as dist
from torch.distributed._composable import checkpoint as activation_checkpoint
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForImageTextToText, AutoProcessor

from mental_imagery.model.families import STRUCTURAL_TOKENS, resolve_family, visual_token

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def check_pinned_revision(model_name: str, revision) -> None:
    """Raise unless ``revision`` is a full Hub commit SHA (branches and tags such as main move)."""
    if not isinstance(revision, str) or not COMMIT_SHA.fullmatch(revision):
        raise ValueError(
            f"model.revision of {model_name} is {revision!r}; set it to the full 40-character "
            "Hugging Face Hub commit SHA of the weights to load"
        )


def rank0_first(load_fn):
    """Run ``load_fn`` on rank 0 first (it downloads), then on the other ranks."""
    result = load_fn() if dist.get_rank() == 0 else None
    dist.barrier()
    return result if dist.get_rank() == 0 else load_fn()


def load_processor(model_name: str, revision: str | None, visual_tokens_config):
    """Processor of a Hub model at a pinned revision, with left padding and the run's added tokens."""
    check_pinned_revision(model_name, revision)
    processor = AutoProcessor.from_pretrained(model_name, revision=revision)
    processor.tokenizer.padding_side = "left"
    tokens = list(STRUCTURAL_TOKENS)
    if visual_tokens_config.enabled:
        tokens += [visual_token(code) for code in range(visual_tokens_config.codebook_size)]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return processor


def load_checkpoint_processor(checkpoint_dir: str):
    """Processor saved with a training checkpoint (it holds the run's added tokens), with left padding."""
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    processor.tokenizer.padding_side = "left"
    return processor


def load_pretrained(path_or_name: str, revision: str | None, model_config):
    """Model in fp32: FSDP keeps fp32 master weights and computes in bf16."""
    family = resolve_family(model_config.family)
    model = AutoModelForImageTextToText.from_pretrained(
        path_or_name,
        revision=revision,
        dtype=torch.float32,
        attn_implementation=model_config.attn_implementation,
    )
    family.check_model_type(model.config)
    family.freeze_unused_modules(model)
    return model


def init_mean_noise(model, old_vocab_size: int, token_ids) -> None:
    """New token = vocabulary mean + 2% noise; LM-head rows rescaled to the mean existing norm."""
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    input_mean, input_std = (
        input_weight[:old_vocab_size].mean(dim=0),
        input_weight[:old_vocab_size].std().item(),
    )
    output_mean, output_std = (
        output_weight[:old_vocab_size].mean(dim=0),
        output_weight[:old_vocab_size].std().item(),
    )
    output_norm = output_weight[:old_vocab_size].norm(dim=1).mean()
    for token_id in token_ids:
        input_weight[token_id] = input_mean + torch.randn_like(input_mean) * 0.02 * input_std
        if input_weight is not output_weight:  # untied embeddings
            vec = output_mean + torch.randn_like(output_mean) * 0.02 * output_std
            output_weight[token_id] = vec * (output_norm / vec.norm())


def init_from_embed_table(model, processor, config, old_vocab_size: int, device) -> None:
    """Visual tokens get their precomputed vision feature (see ``prepare/embed_init.py``).

    The LM-head rows are rescaled to the mean existing norm; with tied
    embeddings this also sets the input rows. Structural tokens get mean+noise init.
    """
    table = torch.load(config.visual_tokens.embed_init_path, map_location=device, weights_only=True)
    expected = {
        "model_name": config.model.name,
        "visual_tokenizer": config.visual_tokens.tokenizer,
        "codebook_size": config.visual_tokens.codebook_size,
        "render_resolution": config.engine.render_resolution,
    }
    for key, value in expected.items():
        if table["metadata"][key] != value:
            raise ValueError(
                f"embed-init table has {key}={table['metadata'][key]!r}, run uses {value!r}"
            )
    embeddings = table["embeddings"].to(dtype=DTYPES[config.model.dtype], device=device)

    vocab = processor.tokenizer.get_vocab()
    visual_ids = [vocab[visual_token(code)] for code in range(config.visual_tokens.codebook_size)]
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    output_norm = output_weight[:old_vocab_size].norm(dim=1).mean()
    input_weight[visual_ids] = embeddings.to(input_weight.dtype)
    # Scale in fp32 and round the product to the model dtype before storing it in the (fp32) weights.
    scale = output_norm / embeddings.norm(dim=1, keepdim=True).float()
    output_weight[visual_ids] = (
        (embeddings.float() * scale).to(embeddings.dtype).to(output_weight.dtype)
    )
    structural_ids = sorted(set(range(old_vocab_size, len(processor.tokenizer))) - set(visual_ids))
    init_mean_noise(model, old_vocab_size, structural_ids)


def load_base_model(config, device):
    """Pretrained model with embeddings for the new tokens, identical on every rank."""
    processor = rank0_first(
        lambda: load_processor(config.model.name, config.model.revision, config.visual_tokens)
    )
    model = rank0_first(
        lambda: load_pretrained(config.model.name, config.model.revision, config.model)
    )
    old_vocab_size = model.config.text_config.vocab_size
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    model.to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            if config.model.new_token_init == "mean_noise":
                init_mean_noise(
                    model, old_vocab_size, range(old_vocab_size, len(processor.tokenizer))
                )
            elif config.model.new_token_init == "embed_init":
                init_from_embed_table(model, processor, config, old_vocab_size, device)
            else:
                raise ValueError(f"unknown model.new_token_init {config.model.new_token_init!r}")
    for weight in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
        dist.broadcast(weight.data, src=0)
    return model, processor


def load_resume_model(checkpoint_dir: str, config, device):
    model = rank0_first(lambda: load_pretrained(checkpoint_dir, None, config.model))
    return model.to(device), load_checkpoint_processor(checkpoint_dir)


def wrap_fsdp(model, config):
    """FSDP2: shard each decoder/vision block, the visual head, then the root."""
    family = resolve_family(config.model.family)
    family.prepare_for_fsdp()
    dtype = DTYPES[config.model.dtype]
    kwargs = dict(
        mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype),
        reshard_after_forward=config.fsdp.reshard_after_forward,
    )
    layers = [
        module for module in model.modules() if isinstance(module, family.fsdp_layer_classes())
    ]
    if not layers:
        raise RuntimeError(f"no {family.name} decoder or vision blocks found to shard")
    for layer in layers:
        fully_shard(layer, **kwargs)
    if hasattr(model, "visual_prediction_head"):
        fully_shard(model.visual_prediction_head, **kwargs)
    fully_shard(model, **kwargs)
    model.set_modules_to_forward_prefetch(layers)
    if config.fsdp.activation_checkpointing:
        for layer in layers:
            activation_checkpoint(layer)
    return model
