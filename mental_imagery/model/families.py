#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Model families: what differs between the supported VLMs.

``model.family`` in a model config selects one of ``FAMILIES``: Qwen 3.5 (dense
and MoE), Gemma 4, or InternVL3.5. A family fixes the engine's special tokens,
the prompt format, how images reach the processor, the FSDP-sharded layers, the
vision features that ``embed_init`` averages, and its loading and checkpoint
quirks. An unknown family, a checkpoint of another family, or a token missing
from the tokenizer raises.
"""

from __future__ import annotations

import os

import torch
import transformers

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "templates")
STRUCTURAL_TOKENS = ["<board>", "</board>", "<action>", "</action>"]


def visual_token(code: int) -> str:
    return f"<v{code}>"


def user_message_text(prompt: str, actions: str) -> str:
    return f"{prompt} Actions: {actions}" if actions else prompt


def resolve_token_id(tokenizer, token: str) -> int:
    """Token id, raising if absent (``convert_tokens_to_ids`` silently returns unk)."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"token {token!r} is not in the tokenizer vocabulary")
    return token_id


def chat_template_prompt(processor, user_text: str) -> str:
    """The processor's chat template applied to one user turn: an image, then ``user_text``."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class ModelFamily:
    """Behavior shared by the families; each subclass sets the attributes below."""

    name: str
    model_types: tuple[str, ...]  # config.model_type of the family's checkpoints
    think_close_token: str  # closes the reasoning span (the engine's think_close)
    turn_end_token: str  # ends the assistant turn (the engine's im_end)
    eos_token: str
    pad_token: str
    uses_mrope: bool  # Qwen's 3-D mRoPE positions (Qwen-only patches and decoding)
    vllm_chat_template: str | None  # file in configs/templates; None: the checkpoint's own template

    def engine_token_kwargs(self, tokenizer) -> dict:
        return {
            "think_close_token": self.think_close_token,
            "im_end_token": self.turn_end_token,
            "eos_token_id": resolve_token_id(tokenizer, self.eos_token),
            "pad_token_id": resolve_token_id(tokenizer, self.pad_token),
        }

    def vllm_chat_template_path(self) -> str | None:
        """Chat template for the vLLM server, reproducing ``build_prompt_text`` (None: the model's own)."""
        return (
            os.path.join(TEMPLATES_DIR, self.vllm_chat_template)
            if self.vllm_chat_template
            else None
        )

    def check_model_type(self, model_config) -> None:
        if model_config.model_type not in self.model_types:
            raise ValueError(
                f"model_type {model_config.model_type!r} is not a {self.name} model "
                f"(expected one of {self.model_types}); check model.family"
            )

    def build_prompt_text(self, processor, user_text: str) -> str:
        """Prompt with one image, then ``user_text``, ending inside the opened reasoning span.

        Generation therefore starts with episode content.
        """
        raise NotImplementedError

    def processor_images(self, images: list) -> list:
        """Images as the processor expects them, one per prompt."""
        return images

    def check_image_inputs(self, inputs, num_images: int) -> None:
        """Raise if the processor encoded the boards in a way the family does not support."""

    def fsdp_layer_classes(self) -> tuple[type, ...]:
        """Decoder and vision blocks that are sharded and activation-checkpointed."""
        raise NotImplementedError

    def prepare_for_fsdp(self) -> None:
        """Patches needed before FSDP2 wrapping."""

    def freeze_unused_modules(self, model) -> None:
        """Freeze parameters that no input reaches (kept out of the optimizer state)."""

    def prepare_checkpoint(self, model, weights: dict) -> None:
        """Adjust the model or the gathered ``weights`` so the saved checkpoint loads in vLLM."""

    def vision_encoder(self, model, device):
        """The model's vision path for ``embed_init``, with a ``features`` method."""
        raise NotImplementedError

    # Sequence packing (mental_imagery/utils/packing.py)

    def packed_row_tokens(self, max_tokens: int) -> int | None:
        """Capacity of a packed row; None packs a micro-batch into one row (FlashAttention-2 varlen)."""
        return None

    def prepare_packing(self, model):
        """Patch layers that need the packed segments themselves; returns their ``LinearAttentionSegments``."""
        return None

    def packed_position_ids(self, model, batch: dict, plan) -> torch.Tensor:
        """Position ids of the packed micro-batch: (num_rows, row_length), restarting at 0 for each episode."""
        return plan.positions


class Qwen35(ModelFamily):
    """Qwen 3.5 (dense and MoE). Tokens from its tokenizer_config.json and chat template."""

    name = "qwen35"
    model_types = ("qwen3_5", "qwen3_5_moe")
    think_close_token = "</think>"
    turn_end_token = "<|im_end|>"
    eos_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"
    uses_mrope = True
    vllm_chat_template = None  # its own template is the training prompt

    def build_prompt_text(self, processor, user_text: str) -> str:
        """The chat template ends with "<think>\\n"; vLLM applies the same template."""
        return chat_template_prompt(processor, user_text)

    def fsdp_layer_classes(self) -> tuple[type, ...]:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5VisionBlock,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeVisionBlock,
        )

        return (
            Qwen3_5DecoderLayer,
            Qwen3_5VisionBlock,
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeVisionBlock,
        )

    def prepare_checkpoint(self, model, weights: dict) -> None:
        model._weight_conversions = []  # transformers >= 5.4 would rename Qwen 3.5 vision keys on save

    def vision_encoder(self, model, device):
        return QwenVisionEncoder(model, device)

    def prepare_packing(self, model):
        return patch_gated_delta_net_for_packing(model)

    def packed_position_ids(self, model, batch: dict, plan) -> torch.Tensor:
        """(4, rows, length): text positions (restarting per episode), then the 3 mRoPE rows.

        The mRoPE positions come from the padded micro-batch (one episode per row, so they also
        restart per episode). FlashAttention-2 reads the packing from the 2-D text positions.
        """
        with torch.no_grad():
            mrope, _ = model.model.get_rope_index(
                batch["input_ids"],
                batch["mm_token_type_ids"],
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch["attention_mask"],
            )
        packed = torch.stack([plan.pack(axis, fill_value=0) for axis in mrope])
        is_token = plan.pack(batch["attention_mask"].bool(), fill_value=False)
        packed = torch.where(
            is_token, packed, plan.positions
        )  # padding segments: their own positions
        return torch.cat([plan.positions[None], packed])


class Gemma4(ModelFamily):
    """Gemma 4 (base checkpoints). Tokens from its tokenizer_config.json; reasoning is a "thought" channel."""

    name = "gemma4"
    model_types = ("gemma4",)
    think_close_token = "<channel|>"
    turn_end_token = "<turn|>"
    eos_token = "<eos>"
    pad_token = "<pad>"
    uses_mrope = False  # 1-D RoPE in the language model
    vllm_chat_template = "gemma4.jinja"  # the base model ships no chat template

    def build_prompt_text(self, processor, user_text: str) -> str:
        """Thinking enabled by <|think|> in the system turn; the model turn opens the thought channel.

        The processor prepends <bos> and expands <|image|> into the image's soft tokens.
        """
        return (
            "<|turn>system\n<|think|>\n<turn|>\n"
            f"<|turn>user\n<|image|>{user_text}<turn|>\n"
            "<|turn>model\n<|channel>thought\n"
        )

    def processor_images(self, images: list) -> list:
        return [
            [image] for image in images
        ]  # the Gemma 4 processor pairs each text with a list of images

    def fsdp_layer_classes(self) -> tuple[type, ...]:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4AudioLayer,
            Gemma4TextDecoderLayer,
            Gemma4VisionEncoderLayer,
        )

        return (Gemma4TextDecoderLayer, Gemma4VisionEncoderLayer, Gemma4AudioLayer)

    def prepare_for_fsdp(self) -> None:
        patch_gemma4_shared_kv_states_for_fsdp2()

    def freeze_unused_modules(self, model) -> None:
        for module in (model.model.audio_tower, model.model.embed_audio):  # boards have no audio
            module.requires_grad_(False)

    def prepare_checkpoint(self, model, weights: dict) -> None:
        fill_gemma4_shared_kv_weights(weights, model.config.text_config)

    def vision_encoder(self, model, device):
        return GemmaVisionEncoder(model, device)

    def packed_row_tokens(self, max_tokens: int) -> int | None:
        """SDPA (its 512-wide global heads exceed FlashAttention-2) costs the square of a row's length,
        so rows hold at most one padded episode's worth of tokens."""
        return max_tokens


class InternVL(ModelFamily):
    """InternVL3.5 in the native transformers format (the "-HF" checkpoints): InternViT + Qwen3."""

    name = "internvl"
    model_types = ("internvl",)
    think_close_token = "</think>"
    turn_end_token = "<|im_end|>"
    eos_token = "<|im_end|>"
    pad_token = "<|endoftext|>"
    uses_mrope = False  # Qwen3 language model with 1-D RoPE
    vllm_chat_template = "internvl.jinja"  # its own template does not open the reasoning span

    def build_prompt_text(self, processor, user_text: str) -> str:
        """Its chat template ends at "<|im_start|>assistant\\n"; "<think>\\n" opens the reasoning span as for Qwen."""
        return chat_template_prompt(processor, user_text) + "<think>\n"

    def check_image_inputs(self, inputs, num_images: int) -> None:
        num_tiles = inputs["pixel_values"].shape[0]
        if num_tiles != num_images:  # the processor tiles non-square images (crop_to_patches)
            raise ValueError(
                f"{num_images} boards became {num_tiles} 448x448 tiles; each board must be one tile "
                "(256 soft tokens), so boards must be square"
            )

    def fsdp_layer_classes(self) -> tuple[type, ...]:
        from transformers.models.internvl.modeling_internvl import InternVLVisionLayer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        return (Qwen3DecoderLayer, InternVLVisionLayer)

    def vision_encoder(self, model, device):
        return InternVLVisionEncoder(model, device)


FAMILIES = {family.name: family for family in (Qwen35(), Gemma4(), InternVL())}


def resolve_family(name: str) -> ModelFamily:
    if name not in FAMILIES:
        raise ValueError(f"unknown model family {name!r}; expected one of {sorted(FAMILIES)}")
    return FAMILIES[name]


# Vision features for embed_init: each ``features`` call takes same-size images and returns
# (num_images * h * w, hidden) float64 features in language space and the feature grid (h, w).


class QwenVisionEncoder:
    """Vision blocks and patch merger (``model.visual``)."""

    def __init__(self, model, device):
        self.visual = model.model.visual.to(device).eval()

    @torch.no_grad()
    def features(self, processor, images, batch_size: int):
        features = []
        for start in range(0, len(images), batch_size):
            inputs = processor.image_processor(
                images=images[start : start + batch_size], return_tensors="pt"
            )
            grid_thw = inputs["image_grid_thw"].to(self.visual.device)
            output = self.visual(
                inputs["pixel_values"].to(self.visual.device, self.visual.dtype), grid_thw=grid_thw
            )
            features.append(output.pooler_output.double().cpu())
        merge = self.visual.spatial_merge_size
        return torch.cat(features), (grid_thw[0, 1].item() // merge, grid_thw[0, 2].item() // merge)


class GemmaVisionEncoder:
    """Vision tower (with its k x k soft-token pooling) and the ``embed_vision`` projection."""

    def __init__(self, model, device):
        self.vision_tower = model.model.vision_tower.to(device).eval()
        self.embed_vision = model.model.embed_vision.to(device).eval()

    @torch.no_grad()
    def features(self, processor, images, batch_size: int):
        parameter = next(self.vision_tower.parameters())
        features = []
        for start in range(0, len(images), batch_size):
            inputs = processor.image_processor(
                images=images[start : start + batch_size], return_tensors="pt"
            )
            position_ids = inputs["image_position_ids"].to(
                parameter.device
            )  # (x, y) per patch; (-1, -1) pads
            output = self.vision_tower(
                pixel_values=inputs["pixel_values"].to(parameter.device, parameter.dtype),
                pixel_position_ids=position_ids,
            )
            features.append(
                self.embed_vision(output.last_hidden_state).double().cpu()
            )  # padding stripped
        # Soft tokens are pooled k x k patches in row-major order.
        pool = processor.image_processor.pooling_kernel_size
        valid = position_ids[0][position_ids[0, :, 0] >= 0]
        grid = ((valid[:, 1].max().item() + 1) // pool, (valid[:, 0].max().item() + 1) // pool)
        features = torch.cat(features)
        if len(features) != len(images) * grid[0] * grid[1]:
            raise RuntimeError(
                f"{len(features)} soft tokens for {len(images)} images of {grid[0]}x{grid[1]}"
            )
        return features, grid


class InternVLVisionEncoder:
    """InternViT, pixel shuffle, and projector (``get_image_features``), on one 448x448 tile per image."""

    def __init__(self, model, device):
        model.model.language_model = None  # only the vision path is used
        self.model = model.model.to(device).eval()

    @torch.no_grad()
    def features(self, processor, images, batch_size: int):
        config = self.model.config
        features = []
        for start in range(0, len(images), batch_size):
            inputs = processor.image_processor(
                images=images[start : start + batch_size], return_tensors="pt"
            )
            output = self.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.model.device, self.model.dtype),
                vision_feature_layer=config.vision_feature_layer,
                vision_feature_select_strategy=config.vision_feature_select_strategy,
            )
            features.append(
                output.pooler_output.double().cpu()
            )  # (tiles, h * w, hidden), row-major
        features = torch.cat(features)
        side = int(features.shape[1] ** 0.5)
        if len(features) != len(images) or side * side != features.shape[1]:
            raise RuntimeError(
                f"expected one square tile per image, got features of shape {tuple(features.shape)}"
            )
        return features.reshape(-1, features.shape[-1]), (side, side)


def fill_gemma4_shared_kv_weights(weights: dict, text_config) -> None:
    """Copy each KV-sharing layer's source k_proj / v_proj / k_norm weights under its own keys.

    Transformers has no such modules in the layers that reuse an earlier layer's
    keys and values, but the released checkpoints store them and vLLM loads them.
    """
    first_shared = text_config.num_hidden_layers - text_config.num_kv_shared_layers
    earlier_types = text_config.layer_types[:first_shared]
    for layer in range(first_shared, text_config.num_hidden_layers):
        source = len(earlier_types) - 1 - earlier_types[::-1].index(text_config.layer_types[layer])
        for suffix in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
            source_key = f"model.language_model.layers.{source}.self_attn.{suffix}"
            if source_key in weights:
                weights[f"model.language_model.layers.{layer}.self_attn.{suffix}"] = weights[
                    source_key
                ].clone()


class SharedKVStates:
    """Gemma 4's per-forward store of the keys and values that later layers reuse.

    FSDP2's pre-forward hook flattens and rebuilds each layer's keyword arguments
    as a pytree, so a plain dict reaches every layer as a fresh copy and later
    layers miss the keys and values stored by earlier ones. This class is not a
    registered pytree node, so the same object reaches every layer.
    """

    __slots__ = ("states",)

    def __init__(self):
        self.states = {}

    def __getitem__(self, layer):
        return self.states[layer]

    def __setitem__(self, layer, key_value):
        self.states[layer] = key_value

    def __contains__(self, layer):
        return layer in self.states


def patch_gemma4_shared_kv_states_for_fsdp2() -> None:
    """Replace ``Gemma4TextModel.forward`` by a copy of transformers 5.6.2's that passes ``SharedKVStates``.

    The container is the only change to the upstream forward.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs

    if getattr(modeling_gemma4.Gemma4TextModel.forward, "patched", False):
        return
    if transformers.__version__ != "5.6.2":
        raise RuntimeError(
            f"the Gemma 4 FSDP2 patch copies transformers 5.6.2's Gemma4TextModel.forward; "
            f"re-check it against transformers {transformers.__version__}"
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        per_layer_inputs=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }
        shared_kv_states = SharedKVStates()
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values
        )

    forward = merge_with_config_defaults(capture_outputs(forward))
    forward.patched = True
    modeling_gemma4.Gemma4TextModel.forward = forward


def patch_flash_attention_for_mrope() -> None:
    """Keep transformers from treating Qwen 3.5's 3-D mRoPE positions as packed sequences.

    Otherwise FlashAttention-2 reads out of bounds
    (https://github.com/huggingface/transformers/issues/44910).
    """
    import transformers.modeling_flash_attention_utils as flash_utils

    original = flash_utils._is_packed_sequence
    if getattr(original, "patched", False):
        return

    def is_packed_sequence(position_ids, batch_size):
        if position_ids is not None and position_ids.dim() > 2:
            return False
        return original(position_ids, batch_size)

    is_packed_sequence.patched = True
    flash_utils._is_packed_sequence = is_packed_sequence


GATED_DELTA_NET_CLASSES = ("Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet")


def patch_gated_delta_net_for_packing(model):
    """Give Qwen 3.5's Gated DeltaNet layers the segments of a packed micro-batch.

    transformers calls their short convolution with ``seq_idx=None`` and their chunked delta rule
    without ``cu_seqlens``, so packed episodes would share recurrent state. The kernels take both;
    each layer's kernel functions are wrapped to pass the boundaries in the returned
    ``LinearAttentionSegments`` (set per micro-batch; without segments the wrappers change nothing).
    """
    from mental_imagery.utils.packing import LinearAttentionSegments

    segments = LinearAttentionSegments()
    layers = [
        module for module in model.modules() if type(module).__name__ in GATED_DELTA_NET_CLASSES
    ]
    if not layers:
        raise ValueError("no Gated DeltaNet layers found to patch for sequence packing")
    for layer in layers:
        convolution, delta_rule = layer.causal_conv1d_fn, layer.chunk_gated_delta_rule
        if convolution is None or not delta_rule.__module__.startswith("fla."):
            raise RuntimeError(
                "sequence packing needs the causal-conv1d and flash-linear-attention kernels "
                "(the torch fallbacks do not take segment boundaries); install them "
                "(scripts/kernels.sh) or train padded with training.sequence_packing=false"
            )
        layer.causal_conv1d_fn = _segmented_convolution(convolution, segments)
        layer.chunk_gated_delta_rule = _segmented_delta_rule(delta_rule, segments)
    return segments


def _segmented_convolution(convolution, segments):
    def segmented(*, x, weight, bias=None, activation=None, seq_idx=None):
        if segments.seq_idx is not None:
            if x.shape[0] != 1:
                raise ValueError("packed Gated DeltaNet input must be a single row")
            if x.stride(1) != 1:  # causal-conv1d takes seq_idx only for channel-last inputs
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
            seq_idx = segments.seq_idx
        return convolution(x=x, weight=weight, bias=bias, activation=activation, seq_idx=seq_idx)

    return segmented


def _segmented_delta_rule(delta_rule, segments):
    def segmented(*args, **kwargs):
        if segments.cu_seqlens is not None:
            kwargs["cu_seqlens"] = segments.cu_seqlens
        return delta_rule(*args, **kwargs)

    return segmented
