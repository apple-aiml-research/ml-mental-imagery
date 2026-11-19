#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Model families: engine tokens, prompts, train/eval prompt parity, and the Gemma 4 patches.

Processors are the real transformers classes with byte-level mock tokenizers and
the image settings of the released checkpoints, so no model files are needed;
the tests with the released processors run when their files are in the local
Hugging Face cache (no download).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from huggingface_hub import try_to_load_from_cache
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from tokenizers.processors import TemplateProcessing
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import (
    AutoProcessor,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4Processor,
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLProcessor,
    PreTrainedTokenizerFast,
)
from transformers.models.gemma4 import modeling_gemma4
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor
from transformers.models.got_ocr2.image_processing_got_ocr2 import GotOcr2ImageProcessor
from transformers.models.internvl.video_processing_internvl import InternVLVideoProcessor
from vllm.renderers.hf import _detect_content_format

from mental_imagery.model.families import (
    FAMILIES,
    STRUCTURAL_TOKENS,
    fill_gemma4_shared_kv_weights,
    patch_gemma4_shared_kv_states_for_fsdp2,
    resolve_family,
)
from mental_imagery.model.loading import load_pretrained
from mental_imagery.puzzles import build_game_configs
from mental_imagery.utils.data import build_prompts, process_prompts, reprocess_prompt_images
from tests.conftest import byte_level_tokenizer, compose_config, make_mock_tokenizer
from tests.test_engine import FULL_EPISODE, load_games, make_engine

USER_TEXT = "Push every box onto a goal. Actions: up, down, left, right"


BOARD = Image.new("RGB", (256, 256), (40, 120, 200))
# Image soft tokens per 256x256 board in the prompt.
IMAGE_TOKENS = {
    "gemma4": ("<|image|>", 256),
    "internvl": ("<IMG_CONTEXT>", 256),
}
# The model turn opens the reasoning span, so generation starts with episode content.
PROMPT_ENDINGS = {
    "gemma4": "<turn|>\n<|turn>model\n<|channel>thought\n",
    "internvl": "<|im_end|>\n<|im_start|>assistant\n<think>\n",
}
# Renders one image and a text as InternVL3.5-1B-HF's own chat template does (checked against the released template).
INTERNVL_OWN_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% for item in message['content'] %}{% if item['type'] == 'image' %}{{ '<IMG_CONTEXT>\\n' }}"
    "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def gemma4_processor():
    """Gemma 4 processor with its default image settings (16 px patches, 3x3 pooling) and a mock tokenizer."""
    special = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<|channel>",
        "<channel|>",
        "<|image>",
        "<image|>",
        "<|image|>",
        "<|audio>",
        "<audio|>",
        "<|audio|>",
        *STRUCTURAL_TOKENS,
    ]
    backend = byte_level_tokenizer(special)
    backend.post_processor = TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", backend.token_to_id("<bos>"))]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        extra_special_tokens={
            "image_token": "<|image|>",
            "boi_token": "<|image>",
            "eoi_token": "<image|>",
            "audio_token": "<|audio|>",
            "boa_token": "<|audio>",
            "eoa_token": "<audio|>",
        },
    )
    return Gemma4Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(),
        image_processor=Gemma4ImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Gemma4VideoProcessor(),
    )


def internvl_processor():
    """InternVL processor with the -HF checkpoints' image settings (one 448 px tile) and a mock tokenizer."""
    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<img>",
        "</img>",
        "<IMG_CONTEXT>",
        "<video>",
        *STRUCTURAL_TOKENS,
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=byte_level_tokenizer(special),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        extra_special_tokens={
            "start_image_token": "<img>",
            "end_image_token": "</img>",
            "context_image_token": "<IMG_CONTEXT>",
            "video_token": "<video>",
        },
    )
    image_processor = GotOcr2ImageProcessor(
        crop_to_patches=False, size={"height": 448, "width": 448}, min_patches=1, max_patches=12
    )
    return InternVLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=InternVLVideoProcessor(),
        image_seq_length=256,
        chat_template=INTERNVL_OWN_TEMPLATE,
    )


MOCK_PROCESSORS = {"gemma4": gemma4_processor, "internvl": internvl_processor}


def cached_processor(model: str):
    """The model config and its released processor from the local Hugging Face cache; skips if absent."""
    model_config = compose_config("embed_init", [f"model={model}"]).model
    if model_config.revision is None:
        pytest.skip(f"{model_config.name} has no pinned revision")
    if not isinstance(
        try_to_load_from_cache(
            model_config.name, "tokenizer_config.json", revision=model_config.revision
        ),
        str,
    ):
        pytest.skip(
            f"{model_config.name}@{model_config.revision} is not in the local Hugging Face cache"
        )
    return model_config, AutoProcessor.from_pretrained(
        model_config.name, revision=model_config.revision, local_files_only=True
    )


def rollout_with(images):
    return SimpleNamespace(
        prompts=["Push every box onto a goal."] * len(images),
        initial_images=images,
        actions=["up, down, left, right"] + [""] * (len(images) - 1),
    )


def vllm_prompt_ids(processor, family) -> list[int]:
    """Prompt ids as vLLM builds them with ``family``'s chat template.

    vLLM passes the content of a template that loops over it as typed parts
    ({"type": "image"}, {"type": "text", ...}) and tokenizes the rendered prompt
    without adding special tokens. (Qwen 3.5's own template loops inside a
    macro, which vLLM reads as plain-string content: it then renders the image
    placeholder and the text joined by a newline that training does not have.)
    """
    with open(family.vllm_chat_template_path()) as f:
        template = f.read()
    assert _detect_content_format(template, default="string") == "openai"
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]}
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, chat_template=template, add_generation_prompt=True, tokenize=False
    )
    images = family.processor_images([BOARD])
    return list(processor(text=[prompt], images=images, add_special_tokens=False)["input_ids"][0])


def training_prompt_ids(processor, family) -> list[int]:
    images = family.processor_images([BOARD])
    return list(
        processor(text=[family.build_prompt_text(processor, USER_TEXT)], images=images)[
            "input_ids"
        ][0]
    )


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown model family"):
        resolve_family("llava")


def test_a_checkpoint_of_another_family_raises():
    with pytest.raises(ValueError, match="not a gemma4 model"):
        FAMILIES["gemma4"].check_model_type(SimpleNamespace(model_type="qwen3_5"))


@pytest.mark.parametrize("family", FAMILIES.values(), ids=list(FAMILIES))
def test_engine_closes_reasoning_and_the_turn_with_the_familys_tokens(family):
    tokenizer = make_mock_tokenizer(family)
    token_ids = {
        role: tokenizer.convert_tokens_to_ids(getattr(family, role))
        for role in ("think_close_token", "turn_end_token", "eos_token", "pad_token")
    }
    assert family.engine_token_kwargs(tokenizer) == {
        "think_close_token": family.think_close_token,
        "im_end_token": family.turn_end_token,
        "eos_token_id": token_ids["eos_token"],
        "pad_token_id": token_ids["pad_token"],
    }
    puzzle_engine = make_engine(tokenizer, {"enabled": False}, family=family)
    assert puzzle_engine.special_tokens["eos"] == token_ids["eos_token"]
    assert puzzle_engine.special_tokens["im_end"] == token_ids["turn_end_token"]
    sokoban = [entry for entry in load_games("benchmark") if entry.name == "sokoban"]
    rollout = puzzle_engine.loader(
        games=build_game_configs(sokoban), n=2, prefetch=0
    ).next_with_seeds(seeds=[0, 1])
    inputs, _, lengths, _ = rollout.teacher_force_labels(FULL_EPISODE)
    for sequence, length in zip(inputs, lengths, strict=True):
        assert np.count_nonzero(sequence[:length] == token_ids["think_close_token"]) == 1
        assert sequence[length - 1] == token_ids["turn_end_token"]


def test_a_family_token_missing_from_the_tokenizer_raises(mock_tokenizer):
    with pytest.raises(ValueError, match="<eos>"):
        FAMILIES["gemma4"].engine_token_kwargs(mock_tokenizer)


@pytest.mark.parametrize("family", FAMILIES.values(), ids=list(FAMILIES))
def test_fsdp_layer_classes_are_transformers_modules(family):
    layer_classes = family.fsdp_layer_classes()
    assert layer_classes and all(
        issubclass(layer_class, torch.nn.Module) for layer_class in layer_classes
    )


@pytest.mark.parametrize("family_name", sorted(MOCK_PROCESSORS))
def test_prompts_hold_one_board_and_open_the_reasoning_span(family_name):
    family, processor = FAMILIES[family_name], MOCK_PROCESSORS[family_name]()
    processor.tokenizer.padding_side = "left"
    batch = build_prompts(
        processor, rollout_with([BOARD, BOARD]), max_tokens=1536, device="cpu", family=family
    )
    image_token, tokens_per_board = IMAGE_TOKENS[family_name]
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    assert (batch["input_ids"] == image_token_id).sum(dim=1).tolist() == [tokens_per_board] * 2
    gen_start = int(batch["prompt_lengths"].max())
    assert not batch["attention_mask"][
        :, gen_start:
    ].any()  # left-padded: every prompt ends at gen_start
    for row in batch["input_ids"]:
        assert processor.tokenizer.decode(row[:gen_start]).endswith(PROMPT_ENDINGS[family_name])


def test_internvl_rejects_boards_that_would_be_tiled():
    with pytest.raises(ValueError, match="must be one tile"):
        build_prompts(
            internvl_processor(),
            rollout_with([Image.new("RGB", (512, 256))]),
            max_tokens=1536,
            device="cpu",
            family=FAMILIES["internvl"],
        )


def random_board(seed: int):
    return Image.fromarray(
        np.random.default_rng(seed).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    )


def check_reprocessed_prompt_images(processor, family):
    """Random truncation swaps some prompt images for boards of the same size: recomputing only the image
    tensors must give what processing the prompts again gives."""
    rollout = rollout_with([random_board(0), random_board(1), random_board(2)])
    inputs = process_prompts(processor, rollout, family)
    rollout.initial_images = [random_board(3), rollout.initial_images[1], random_board(4)]
    reprocessed = reprocess_prompt_images(processor, rollout, family, inputs)
    fresh = process_prompts(processor, rollout, family)
    assert reprocessed.keys() == fresh.keys()
    for key in fresh:
        assert torch.equal(reprocessed[key], fresh[key]), key


@pytest.mark.parametrize("family_name", sorted(MOCK_PROCESSORS))
def test_reprocessed_prompt_images_match_processing_the_prompts_again(family_name):
    check_reprocessed_prompt_images(MOCK_PROCESSORS[family_name](), FAMILIES[family_name])


@pytest.mark.parametrize("family_name", sorted(MOCK_PROCESSORS))
def test_vllm_chat_template_gives_the_training_prompt(family_name):
    """Evaluation must see the training prompt token for token (vLLM applies its own chat template)."""
    family, processor = FAMILIES[family_name], MOCK_PROCESSORS[family_name]()
    assert vllm_prompt_ids(processor, family) == training_prompt_ids(processor, family)


def test_qwen_prompt_is_its_chat_template_with_the_reasoning_span_open():
    model_config, processor = cached_processor("qwen3_5_9b")
    prompt = FAMILIES[model_config.family].build_prompt_text(processor, USER_TEXT)
    assert prompt == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        + USER_TEXT
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


# Six layers, the last two reusing the keys and values of the last earlier layer of their type.
TINY_GEMMA4_TEXT = dict(
    vocab_size=300,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=6,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    global_head_dim=16,
    num_kv_shared_layers=2,
    sliding_window=4,
    hidden_size_per_layer_input=8,
    vocab_size_per_layer_input=300,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
)


def tiny_gemma4():
    """Randomly initialized Gemma 4 with text, vision (16 px patches, 3x3 pooling), and audio towers."""
    torch.manual_seed(0)
    config = Gemma4Config(
        text_config=TINY_GEMMA4_TEXT,
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            head_dim=16,
            num_key_value_heads=2,
        ),
        audio_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2),
    )
    return Gemma4ForConditionalGeneration(config).eval()


def tiny_internvl():
    """Randomly initialized InternVL: InternViT on 448 px tiles of 14 px patches, pixel shuffle 0.5, and Qwen3."""
    torch.manual_seed(0)
    config = InternVLConfig(
        text_config=dict(
            model_type="qwen3",
            vocab_size=300,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=[448, 448],
            patch_size=[14, 14],
        ),
        downsample_ratio=0.5,
    )
    return InternVLForConditionalGeneration(config).eval()


TINY_MODELS = {"gemma4": tiny_gemma4, "internvl": tiny_internvl}


@pytest.mark.parametrize("family_name", sorted(TINY_MODELS))
def test_fsdp_shards_every_kind_of_block(family_name):
    family, model = FAMILIES[family_name], TINY_MODELS[family_name]()
    sharded = {
        type(module)
        for module in model.modules()
        if isinstance(module, family.fsdp_layer_classes())
    }
    assert sharded == set(family.fsdp_layer_classes())


@pytest.mark.parametrize("family_name", sorted(TINY_MODELS))
def test_embed_init_features_are_the_prompt_image_features_on_the_board_grid(family_name):
    family, model, processor = (
        FAMILIES[family_name],
        TINY_MODELS[family_name](),
        MOCK_PROCESSORS[family_name](),
    )
    image_inputs = processor.image_processor(images=[BOARD, BOARD, BOARD], return_tensors="pt")
    model_inputs = {
        key: image_inputs[key]
        for key in ("pixel_values", "image_position_ids")
        if key in image_inputs
    }
    with torch.no_grad():
        expected = model.model.get_image_features(
            **model_inputs
        ).pooler_output  # what the language model receives
    features, grid = family.vision_encoder(model, "cpu").features(
        processor, [BOARD, BOARD, BOARD], batch_size=2
    )
    assert grid == (16, 16) and features.dtype == torch.float64
    assert torch.allclose(features.float(), expected.reshape(features.shape).float(), atol=1e-6)


@pytest.mark.parametrize("family_name", sorted(TINY_MODELS))
def test_checkpoint_saves_and_reloads_for_training_and_vllm(family_name, tmp_path):
    family, model = FAMILIES[family_name], TINY_MODELS[family_name]()
    weights = dict(model.state_dict())
    family.prepare_checkpoint(model, weights)
    model.save_pretrained(tmp_path, state_dict=weights)
    model_config = OmegaConf.create({"family": family_name, "attn_implementation": "sdpa"})
    reloaded = load_pretrained(
        str(tmp_path), revision=None, model_config=model_config
    )  # as a resumed run does
    assert reloaded.state_dict().keys() == model.state_dict().keys()
    assert all(
        torch.equal(value, reloaded.state_dict()[key]) for key, value in model.state_dict().items()
    )
    if family_name == "gemma4":
        assert not any(
            parameter.requires_grad for parameter in reloaded.model.audio_tower.parameters()
        )
        assert all(
            parameter.requires_grad for parameter in reloaded.model.vision_tower.parameters()
        )
        with safe_open(
            tmp_path / "model.safetensors", "pt"
        ) as saved:  # vLLM loads every layer's k/v projections
            saved_keys = set(saved.keys())
        assert all(
            f"model.language_model.layers.{layer}.self_attn.{suffix}" in saved_keys
            for layer in range(TINY_GEMMA4_TEXT["num_hidden_layers"])
            for suffix in ("k_proj.weight", "v_proj.weight")
        )


def test_gemma4_checkpoint_stores_the_shared_kv_weights_of_their_source_layers():
    model = tiny_gemma4()
    weights = dict(model.state_dict())
    fill_gemma4_shared_kv_weights(weights, model.config.text_config)
    for index, layer in enumerate(model.model.language_model.layers):
        source = (
            layer.self_attn.kv_shared_layer_index if layer.self_attn.is_kv_shared_layer else index
        )
        for suffix in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
            filled = weights[f"model.language_model.layers.{index}.self_attn.{suffix}"]
            source_weight = weights[f"model.language_model.layers.{source}.self_attn.{suffix}"]
            assert torch.equal(filled, source_weight)
            assert (filled.data_ptr() == source_weight.data_ptr()) == (
                source == index
            )  # copies, not shared storage


@pytest.fixture
def single_process_group():
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531")
    dist.init_process_group("gloo", rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,))
    dist.destroy_process_group()


def test_gemma4_kv_sharing_survives_fsdp2_only_with_the_patch(single_process_group, monkeypatch):
    monkeypatch.setattr(
        modeling_gemma4.Gemma4TextModel, "forward", modeling_gemma4.Gemma4TextModel.forward
    )
    config = Gemma4TextConfig(**TINY_GEMMA4_TEXT)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    reference_model = modeling_gemma4.Gemma4TextModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 9))
    reference = reference_model(input_ids=input_ids, use_cache=False).last_hidden_state

    model = modeling_gemma4.Gemma4TextModel(config)
    model.load_state_dict(reference_model.state_dict())
    for layer in model.layers:
        fully_shard(layer, mesh=single_process_group)
    fully_shard(model, mesh=single_process_group)
    with pytest.raises(KeyError):  # the shared-KV dict reaches each layer as a copy
        model(input_ids=input_ids, use_cache=False)
    patch_gemma4_shared_kv_states_for_fsdp2()
    output = model(input_ids=input_ids, use_cache=False).last_hidden_state
    output.sum().backward()
    assert torch.equal(output, reference)
