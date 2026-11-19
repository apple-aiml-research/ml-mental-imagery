#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Config tests: representative experiment methods compose and the pipeline's artifacts line up.

The preparation steps (FSQ training, embedding init) write files that training
reads by path; if the paths drifted apart, the README pipeline would break.
"""

from __future__ import annotations

import glob
import os

import pytest

from mental_imagery.model.families import FAMILIES
from mental_imagery.model.loading import DTYPES
from tests.conftest import CONFIG_DIR, compose_config

EXPERIMENTS = [
    "qwen3_5_9b/action_supervision",
    "qwen3_5_9b/visual_supervision",
    "qwen3_5_9b/visual_cot_64",
    "qwen3_5_9b/visual_cot_128",
    "qwen3_5_9b/state_cot",
    "qwen3_5_9b/boxes_cot",
]
MODELS = sorted(
    os.path.basename(path)[: -len(".yaml")]
    for path in glob.glob(os.path.join(CONFIG_DIR, "model", "*.yaml"))
)
EPISODES_PER_STEP_ON_8_GPUS = 512
# Prompts of these families hold 256 image tokens (Qwen: 64); with long boards, 1024 tokens truncate too much.
WIDE_PROMPT_FAMILIES = {"gemma4", "internvl"}
LONG_BOARD_METHODS = {"visual_cot_128", "state_cot", "boxes_cot"}


def expected_max_tokens(experiment: str, family: str) -> int:
    method = experiment.split("/")[1]
    return 1536 if family in WIDE_PROMPT_FAMILIES and method in LONG_BOARD_METHODS else 1024


def train_config(experiment: str):
    return compose_config("train", [f"experiment={experiment}"])


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_batch_and_method_settings(experiment):
    config = train_config(experiment)
    assert config.run_name == experiment
    assert (
        config.training.per_gpu_batch_size * config.training.gradient_accumulation_steps * 8
        == EPISODES_PER_STEP_ON_8_GPUS
    )
    assert config.training.sequence_packing is True
    assert config.visual_tokens.enabled == ("visual_cot" in experiment)
    assert config.visual_head.enabled == (
        "visual_supervision" in experiment or "visual_probe" in experiment
    )
    assert config.visual_head.stop_gradient == ("visual_probe" in experiment)
    assert config.text_cot.enabled == ("state_cot" in experiment or "boxes_cot" in experiment)
    assert config.text_cot.boards == ("boxes" if "boxes_cot" in experiment else "state")
    assert config.training.max_tokens == expected_max_tokens(experiment, config.model.family)


@pytest.mark.parametrize("model", MODELS)
def test_every_model_config_names_a_family_and_its_settings(model):
    config = compose_config("embed_init", [f"model={model}"]).model
    assert config.short_name == model
    assert config.family in FAMILIES
    assert config.dtype in DTYPES
    assert config.new_token_init == "mean_noise"  # Visual CoT methods switch it to embed_init
    # Gemma 4's full-attention layers have head_dim 512, above FlashAttention-2's limit.
    assert config.attn_implementation == (
        "sdpa" if config.family == "gemma4" else "flash_attention_2"
    )


@pytest.mark.parametrize("experiment", ["qwen3_5_9b/visual_cot_64", "qwen3_5_9b/visual_cot_64_fsq"])
def test_visual_token_runs_read_the_table_that_embed_init_writes(experiment):
    config = train_config(experiment)
    embed = compose_config(
        "embed_init",
        [f"model={config.model.short_name}", f"visual_tokens={config.visual_tokens.tokenizer}"],
    )
    assert config.visual_tokens.embed_init_path == embed.output_path
    assert embed.visual_tokens.grid_size is None  # built at native resolution
    assert embed.visual_tokens.fsq_checkpoint == config.visual_tokens.fsq_checkpoint


@pytest.mark.parametrize(
    ("experiment", "preset", "section"),
    [
        ("qwen3_5_9b/visual_supervision", "board_256", "visual_head"),
        ("qwen3_5_9b/visual_cot_64_fsq", "board_multires", "visual_tokens"),
    ],
)
def test_runs_read_the_fsq_tokenizer_that_train_fsq_writes(experiment, preset, section):
    fsq = compose_config("train_fsq", [f"fsq={preset}"])
    assert train_config(experiment)[section].fsq_checkpoint == fsq.output_path


def test_multires_tokenizer_covers_the_visual_token_grids():
    resolutions = compose_config("train_fsq", ["fsq=board_multires"]).resolutions
    assert 4 * 16 in resolutions and 8 * 16 in resolutions


@pytest.mark.parametrize("model", ["qwen3_5_0_8b", "qwen3_5_2b", "qwen3_5_9b"])
def test_base_model_evaluation_composes_for_each_model(model):
    config = compose_config("eval_base", [f"model={model}", "output_dir=unused"])
    assert config.max_new_tokens < config.max_tokens
    assert len(config.games) == 12


def test_token_replacement_composes():
    config = compose_config(
        "token_replacement", ["checkpoint=unused", "mode=ground_truth", "output_dir=unused"]
    )
    assert len(config.games) == 12
