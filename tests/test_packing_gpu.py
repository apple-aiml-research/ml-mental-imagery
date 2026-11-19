#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Sequence packing on GPU: packed episodes compute what each episode computes alone.

Tiny random Qwen 3.5 language models (dense and MoE), whose Gated DeltaNet layers are patched by
``patch_gated_delta_net_for_packing``: a packed row of three episodes must give each episode the
logits it gets alone, and an unpatched (leaky) row must not. Needs CUDA, FlashAttention-2 and the
causal-conv1d / flash-linear-attention kernels; skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

from mental_imagery.model.families import FAMILIES
from mental_imagery.utils.packing import assign_rows, build_plan, linear_attention_boundaries

kernels = pytest.importorskip("fla.ops.gated_delta_rule")
pytest.importorskip("causal_conv1d")
pytest.importorskip("flash_attn")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

LENGTHS = [37, 64, 21]
TINY = dict(
    hidden_size=256,
    num_hidden_layers=4,
    layer_types=["linear_attention", "full_attention"] * 2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    intermediate_size=512,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    vocab_size=1000,
)


def tiny_model(kind: str):
    if kind == "dense":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as Model

        config = Config(**TINY)
    else:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeTextConfig as Config,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForCausalLM as Model,
        )

        config = Config(
            **TINY,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(0)
    return Model(config).to("cuda", torch.bfloat16).eval()


def packed_inputs(episodes):
    """One packed row of ``episodes`` and its plan (positions restart per episode)."""
    width = max(LENGTHS)
    attention_mask = torch.zeros(len(episodes), width, dtype=torch.bool, device="cuda")
    tokens = torch.zeros(len(episodes), width, dtype=torch.long, device="cuda")
    for row, episode in enumerate(episodes):
        attention_mask[row, : len(episode)] = True
        tokens[row, : len(episode)] = episode
    plan = build_plan(
        attention_mask, assign_rows(LENGTHS, None), num_rows=1, row_length=sum(LENGTHS)
    )
    return plan.pack(tokens, fill_value=0), plan


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_packed_episodes_match_each_episode_alone(kind):
    model = tiny_model(kind)
    segments = FAMILIES["qwen35"].prepare_packing(model)
    torch.manual_seed(1)
    episodes = [torch.randint(0, 1000, (length,), device="cuda") for length in LENGTHS]
    with torch.no_grad():
        alone = torch.cat(
            [model(input_ids=episode[None]).logits[0] for episode in episodes]
        ).float()
        tokens, plan = packed_inputs(episodes)
        segments.set(*linear_attention_boundaries(plan))
        packed = model(input_ids=tokens, position_ids=plan.positions).logits[0].float()
        segments.clear()
        leaky = model(input_ids=tokens).logits[0].float()
    scale = alone.abs().mean()
    assert (packed - alone).abs().mean() < 0.02 * scale
    assert (leaky - alone).abs().mean() > 5 * (packed - alone).abs().mean()
