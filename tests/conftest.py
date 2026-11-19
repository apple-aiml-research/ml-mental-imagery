#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared fixtures: config composition and byte-level mock tokenizers."""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

from mental_imagery.model.families import FAMILIES, STRUCTURAL_TOKENS, visual_token

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mental_imagery", "configs")
)
CODEBOOK_SIZE = 16384


def compose_config(name: str, overrides: list[str]):
    """Composed config with every value resolved (fails on any missing ``???`` value)."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name=name, overrides=overrides)
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))


class MockTokenizer:
    """The parts of a Hugging Face tokenizer the engine helpers use."""

    unk_token_id = None

    def __init__(self, backend: Tokenizer):
        self.backend_tokenizer = backend

    def convert_tokens_to_ids(self, token: str):
        return self.backend_tokenizer.token_to_id(token)


def byte_level_tokenizer(special_tokens: list[str]) -> Tokenizer:
    """Byte-level tokenizer (encodes any text) with ``special_tokens`` after the 256 byte symbols."""
    vocab = {
        symbol: index for index, symbol in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    backend = Tokenizer(BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(list(dict.fromkeys(special_tokens)))
    return backend


def make_mock_tokenizer(family) -> MockTokenizer:
    """Mock tokenizer with ``family``'s engine tokens, the structural tokens, and the visual tokens."""
    special = [
        family.eos_token,
        family.pad_token,
        family.turn_end_token,
        family.think_close_token,
        *STRUCTURAL_TOKENS,
    ]
    return MockTokenizer(
        byte_level_tokenizer(special + [visual_token(code) for code in range(CODEBOOK_SIZE)])
    )


@pytest.fixture(scope="session")
def mock_tokenizer() -> MockTokenizer:
    """Mock tokenizer with the Qwen 3.5 special tokens."""
    return make_mock_tokenizer(FAMILIES["qwen35"])
