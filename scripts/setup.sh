#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Install everything needed to train and evaluate:
#   - checks for uv; installs the CUDA 12.9 compiler (scripts/install_cuda.sh, apt) and Rust (rustup)
#   - downloads the fonts the engine embeds (scripts/download_fonts.py; not distributed here)
#   - downloads OPTED and recreates the Anagram word list the engine embeds (scripts/download_words.py;
#     not distributed here)
#   - scripts/kernels.sh build: the Python env from uv.lock, release build of the Rust engine, and the
#     FlashAttention-2 / causal-conv1d kernels compiled from source for this machine's GPUs
#     (often tens of minutes; `install` instead when kernels/ already holds them)
#   - runs the Python tests, including GPU tests when available
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
  echo "uv not found; install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi
echo "==> checking CUDA 12.9 and Rust"
scripts/install_cuda.sh  # CUDA 12.9 compiler and headers (apt)
export CUDA_HOME=/usr/local/cuda-12.9
if ! command -v cargo >/dev/null && [ -x "$HOME/.cargo/bin/cargo" ]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v cargo >/dev/null; then
  echo "Rust not found; installing it with rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "$(uv --version), $(cargo --version), nvcc $("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*')"

echo "==> downloading and verifying engine assets"
uv run --no-project --with fonttools==4.62.1 python scripts/download_fonts.py  # fonts the engine embeds
uv run --no-project python scripts/download_words.py  # Anagram word list the engine embeds
if [ -f kernels/build_info.json ]; then
  scripts/kernels.sh install kernels  # fails if they were built for another uv.lock: rebuild them
else
  scripts/kernels.sh build kernels
fi
echo "==> running Python tests, including GPU tests when available"
uv run --locked pytest -q
echo "Setup complete."
