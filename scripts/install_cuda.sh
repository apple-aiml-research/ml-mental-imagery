#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Install the CUDA 12.9 compiler and headers from NVIDIA's apt repository (Ubuntu; root or sudo).
# Building the training kernels (scripts/kernels.sh build) and the kernels vLLM compiles when an
# evaluation starts (FlashInfer, mental_imagery/utils/inference.py) both use /usr/local/cuda-12.9.
# Not cuda-compat-12-9: its libraries fail on newer drivers (CUDA error 803). Safe to re-run.
set -euo pipefail
CUDA_PACKAGES=(cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-nvrtc-dev-12-9 libcublas-dev-12-9 libcurand-dev-12-9)
if ! dpkg -s "${CUDA_PACKAGES[@]}" >/dev/null 2>&1; then
  sudo=""
  [ "$(id -u)" -eq 0 ] || sudo="sudo"
  echo "==> installing ${CUDA_PACKAGES[*]}"
  $sudo apt-get update -qq
  $sudo apt-get install -y -qq --no-install-recommends "${CUDA_PACKAGES[@]}"
fi
/usr/local/cuda-12.9/bin/nvcc --version | tail -1
