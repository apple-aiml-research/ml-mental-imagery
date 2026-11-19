#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# The CUDA kernel packages of the `kernels` extra, flash-attn and causal-conv1d, which compile from
# source against the locked torch.
#
#   scripts/kernels.sh build [dir]     compile wheels of the locked flash-attn and causal-conv1d for this
#                                      machine's GPUs into dir (default: kernels/), then install them
#   scripts/kernels.sh install <dir>   install wheels made by `build` (checked against uv.lock and this
#                                      machine's GPUs)
#
# Both first install every other locked package (`uv sync --locked --extra kernels` without the two)
# and end by running both kernels on the GPU. Building needs the CUDA compiler (nvcc), version 12.8
# or newer for Blackwell GPUs (compute capability 10.0 and up, e.g. B200): older versions build
# without error but leave out their kernels.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPILED_PACKAGES=(flash-attn causal-conv1d)
BUILD_INFO=build_info.json

fail() {
  echo "kernels.sh: $*" >&2
  exit 1
}

sync_other_packages() {
  local arguments=()
  for package in "${COMPILED_PACKAGES[@]}"; do
    arguments+=(--no-install-package "$package")
  done
  uv sync --locked --extra kernels "${arguments[@]}"
}

# Compute capabilities of this machine's GPUs, without the dot (e.g. 100 for 10.0), sorted and unique.
gpu_architectures() {
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u
}

check_gpu_kernels() {
  .venv/bin/python -c "
import torch, flash_attn, causal_conv1d
q = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16)
flash_attn.flash_attn_func(q, q, q)
x = torch.randn(1, 64, 16, device='cuda', dtype=torch.bfloat16)
causal_conv1d.causal_conv1d_fn(x, torch.randn(64, 4, device='cuda', dtype=torch.bfloat16))
torch.cuda.synchronize()
print('CUDA kernels run on', torch.cuda.get_device_name())
"
}

install_wheels() {
  local wheel_dir="$1"
  [ -f "$wheel_dir/$BUILD_INFO" ] || fail "$wheel_dir/$BUILD_INFO not found; make the wheels with: scripts/kernels.sh build"
  sync_other_packages
  # The wheels must come from this uv.lock and contain kernels for every GPU here.
  .venv/bin/python - "$wheel_dir" "$BUILD_INFO" $(gpu_architectures) <<'EOF'
import hashlib, json, os, sys
wheel_dir, build_info_name, *gpu_architectures = sys.argv[1:]
build_info = json.load(open(os.path.join(wheel_dir, build_info_name)))
lock_sha256 = hashlib.sha256(open("uv.lock", "rb").read()).hexdigest()
if build_info["uv_lock_sha256"] != lock_sha256:
    sys.exit(f"kernels.sh: the wheels in {wheel_dir} were built for another uv.lock; rebuild them")
missing = sorted(set(gpu_architectures) - set(build_info["cuda_architectures"]))
if missing:
    sys.exit(f"kernels.sh: the wheels in {wheel_dir} have no kernels for compute capability {missing}; rebuild them")
for wheel, expected_sha256 in build_info["wheels"].items():
    if hashlib.sha256(open(os.path.join(wheel_dir, wheel), "rb").read()).hexdigest() != expected_sha256:
        sys.exit(f"kernels.sh: {wheel} does not match its sha256 in {build_info_name}")
EOF
  local wheels=()
  while read -r wheel; do
    wheels+=("$wheel_dir/$wheel")
  done < <(.venv/bin/python -c "import json, sys; print('\n'.join(json.load(open(sys.argv[1]))['wheels']))" \
             "$wheel_dir/$BUILD_INFO")
  # --no-config: the project's torch build setting for these packages applies to source builds only.
  uv pip install --no-config --no-deps --reinstall "${wheels[@]}"
  check_gpu_kernels
}

build_wheels() {
  local wheel_dir="$1"
  local nvcc="${CUDA_HOME:+$CUDA_HOME/bin/nvcc}"
  nvcc="${nvcc:-$(command -v nvcc || true)}"
  [ -x "$nvcc" ] || fail "nvcc not found; install the CUDA toolkit and put its bin/ on PATH (or set CUDA_HOME)"
  local cuda_version
  cuda_version="$("$nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
  local architectures
  architectures="$(gpu_architectures | tr '\n' ' ')"
  for architecture in $architectures; do
    if [ "$architecture" -ge 100 ] && [ "$(printf '%s\n' 12.8 "$cuda_version" | sort -V | head -1)" != 12.8 ]; then
      fail "nvcc $cuda_version cannot compile kernels for compute capability ${architecture:0:-1}.${architecture: -1}; use CUDA 12.8 or newer"
    fi
  done
  echo "==> building ${COMPILED_PACKAGES[*]} with nvcc $cuda_version for compute capability: $architectures"
  echo "    Source compilation can take tens of minutes. Each nvcc invocation below builds one kernel variant."
  sync_other_packages
  rm -rf "$wheel_dir" && mkdir -p "$wheel_dir"
  # Exactly the locked source archives (URL and sha256 from uv.lock), built against the locked torch.
  # --no-cache-dir: pip's wheel cache is keyed by the source URL, not the GPU targets.
  .venv/bin/python - "$wheel_dir/requirements.txt" "${COMPILED_PACKAGES[@]}" <<'EOF'
import re, sys
requirements_path, *packages = sys.argv[1:]
lock = open("uv.lock").read()
lines = []
for package in packages:
    match = re.search(r'\[\[package\]\]\nname = "%s"\n(?:.*\n)*?sdist = \{ url = "([^"]+)", hash = "sha256:([0-9a-f]+)"'
                      % re.escape(package), lock)
    if match is None:
        sys.exit(f"kernels.sh: no locked source archive of {package} in uv.lock")
    lines.append(f"{package} @ {match.group(1)} --hash=sha256:{match.group(2)}")
open(requirements_path, "w").write("\n".join(lines) + "\n")
EOF
  CUDA_HOME="$(dirname "$(dirname "$nvcc")")" PATH="$(dirname "$nvcc"):$PATH" \
    FLASH_ATTN_CUDA_ARCHS="$(echo $architectures | tr ' ' ';')" \
    uv run --no-sync --with pip==25.2 --with wheel==0.48.0 \
      python -m pip wheel --verbose --no-build-isolation --no-deps --require-hashes --no-cache-dir \
      --requirement "$wheel_dir/requirements.txt" --wheel-dir "$wheel_dir"
  rm "$wheel_dir/requirements.txt"
  .venv/bin/python - "$wheel_dir" "$BUILD_INFO" "$cuda_version" $architectures <<'EOF'
import hashlib, json, os, sys
import torch
wheel_dir, build_info_name, cuda_version, *architectures = sys.argv[1:]
wheels = sorted(name for name in os.listdir(wheel_dir) if name.endswith(".whl"))
build_info = {
    "uv_lock_sha256": hashlib.sha256(open("uv.lock", "rb").read()).hexdigest(),
    "torch": torch.__version__,
    "nvcc": cuda_version,
    "cuda_architectures": architectures,
    "wheels": {name: hashlib.sha256(open(os.path.join(wheel_dir, name), "rb").read()).hexdigest() for name in wheels},
}
json.dump(build_info, open(os.path.join(wheel_dir, build_info_name), "w"), indent=2)
print(json.dumps(build_info, indent=2))
EOF
  install_wheels "$wheel_dir"
}

case "${1:-}" in
  build) build_wheels "${2:-kernels}" ;;
  install)
    [ "$#" -eq 2 ] || fail "usage: scripts/kernels.sh install <dir>"
    install_wheels "$2" ;;
  *) fail "usage: scripts/kernels.sh build [dir] | install <dir>" ;;
esac
