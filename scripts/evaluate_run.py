#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Evaluate every saved checkpoint of a training run that has no results yet.

    uv run python scripts/evaluate_run.py outputs/qwen3_5_0_8b/action_supervision
    uv run python scripts/evaluate_run.py outputs/qwen3_5_0_8b/action_supervision vllm.gpus=[0,1]

Checkpoints (``<run_dir>/checkpoints/step_XXXXXX``, complete ones only) are
evaluated one after another, oldest first, each with
``python -m mental_imagery.eval checkpoint=<checkpoint> <overrides>``; a
checkpoint whose ``results.json`` already exists is skipped. Extra arguments are
passed to every evaluation as Hydra overrides (``output_dir`` cannot be set: each
checkpoint keeps its results in ``<checkpoint>/eval/<games>/``). Prints the
average solve rate of every evaluated checkpoint at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

COMPLETE = "COMPLETE"  # marker of a finished checkpoint (mental_imagery.utils.checkpoint.COMPLETE)
DEFAULT_GAMES = "benchmark"  # games config of mental_imagery/configs/eval.yaml


def complete_checkpoints(run_dir: str) -> list[str]:
    checkpoint_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(checkpoint_root):
        raise SystemExit(f"{run_dir} has no checkpoints/ directory")
    names = sorted(
        name
        for name in os.listdir(checkpoint_root)
        if name.startswith("step_")
        and os.path.exists(os.path.join(checkpoint_root, name, COMPLETE))
    )
    if not names:
        raise SystemExit(f"{checkpoint_root} has no complete checkpoint")
    return [os.path.join(checkpoint_root, name) for name in names]


def games_choice(overrides: list[str]) -> str:
    for override in overrides:
        if override.split("=", 1)[0].lstrip("+") == "output_dir":
            raise SystemExit(
                "output_dir cannot be overridden: results are kept next to each checkpoint"
            )
    choices = [override.split("=", 1)[1] for override in overrides if override.startswith("games=")]
    return choices[-1] if choices else DEFAULT_GAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir", help="training run directory, e.g. outputs/qwen3_5_0_8b/action_supervision"
    )
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides passed to every mental_imagery.eval run"
    )
    args = parser.parse_args()

    games = games_choice(args.overrides)
    summary = []
    for checkpoint in complete_checkpoints(args.run_dir):
        results_path = os.path.join(checkpoint, "eval", games, "results.json")
        if os.path.exists(results_path):
            print(f"[evaluate_run] {checkpoint}: already evaluated", flush=True)
        else:
            print(f"[evaluate_run] {checkpoint}: evaluating", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mental_imagery.eval",
                    f"checkpoint={checkpoint}",
                    *args.overrides,
                ],
                check=True,
            )
        with open(results_path) as f:
            results = json.load(f)
        summary.append((results["step"], results["avg_solve_rate"]))

    print(f"\n{args.run_dir} ({games}):")
    for step, average in summary:
        print(f"  step {step:>7d}  average solve rate {average:.3f}")


if __name__ == "__main__":
    main()
