#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run callbacks: the extension point for logging and syncing runs.

``callbacks`` in the ``train`` and ``train_fsq`` configs is a list of Hydra
``_target_`` specs (the ``callbacks`` config group; ``default`` writes
``metrics.jsonl``). They run on rank 0 only and see, in order:

- ``on_run_start(config, run_dir)``: the run directory exists;
- ``on_train_start(iteration)``: training (re)starts at micro-batch
  ``iteration`` (0 for a new run);
- ``on_metrics(record, step)``: a logged record (it holds its ``iteration``)
  at optimizer step ``step``;
- ``on_checkpoint_saved(path, step, kind)``: a complete checkpoint: an
  ``"eval"`` or ``"resume"`` directory, or an ``"artifact"`` file;
- ``on_run_end()``: the run finished or stopped after SIGTERM.
"""

from __future__ import annotations

import json
import os

from hydra.utils import instantiate


class Callback:
    """No-op hooks; subclasses override the ones they need."""

    def on_run_start(self, config, run_dir: str) -> None:
        pass

    def on_train_start(self, iteration: int) -> None:
        pass

    def on_metrics(self, record: dict, step: int) -> None:
        pass

    def on_checkpoint_saved(self, path: str, step: int, kind: str) -> None:
        pass

    def on_run_end(self) -> None:
        pass


class Callbacks(Callback):
    """Calls every hook on each callback in order."""

    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks

    def on_run_start(self, config, run_dir: str) -> None:
        for callback in self.callbacks:
            callback.on_run_start(config, run_dir)

    def on_train_start(self, iteration: int) -> None:
        for callback in self.callbacks:
            callback.on_train_start(iteration)

    def on_metrics(self, record: dict, step: int) -> None:
        for callback in self.callbacks:
            callback.on_metrics(record, step)

    def on_checkpoint_saved(self, path: str, step: int, kind: str) -> None:
        for callback in self.callbacks:
            callback.on_checkpoint_saved(path, step, kind)

    def on_run_end(self) -> None:
        for callback in self.callbacks:
            callback.on_run_end()


def build_callbacks(callback_configs, rank: int) -> Callbacks:
    """The configured callbacks on rank 0; no callbacks on the other ranks."""
    if rank != 0:
        return Callbacks([])
    callbacks = [instantiate(callback_config) for callback_config in callback_configs]
    for callback in callbacks:
        if not isinstance(callback, Callback):
            raise TypeError(
                f"{type(callback).__name__} is not a mental_imagery.utils.callbacks.Callback"
            )
    return Callbacks(callbacks)


class MetricsFile(Callback):
    """Every logged record as one JSON line in ``<run_dir>/metrics.jsonl``.

    On resume, records from the resume point on are dropped, since the run
    logs them again.
    """

    def __init__(self, filename: str = "metrics.jsonl"):
        self.filename = filename
        self.path = None

    def on_run_start(self, config, run_dir: str) -> None:
        self.path = os.path.join(run_dir, self.filename)

    def on_train_start(self, iteration: int) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            kept = [line for line in f if json.loads(line)["iteration"] < iteration]
        with open(self.path, "w") as f:
            f.writelines(kept)

    def on_metrics(self, record: dict, step: int) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"optimizer_step": step, **record}) + "\n")
