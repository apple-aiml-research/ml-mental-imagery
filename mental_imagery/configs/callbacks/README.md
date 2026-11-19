# Custom run logging

The default callback writes `metrics.jsonl`. W&B, TensorBoard, or another logger can be added without
changing the training loop by subclassing [`Callback`](../../utils/callbacks.py):

- `on_run_start(config, run_dir)`: initialize the logger from the resolved config.
- `on_metrics(record, step)`: send the metrics dictionary to the logger.
- `on_checkpoint_saved(path, step, kind)`: optionally upload checkpoints.
- `on_run_end()`: flush and close the logger.

Callbacks run only on rank 0. `on_train_start(iteration)` is also available for resume handling.

Register the implementation with Hydra in a separate config directory, for example
`my_configs/callbacks/wandb.yaml`:

```yaml
# @package _global_
callbacks:
  - _target_: my_package.logging.WandbCallback
    project: mental-imagery
```

Add `my_configs` to `hydra.searchpath` and select it with `callbacks=wandb`. The target class is
instantiated by Hydra and must be importable from the training environment. Multiple callbacks can be
listed in the same config, for example the default `MetricsFile` alongside an external logger.
