# Train a Sourccey policy

Training does not connect to the robot, so `sourccey-host` is not required.
Run training on the controller or another computer that can access the
dataset.

This first run trains an ACT policy for 20,000 steps:

```bash
uv run lerobot-train --dataset.repo_id=vulcan-studio/sourccey-demo-1 --policy.type=act --policy.device=cuda --output_dir=outputs/train/act-sourccey-demo-1 --job_name=act-sourccey-demo-1 --batch_size=8 --steps=20000 --save_freq=5000 --wandb.enable=false --policy.push_to_hub=false
```

Use `cuda` for a supported NVIDIA GPU, `mps` for Apple silicon, or `cpu` for a
CPU-only run. Lower `--batch_size` if the device runs out of memory.

## Check before continuing

Training should produce checkpoints beneath:

```text
outputs/train/act-sourccey-demo-1/checkpoints/
```

Continue only after the `last/pretrained_model` directory exists and training
completed without a dataset feature or video-decoding error.

For larger datasets, cleanup, and video audits, see the package's
[dataset guide](https://github.com/vulcan-forge/lerobot-robot-sourccey/blob/main/docs/ai/datasets.md).

Next: [Roll out](06-rollout.md).
