# Sourccey training

This package keeps Sourccey training changes isolated from LeRobot's upstream
`lerobot_train.py` entry point.

## How the pieces fit together

`trainer.py` is the shared Sourccey training engine. The other files are small
entry-point wrappers around it:

```text
pretrain.py ───────────────────────┐
                                   ├──> trainer.py (`run_training`)
sft.py ──> configs/ + datasets/ ───┘
  ^
  └── sft_retry.py ──> retry.py
```

- `pretrain.py` supplies the normal LeRobot training configuration and dataset.
- `sft.py` supplies `SFTPipelineConfig` and the weighted SFT dataset factory.
- `trainer.py` owns the shared model, optimizer, dataloader, checkpoint, logging,
  evaluation, and training-loop behavior.
- `retry.py` only supervises a training subprocess. After a failure, it resumes
  that same training command from its last safe checkpoint.
- `sft_retry.py` is a deliberately small adapter that tells `retry.py` to launch
  `sft.py`, rather than `pretrain.py`.

Therefore, SFT and SFT retry already work together. There are not two SFT
implementations. Keep `sft_retry.py`: it provides the unambiguous
`lerobot-sft-retry` command while all retry behavior remains centralized in
`retry.py`.

## Configure an SFT run

Start with [`configs/sft_recipes/example.yaml`](configs/sft_recipes/example.yaml):

```yaml
policy:
  # Pretrained weights used to start this new SFT run.
  path: outputs/train/base/checkpoints/last/pretrained_model
  push_to_hub: false

dataset:
  # Keep normalization statistics from the pretrained checkpoint.
  normalization: checkpoint
  sources:
    - repo_id: your-org/original-demonstrations
      weight: 0.7
    - repo_id: your-org/corner-case-corrections
      weight: 0.3

# Scale the policy preset's original learning rate for fine-tuning.
policy_preset_lr_scale: 0.1
output_dir: outputs/sft/corner-case-v1
steps: 10000
batch_size: 16
save_freq: 1000
log_freq: 100
eval_freq: 0

wandb:
  enable: false
```

Source weights are sampling proportions, not dataset rewrite percentages. A
weight of `0.7` and `0.3` means approximately 70% of sampled training examples
come from the original dataset and 30% from the corrections dataset. The
weights only need to be positive; the sampler normalizes them.

Each source may also specify `root`, `revision`, or an `episodes` list. Set
`dataset.samples_per_epoch` when you want an explicit mixture epoch size.

When one local directory contains multiple separately recorded LeRobot
datasets, point `root` at the parent and set `subdataset_glob`:

```yaml
- repo_id: local/corner-case-corrections
  root: /path/to/correction-datasets
  subdataset_glob: "*"
  weight: 0.3
```

Every matching child with `meta/info.json` is loaded. The source's total
sampling share is divided among children in proportion to their frame counts,
so the result is uniform frame sampling across the collection. Source-fraction
logs remain aggregated under `local/corner-case-corrections`.

## Run SFT

For a normal run that exits if the process fails:

```bash
uv run lerobot-sft --config_path=src/lerobot/scripts/sourccey/train/configs/sft_recipes/example.yaml
```

CLI values can override the config when needed:

```bash
uv run lerobot-sft \
  --config_path=src/lerobot/scripts/sourccey/train/configs/sft_recipes/example.yaml \
  --steps=20000 \
  --output_dir=outputs/sft/corner-case-v2
```

This starts a new optimizer and scheduler from the pretrained weights in
`policy.path`. It does not resume the optimizer state from pretraining.

## Run SFT with automatic retry

Use this for long jobs where the process may be interrupted:

```bash
uv run lerobot-sft-retry \
  --max-attempts=10 \
  --retry-delay-seconds=20 \
  --config_path=src/lerobot/scripts/sourccey/train/configs/sft_recipes/example.yaml
```

The first attempt is the same SFT run described above. If it exits unsuccessfully,
the retry controller looks for:

```text
<output_dir>/checkpoints/last/pretrained_model/train_config.json
```

It then relaunches `sft.py` with `--resume=true` and that saved configuration,
which restores the SFT checkpoint and training state. It will stop immediately
if no safe resume checkpoint exists. `--max-retries=N` may be used instead of
`--max-attempts`; retries do not include the initial attempt.

## Resume SFT manually

To resume without the retry supervisor:

```bash
uv run lerobot-sft \
  --resume=true \
  --config_path=outputs/sft/corner-case-v1/checkpoints/last/pretrained_model/train_config.json
```

Do not pass the original `policy.path` when resuming. The saved training config
contains the checkpoint and optimizer state required for continuation.

## Pretraining

Run Sourccey pretraining with the standard LeRobot training arguments:

```bash
uv run sourccey-pretrain \
  --policy.type=act \
  --dataset.repo_id=your-org/pretraining-data \
  --output_dir=outputs/train/base
```

Pretraining and SFT share `trainer.py`; their wrappers determine which config
and dataset factory the engine receives.

## Maintaining the private trainer

The upstream LeRobot trainer can change independently. When upstream behavior
is intentionally adopted, sync it into `trainer.py` explicitly and rerun the
Sourccey training tests. SFT-specific behavior should normally remain in
`configs/`, `datasets/`, or `sft.py`, keeping the shared engine close to the
upstream implementation.
