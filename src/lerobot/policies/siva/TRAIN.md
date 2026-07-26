# Train SIVA on Sourccey C-010

## 1. Convert the C-009 XVLA checkpoint

```bash
uv run lerobot-convert-xvla-to-siva \
  --source=outputs/train/xvla_s_sourccey-shirt-fold-c-009/checkpoints/1000000/pretrained_model \
  --output-dir=outputs/converted/xvla_s_sourccey-shirt-fold-c-009-to-siva
```

## 2. Train from the YAML recipe

```bash
CUDA_VISIBLE_DEVICES=1 uv run lerobot-train-retry \
  --max-attempts=30 \
  --retry-delay-seconds=20 \
  --config_path=src/lerobot/scripts/sourccey/train/configs/siva_recipes/siva_shirt_fold_c_010.yaml
```

The recipe trains on `Combination/sourccey-shirt-fold-c-010` for one million
steps and saves every 50,000 steps. Edit the
[`YAML recipe`](../../scripts/sourccey/train/configs/siva_recipes/siva_shirt_fold_c_010.yaml)
to change training parameters.

Retry automatically resumes from the last safe checkpoint. To resume manually:

```bash
CUDA_VISIBLE_DEVICES=1 uv run lerobot-train \
  --resume=true \
  --config_path=outputs/train/siva_s_sourccey-shirt-fold-c-010/checkpoints/last/pretrained_model/train_config.json
```
