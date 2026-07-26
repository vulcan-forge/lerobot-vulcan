# Train SIVA2 on Sourccey C-010

## 1. Convert the C-009 XVLA checkpoint

```bash
uv run lerobot-convert-xvla-to-siva2 \
  --source=outputs/train/xvla_s_sourccey-shirt-fold-c-009/checkpoints/1000000/pretrained_model \
  --output-dir=outputs/converted/xvla_s_sourccey-shirt-fold-c-009-to-siva2
```

## 2. Train from the YAML recipe

```bash
CUDA_VISIBLE_DEVICES=1 uv run lerobot-train-retry \
  --max-attempts=30 \
  --retry-delay-seconds=20 \
  --config_path=src/lerobot/scripts/sourccey/train/configs/siva2_recipes/siva2_shirt_fold_c_010.yaml
```

The recipe trains on `Combination/sourccey-shirt-fold-c-010` for one million
steps and saves every 50,000 steps. Edit the
[`YAML recipe`](../../scripts/sourccey/train/configs/siva2_recipes/siva2_shirt_fold_c_010.yaml)
to change training parameters. C-010 works as demonstration data without extra
SIVA2 metadata.

## Optional Florence cache

The recipe enables caching with:

```yaml
policy:
  freeze_vlm: true
  cache_florence_features: true
  florence_cache_path: outputs/cache/siva2_shirt_fold_c_010/florence_features.sqlite

dataset:
  image_transforms:
    enable: false
```

The converted checkpoint already sets `freeze_vlm: true`. The first visit to a
sample fills the cache; repeated visits skip frozen Florence computation. Use
fast local storage and a new path after changing the dataset, VLM, tokenizer,
cameras, dtype, or preprocessing. Set `cache_florence_features: false` to opt out.

Retry automatically resumes from the last safe checkpoint. To resume manually:

```bash
CUDA_VISIBLE_DEVICES=1 uv run lerobot-train \
  --resume=true \
  --config_path=outputs/train/siva2_sourccey-shirt-fold-c-010/checkpoints/last/pretrained_model/train_config.json
```
