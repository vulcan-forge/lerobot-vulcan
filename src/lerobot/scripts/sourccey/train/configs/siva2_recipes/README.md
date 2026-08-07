# SIVA2 recipes

Convert the C-009 XVLA checkpoint:

```bash
uv run lerobot-convert-xvla-to-siva2 \
  --source=outputs/train/xvla_s_sourccey-shirt-fold-c-009/checkpoints/1000000/pretrained_model \
  --output-dir=outputs/converted/xvla_s_sourccey-shirt-fold-c-009-to-siva2
```

Train SIVA2 on C-010:

```bash
CUDA_VISIBLE_DEVICES=1 uv run lerobot-train-retry \
  --max-attempts=30 \
  --retry-delay-seconds=20 \
  --config_path=src/lerobot/scripts/sourccey/train/configs/siva2_recipes/siva2_shirt_fold_c_010.yaml
```

The example enables the optional frozen-Florence cache. Its first pass fills
the SQLite file; repeated samples skip Florence. Set
`policy.cache_florence_features: false` to disable it.
