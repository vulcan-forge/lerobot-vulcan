# Sourccey training

This package keeps Sourccey training changes isolated from LeRobot's upstream
`lerobot_train.py` entry point.

- `trainer.py` is the Sourccey-owned training engine copied from LeRobot.
- `pretrain.py` runs the engine with the standard LeRobot dataset configuration.
- `sft.py` runs the same engine with a weighted SFT dataset mixture.
- `sft_retry.py` retries SFT from its last safe checkpoint.
- `configs/` and `datasets/` contain SFT-only configuration and sampling code.

Run pretraining with `sourccey-pretrain`. Run SFT with `lerobot-sft`, or use
`lerobot-sft-retry` for automatic recovery.

The upstream trainer can change independently. When upstream training behavior
is intentionally adopted, sync those changes into `trainer.py` explicitly and
re-run the Sourccey training tests.
