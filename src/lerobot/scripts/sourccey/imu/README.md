# Sourccey IMU (LSM6DSOX + LIS3MDL)

## Quick Check

Print one sample:

```bash
uv run python src/lerobot/scripts/sourccey/imu/check_imu.py --once --pretty
```

Continuous monitoring (default every 10 seconds):

```bash
uv run python src/lerobot/scripts/sourccey/imu/check_imu.py
```

Custom interval:

```bash
uv run python src/lerobot/scripts/sourccey/imu/check_imu.py --interval-s 2
```

One-line sanity check (single read):

```bash
uv run python src/lerobot/scripts/sourccey/imu/one_line_imu_check.py
```

## Host Logging

`sourccey_host.py` now starts an IMU reporter and logs one IMU line every 10 seconds by default.
Configure in `SourcceyHostConfig`:

- `imu_print_enabled`
- `imu_print_interval_s`
- `imu_bus_num`
- `imu_lsm6dsox_address`
- `imu_lis3mdl_address`

