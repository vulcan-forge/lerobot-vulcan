# Sourccey IMU (LSM6DSOX + LIS3MDL)

## I2C Bus Check (Run First)

If you get `Failed to find LSM6DSOX - check your wiring!`, verify I2C first:

```bash
sudo apt update
sudo apt install -y i2c-tools
ls -l /dev/i2c-*
```

Scan bus 1:

```bash
sudo i2cdetect -y 1
```

Expected IMU addresses:

- LSM6DSOX: `0x6A` (sometimes `0x6B`)
- LIS3MDL: `0x1C` (sometimes `0x1E`)

If your addresses are different, pass them explicitly:

```bash
uv run python src/lerobot/scripts/sourccey/imu/one_line_imu_check.py --lsm6dsox-address 0x6B --lis3mdl-address 0x1E
```

If probing still fails, read WHO_AM_I directly:

```bash
sudo i2cget -y 1 0x6a 0x0f
sudo i2cget -y 1 0x1c 0x0f
```

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

