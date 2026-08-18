# Sourccey Battery Setup (bq34z100)

This folder contains the battery setup and diagnostics tools for Sourccey.

If you only do one thing, follow **Quick Start** below in order.

## Files in this folder

- `configure_bq34z100.py` - write/read bq34z100 Data Flash fields over I2C
- `battery.py` - minimal frontend battery telemetry JSON
- `check_bq34z100.py` - deep diagnostics + watch mode for learning cycle checks
- `golden/flash_bq34z100.py` - TI FlashStream runner for `.df.fs` / `.bq.fs` files

## Prerequisites

- Run on the machine that has direct I2C access to the gauge (typically robot host/RPi).
- Gauge is reachable at I2C address `0x55` (default).
- Use `uv run ...` for all commands.

## I2C install and sanity checks (Linux/Raspberry Pi)

Install basic I2C tooling:

```bash
sudo apt update
sudo apt install -y i2c-tools
```

Enable I2C on Raspberry Pi (if not already enabled):

```bash
sudo raspi-config nonint do_i2c 0
sudo reboot
```

After reboot, verify I2C device nodes exist:

```bash
ls -l /dev/i2c-*
```

Check your user has I2C permission:

```bash
groups
```

If `i2c` is missing from your groups:

```bash
sudo usermod -aG i2c $USER
```

Then log out/log back in (or reboot) and verify again with `groups`.

Scan for devices (bus 1 is typical on Raspberry Pi):

```bash
sudo i2cdetect -y 1
```

Expected: the gauge appears at `0x55`.

Verify Python I2C dependency (`smbus2`) in this project env:

```bash
uv run python -c "import smbus2; print('smbus2 OK')"
```

## New Robot Quick Start (Standard Battery)

Use this workflow for a new or recovered Sourccey robot with all of the following:

- GoldenMate 12.8 V 10 Ah LiFePO4 battery
- 4 cells in series (4S)
- TI bq34z100 gauge (`device_type=0x0100`)
- Gauge firmware `0x0201`

The golden image contains the matching firmware, LiFePO4 chemistry, configuration,
and learned battery parameters. Do not use it for a different battery, gauge, or
firmware version.

### 1) Prepare the robot

Before writing the gauge:

1. Stop the Sourccey desktop app and every other process polling battery telemetry.
2. Stop the motors and other heavy loads.
3. Keep the battery connected and robot power stable throughout the flash.
4. Do not interrupt the flash or power-cycle the robot while it is running.

### 2) Confirm communication and compatibility

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py info
```

For the bundled golden image, confirm the output contains:

```json
{
  "device_type": "0x0100",
  "fw_version": "0x0201"
}
```

The initial chemistry ID can differ on a new chip. Do not continue if the device
type or firmware version differs.

### 3) Flash the golden image

Optional parse-only preview (does not write the gauge):

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py flash-golden --profile bq --dry-run
```

Program the full firmware and data-flash image:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py flash-golden --profile bq
```

A successful run ends with `Flashstream complete` and a flash summary. All strict
comparison checks must pass. If the command reports an error, preserve the complete
output and do not run another write command until the I2C/power issue is resolved.

### 4) Validate the flashed gauge

```bash
uv run python src/lerobot/scripts/sourccey/battery/check_bq34z100.py --pretty
```

Confirm these key values:

- `chip_info.chem_id` is `0x4203`
- `chip_info.device_type` is `0x0100`
- `chip_info.fw_version` is `0x0201`
- `pack.series_cells` is `4`
- `pack.voltage_divider` is `4023`
- `pack.voltsel_enabled` is `true`
- `pack.update_status` is `0x06`
- `control_flags.QEN` is `true`
- `learning.charge_voltage_target_mv` is `14300`
- `learning.taper_current_ma` is `250`
- `telemetry.max_error` is low (the bundled learned image normally reports about `1`)

Do not run `setup-4s-lifepo4` after a successful golden-image flash. That command
is for manual configuration and can overwrite learned golden-image parameters.

### 5) Synchronize state of charge

Immediately after flashing, `state_of_charge` and `remaining_capacity_ah` can be
stale (including `0`) even when voltage is sensible. Use a proper 4S LiFePO4
charger, charge toward 14.3 V until current tapers, and then allow the battery to
rest with minimal load so the gauge can synchronize with the attached pack.

Monitor the adjustment:

```bash
uv run python src/lerobot/scripts/sourccey/battery/check_bq34z100.py --watch --interval-s 5
```

## Manual Configuration (Custom or Unflashed Gauge)

Use this section only when the bundled golden image is not appropriate or when
you intentionally need to change individual profile values.

### 1) Apply standard Sourccey fields

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py setup-4s-lifepo4
```

This applies the project defaults (4S LiFePO4 profile, divider/config fields, thresholds, IT enable flow).

### 2) Verify key written fields

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py read-field --field voltage_divider
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py read-field --field flash_update_ok_cell_volt_mv
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py read-field --field number_of_series_cells
```

### 3) Show all stats (pretty JSON, useful for debugging)

```bash
uv run python src/lerobot/scripts/sourccey/battery/check_bq34z100.py --pretty
```

```bash
uv run python src/lerobot/scripts/sourccey/battery/check_bq34z100.py --full --pretty
```

Use this output as your primary debugging snapshot when values look wrong.

### 4) Verify runtime telemetry

```bash
uv run python src/lerobot/scripts/sourccey/battery/battery.py
```

Expected: JSON with `voltage`, `current_a`, `remaining_capacity_ah`, `max_capacity_ah`, `state_of_charge`, `max_error`.

### 5) Run deeper health snapshot (recommended)

```bash
uv run python src/lerobot/scripts/sourccey/battery/check_bq34z100.py --pretty
```

Optional watch mode:

```bash
uv run python src/lerobot/scripts/sourccey/battery/check_bq34z100.py --watch --interval-s 5
```

## Golden Image Command Reference

Full firmware + data flash image:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py flash-golden --profile bq
```

Data-flash-only image:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py flash-golden --profile df
```

Preview without writes:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py flash-golden --profile bq --dry-run
```

## Common commands

List editable built-in fields:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py list-fields
```

Read one field:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py read-field --field <field_name>
```

Write one field:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py write-field --field <field_name> --value <int_value>
```

Set divider from resistor values and enable external divider mode:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py set-divider --top-ohm 249000 --bottom-ohm 16500 --enable-voltsel
```

Use non-default bus/address if needed:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py --bus 1 --address 0x55 info
```

Flash from a specific file path:

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py flash-golden --fs-file src/lerobot/scripts/sourccey/battery/golden/0100_2_01-bq34z100.bq.fs
```

## Troubleshooting write failures

If you see errors like:

- `A read of data written failed comparison` (bqStudio), or
- `RuntimeError: Block verify failed for subclass=...`

then communication is working, but Data Flash commit was rejected by gauge state.

### Typical symptoms

- `configure_bq34z100.py info` works
- field writes are planned, but readback returns old values
- pack voltage in telemetry is clearly wrong (for example, ~2.2V for a 4S pack)

### What to check first

```bash
uv run python src/lerobot/scripts/sourccey/battery/configure_bq34z100.py info
uv run python src/lerobot/scripts/sourccey/battery/battery.py
uv run python src/lerobot/scripts/sourccey/battery/check_bq34z100.py --pretty
```

If voltage path is wrong, writes can be blocked by flash update safety behavior.

### Recovery sequence (recommended)

1. Stop any other process talking to the gauge.
2. Confirm stable power, wiring, I2C address, device type, and firmware version.
3. For the standard supported hardware, re-run the full golden-image command from
   **New Robot Quick Start step 3**.
4. Run the validation command from **New Robot Quick Start step 4**.
5. Confirm the chemistry/configuration values match and telemetry voltage is sane.

## Notes

- `configure_bq34z100.py` auto-unseals on write commands by default using standard keys.
- `--dry-run` shows planned writes without modifying flash.
- `--no-verify` skips readback verify (use only for debugging).
