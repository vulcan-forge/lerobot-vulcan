# Set up Sourccey

Complete the controller setup first, then the robot setup. Python 3.12 or 3.13
and [`uv`](https://docs.astral.sh/uv/getting-started/installation/) are required
on both computers.

## 1. Set up the controller computer

Open a terminal at the root of the `lerobot-vulcan` checkout.

Install the pinned Sourccey desktop profile and the training dependencies:

```bash
uv sync --locked --extra sourccey-desktop --extra training
```

Check that the pinned plugin imports successfully:

```bash
uv run python -c "import importlib.metadata as m; print('Sourccey plugin', m.version('lerobot-robot-sourccey'), 'is installed')"
```

The command should print the installed Sourccey plugin version without a
traceback.

Find the serial port for each leader arm. Run the command once with only the
left arm connected, then once with only the right arm connected:

```bash
uv run lerobot-find-port
```

Record both results. Windows ports look like `COM5`; stable Linux aliases are
normally `/dev/robotLeftArm` and `/dev/robotRightArm`.

Calibrate both leader arms after replacing the example ports:

```bash
uv run sourccey-teleop-calibrate --left-arm-port=COM5 --right-arm-port=COM6
```

The controller is ready when calibration completes without a serial-port
error.

### Developing the plugin locally

The normal path above uses the plugin version pinned in `uv.lock`. If you are
also editing a sibling `lerobot-robot-sourccey` checkout, install it as
editable instead:

```bash
uv pip install -e "../packages/lerobot-robot-sourccey[desktop,dev]"
```

Use `uv run --no-sync` in later commands while that editable install is active;
otherwise `uv` may restore the version pinned by this repository.

## 2. Set up the robot computer

These commands run on Sourccey's Linux computer from its
`lerobot-vulcan` checkout.

Install the pinned robot profile:

```bash
uv sync --locked --extra sourccey-robot
```

Preview udev and battery checks without changing the computer:

```bash
uv run sourccey-setup robot --dry-run
```

Apply the robot setup. This can request `sudo` to install packaged udev rules;
battery writes are never performed unless explicitly requested:

```bash
uv run sourccey-setup robot
```

Apply the packaged arm ranges and save the current lift sensor limits:

```bash
uv run sourccey-calibrate
```

Only use a full reset when physical arm and lift limits must be redetected.
This command moves hardware, so clear the entire workspace first:

```bash
uv run sourccey-calibrate --full-reset --yes
```

Find the robot computer's IP address:

```bash
hostname -I
```

Record the address on the same network as the controller.

## 3. Start the robot host

On the robot computer, start the host:

```bash
uv run sourccey-host
```

Leave this terminal running. The host is ready when it connects to the robot
hardware and prints `Waiting for commands...`.

Do not use `sourccey-host --help` as an installation check. The host currently
starts connecting to hardware even when `--help` is supplied.

Next: [Teleoperate](02-teleoperate.md).
