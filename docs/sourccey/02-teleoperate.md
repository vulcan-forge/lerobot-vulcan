# Teleoperate Sourccey

Teleoperation is the end-to-end readiness check. Before starting:

- `sourccey-host` is running on the robot computer.
- The robot workspace is clear.
- Both leader arms are connected and calibrated.
- The example IP address and ports below have been replaced.

On the controller computer, run:

```bash
uv run lerobot-teleoperate --robot.type=sourccey_client --robot.id=sourccey --robot.remote_ip=192.168.1.243 --teleop.type=bi_sourccey_leader --teleop.id=sourccey_leader --teleop.left_arm_port=COM5 --teleop.right_arm_port=COM6 --teleop_keyboard.type=keyboard --teleop_keyboard.id=sourccey_keyboard --fps=30 --display_data=true
```

If you installed the Sourccey plugin as an editable checkout, change the
beginning to `uv run --no-sync lerobot-teleoperate`.

## Keyboard controls

| Keys      | Action                           |
| --------- | -------------------------------- |
| `W` / `S` | Move forward / backward          |
| `A` / `D` | Move left / right                |
| `Z` / `X` | Rotate left / right              |
| `Q` / `E` | Raise / lower the lift target    |
| `R` / `F` | Increase / decrease base speed   |
| `N` / `M` | Toggle left / right arm untorque |

## Check before continuing

Confirm all of the following:

- Each leader arm moves only its matching robot arm.
- Base and lift keys move in the expected direction.
- Camera images update in the display.
- Releasing the controls stops motion.
- The host terminal does not show repeated timeouts or device errors.

Stop teleoperation with `Ctrl+C`. Fix any failed check before collecting data.

Next: [Record](03-record.md).
