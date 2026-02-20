# Sourccey Gripper Force Control (Current-Sensing)

This note documents the force-aware gripper behavior added to `sourccey_follower`.

## What Changed

### 1) New config parameters
File: `config_sourccey_follower.py`

- `gripper_force_control_enabled: bool = True`
- `gripper_contact_current_threshold: int = 150`
- `gripper_grip_current_threshold: int = 250`
- `gripper_closing_deadband: float = 1.0`

These parameters enable and tune current-based contact detection and grip maintenance.

### 2) New runtime state
File: `sourccey_follower.py`

- `_gripper_contact_detected`
- `_gripper_hold_position`

This stores whether contact has been detected and which position should be held.

### 3) `send_action()` force-aware gripper handling
File: `sourccey_follower.py`

When force control is enabled and a gripper action is present:

1. Determine closing/opening direction from commanded vs present position.
2. If opening:
   - clear contact state and hold position.
3. If closing:
   - read `Present_Current` for `gripper`.
   - if current exceeds `gripper_contact_current_threshold`, mark contact and hold.
   - if current drops below `gripper_grip_current_threshold`, nudge slightly tighter.
4. Replace commanded gripper goal with held goal after contact.

All non-gripper joints continue to use normal position control.

## Why This Solves the Problem

Before this change, gripping was position-only, so the gripper would just go to a target pose.
Now it can detect contact and stop/hold based on motor current, which is a practical proxy for squeeze force.

## How To Start

1. Ensure dependencies and run env are correct:
   - `uv pip install -e ".[sourccey,smolvla,feetech]"`
   - run control scripts with `uv run ...`
2. Keep default thresholds for first pass:
   - contact = 150 mA
   - grip = 250 mA
3. Start teleop/record as usual and test with soft + rigid objects.

## Test Plan

### A) Baseline behavior
1. Command open/close without object.
2. Confirm full close still works and reopening resets state.

### B) Contact detection
1. Place an object in gripper path.
2. Close gripper.
3. Verify gripper stops on contact (instead of forcing to final closed pose).

### C) Grip maintenance
1. With object in gripper, apply gentle disturbance.
2. Verify gripper performs small tightening adjustments when current falls below grip threshold.

### D) Safety and false positives
1. Test empty close at different arm poses and speeds.
2. If it "detects contact" too early, increase `gripper_contact_current_threshold`.
3. If grip is too weak, increase `gripper_grip_current_threshold`.

## Tuning Guidance

- Increase `gripper_contact_current_threshold`:
  - if contact triggers too early / no object present.
- Decrease `gripper_contact_current_threshold`:
  - if it pushes too hard before detecting contact.
- Increase `gripper_grip_current_threshold`:
  - if objects slip.
- Decrease `gripper_grip_current_threshold`:
  - if squeeze is too aggressive.
- Increase `gripper_closing_deadband`:
  - if direction toggles/noise causes unstable state transitions.

## Known Limitations

- Force estimate is indirect (current-based), not a dedicated tactile sensor.
- Thresholds likely need per-hardware tuning.
- Current spikes/noise can vary by object material and battery/power conditions.

