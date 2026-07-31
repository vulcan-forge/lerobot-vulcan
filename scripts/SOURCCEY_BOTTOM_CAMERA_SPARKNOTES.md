# Sourccey Bottom Camera — SparkNotes

The bottom camera has **two jobs**:

1. Estimate how far the robot translated across the floor.
2. Detect low obstacles that the 2-D LiDAR beam may pass over.

It does **not** decide the robot's global location, control yaw, or command the
motors.

## The authority chain

Think of localization as three people checking one another:

1. **IMU yaw:** “Which direction is the robot facing?”
2. **Bottom camera:** “The floor appears to have moved, so the robot probably
   translated this far.”
3. **LiDAR scan matching:** “Here is the corrected final pose that fits the
   map.”

The camera measurement is only a short-distance prediction that helps LiDAR
start its search near the right answer. If camera flow is weak or nonsensical,
it returns nothing and LiDAR + IMU continue without it.

Motor commands are not odometry. The wheels have no encoders, so requested
velocity is never accepted as evidence that the robot physically moved.

Before any of this is enabled, `wait_for_live_bottom_camera()` requires several
fresh frame numbers, real image texture, and changing image content. One cached
or black image is not treated as a working camera. With
`--bottom-odometry required`, failure of that check stops startup and reports
the exact reason; optional mode falls back to LiDAR + IMU.

## Localization flow

```text
new bottom frame + previous bottom frame
                  |
                  v
find recognizable floor corners/texture
                  |
                  v
track those points forward and backward
                  |
                  v
reject tracks that do not agree
                  |
                  v
project good pixels onto the floor in metres
                  |
                  v
remove the motion caused by IMU-measured rotation
                  |
                  v
produce a short robot-translation estimate
                  |
                  v
accumulate it until navigation asks for it
                  |
                  v
use it as the seed for the next LiDAR match
                  |
                  v
LiDAR corrects and owns the final pose
```

## Low-obstacle safety flow

```text
bottom-camera frame
        |
        v
find strong image edges below the camera horizon
        |
        v
use the lowest edge as the object's floor contact
        |
        v
convert image row into distance in metres
        |
        v
check whether the object silhouette rises above the horizon
        |
        +--> flat/short: report it, but do not hard-stop
        |
        +--> tall + close + inside robot corridor
                         |
                         v
              require several confirming frames
                         |
                         v
                 activate ground safety stop
```

Several clean frames are likewise required to release an active stop. This
prevents one noisy image from stopping the robot and prevents flicker from
prematurely releasing it.

## Function cheat sheet

### `wait_for_live_bottom_camera()`

The startup proof-of-life check. It rejects a missing stream, stale frame IDs,
an unchanged cached image, a black/blank placeholder, and an image with too
little texture for floor tracking. It must pass before the program prints that
bottom-camera odometry is enabled.

### `default_bottom_camera_model()`

Returns the one shared description of camera height, forward offset, pitch, and
field of view. Every bottom-camera calculation uses this model so different
parts of the program cannot quietly disagree about the camera mounting.

### `floor_points_robot_frame(...)`

Takes image pixels and asks: “If this pixel is looking at the floor, where is
that floor point in metres relative to robot centre?” Pixels at/above the
horizon or at implausible distances are rejected.

### `estimate_ground_translation_from_tracks(...)`

Takes already-matched feature points from two frames. It uses IMU yaw to remove
apparent motion caused by rotation, then finds the translation on which most
features agree. Moving objects and bad tracks are rejected using median
consensus.

### `estimate_ground_flow(...)`

The full per-frame optical-flow operation. It finds floor features, tracks them
into the next frame, checks them backward, rejects bad tracks, and calls the
metric translation function above.

### `BottomCameraGroundOdometry`

Runs optical flow in a background thread so camera processing cannot block base
control or safety. It accumulates accepted movement until navigation calls
`take_delta()`.

- `start()` starts the worker.
- `stop()` shuts it down cleanly.
- `take_delta()` returns accumulated movement exactly once, then clears it.
- `discard()` clears old movement after a pivot or relocalization.
- `healthy` means at least one frame pair has passed every quality check.
- `totals` reports accepted and rejected frame-pair counts.

### `detect_bottom_floor_obstacles(...)`

Looks for obstacle silhouettes beneath the LiDAR scan height. It estimates the
floor-contact distance and whether the object is tall enough to matter. It is
stateless: it describes one frame but does not itself stop the robot.

### `BottomFloorDetector`

A tiny wrapper that remembers the shared camera model and safety settings. The
larger safety monitor calls `detect(frame)` on it.

### `bottom_obstacle_explains_floor_point(...)`

Helps the eye-camera system classify objects. If an eye edge projects onto the
same floor location as a bottom-camera obstacle, the edge is probably a low
floor object rather than a floating tabletop.

### `BottomGroundSafetyGate`

Owns the bottom camera's stop/release hysteresis. It checks whether a tall
candidate is close enough and inside the robot's forward corridor, then counts
consecutive hit or clear frames.

### `render_bottom_overlay(...)`

Draws what the system believes onto a copy of the camera frame. It is diagnostic
only and cannot change a safety decision.

## What gets rejected

Ground odometry contributes nothing when:

- The frame is stale or repeated.
- IMU yaw is not available.
- Too few floor features are visible.
- Forward/backward optical flow disagrees.
- Feature translations do not form a consistent group.
- The inferred movement is physically too large for the elapsed time.

This is intentional. A missing camera estimate is safer than a confident fake
estimate because LiDAR and IMU remain available.

## What resets accumulated camera movement

Navigation discards pending ground movement after:

- A deliberate pivot.
- Stationary relocalization.
- Global pose replacement.

Those operations already establish a newer pose, so applying older camera
movement afterward would count it twice.

## The most important rule

```text
IMU owns yaw.
Bottom camera predicts short translation.
LiDAR owns the corrected map pose.
Collision safety owns permission to move.
```

The bottom camera assists those systems; it never overrules them.
