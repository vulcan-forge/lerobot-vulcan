# Sourccey wander / SLAM — architecture map

This is the map for reading the autonomous mapping system. It answers "what file
is X in?" and "how do the pieces fit together?".

The system used to be one 8,300-line file. The **behavior** still lives in one
orchestrator, but every reusable **capability** has been pulled out into its own
small module under `scripts/sourccey_wander/` so you can read one concern at a
time. Nothing about what the robot does changed in the split — the function
bodies were moved verbatim.

---

## Where everything lives

### The orchestrator (still one file)
- **`scripts/ldlidar_wander_snapshot_stitch.py`** (~5,900 lines)
  The entry point and the main loop. It imports every module below and wires
  them together. Two big pieces of logic still live *inside* `main()` as nested
  functions (they lean on ~50 shared loop variables, so pulling them out safely
  is a separate job):
  - **the "now look for an exit" state machine** — EXIT RUN: room-centroid
    completion test, the no-going-back veto, the no-regression ratchet, and the
    `_exit_ratchet_guard` hard-stop.
  - **the "I see an elevated object" edge mapping** — `_stamp_confirmed_edges`,
    `_plan_edge_survey`, `_map_side_guard`, `_investigate_edge_hint`, etc.

### The `sourccey_wander/` package (one concern per file)
Listed bottom-up: each module depends only on the ones above it.

| File | Concern | Key names |
|---|---|---|
| `wander_types.py` | shared value types + pure geometry math | `FrontierChoice`, `ExploreTarget`, `DriveSequenceMeta`, `DriveBurstMeta`, `_normalize_angle_deg`, `_compose_motion_hints`, `_turn_lever_arm_local_delta` |
| `imu_heading.py` | IMU gyro-yaw heading prior (advisory only) | `ImuYawClient`, `_imu_anchor`, `_imu_resolved_theta`, `IMU_DEAD_RECK_SLACK_DEG` |
| `lidar_feed.py` | filter the robot's own arms out of every scan | `SelfMaskedLidarFeed` |
| `stop_zone.py` | the collision "stop box" | `StopZoneConfig`, `_point_in_stop_zone`, `_blocked_points_for_frame` |
| `frontier.py` | **wander path generation** — where to go next | `_select_frontier_choice` (live scan), `_select_map_frontier_choice` (grid BFS), `_build_world_occupancy`, `_select_survey_target` |
| `boxed_in.py` | recovery: full-rotation escape from a dead end | `_escape_boxed_in` |
| `mapping.py` | **snapshot capture + map stitching** | `_capture_snapshot`, `_append_stitch`, `_rebuild_stitch` |
| `localization.py` | **relocalize + judge the pose** | `_estimate_pose_against_stitched_map`, `_accept_relocalized_pose`, `_log_live_pose_state` |
| `driving.py` | **low-level motion primitives** | `_drive_with_tracking`, `_turn_with_arc_tracking`, `_drive_forward_burst`, `_reverse_escape`, `_run_drive_sequence` |

### The engine + I/O it all sits on (unchanged, pre-existing files)
- **`scripts/ldlidar_direct_snapshot_stitch.py`** — the scan-match / stitch
  engine: `Pose2D`, `MotionHint`, `Snapshot`, `_search_pose`, `_refine_pose`,
  grid builders, `_plan_frontier_path`. This is the math the whole system rests on.
- **`scripts/ldlidar_direct_snapshot_client.py`** — `DirectLidarFeed`, the raw
  LiDAR TCP feed, and `_scan_to_local_points`.
- **`scripts/ldlidar_auto_snapshot_stitch.py`** — Rerun viewer setup, `_send_stop`,
  turn-burst execution.
- **`scripts/ldlidar_defaults.py`** — default numeric constants.
- **`scripts/sourccey_elevated_safety.py`** — the camera+depth elevated-obstacle
  safety gate (the "hazard monitor" that stops the robot for table edges).
- **`scripts/sourccey_eye_panorama.py`**, **`sourccey_depth_perception.py`** —
  camera stitching + the depth model.

### Robot host side (runs on the Pi, separate process)
- **`src/lerobot/robots/sourccey/sourccey/sourccey/sourccey_host.py`** — publishes
  camera/state and the IMU yaw socket.
- **`.../config_sourccey.py`** — host config (IMU publisher settings, etc.).
- **`.../sourccey_client.py`** — `SourcceyClient`, the Windows-side handle used to
  command the base.

---

## How one loop iteration flows

```
                         ┌─────────────────────────────────────────┐
   LiDAR TCP  ──▶ lidar_feed ──▶ (self-mask) ──▶ one clean revolution │
                         └─────────────────────────────────────────┘
                                        │
      localization  ◀────── relocalize the live scan against the map
        (where am I?)                   │
                                        ▼
      frontier  ─────── pick a direction: live opening OR grid BFS target
   (where's unmapped space?)            │
                                        ▼
      exit-run logic (in main) ─── if room is mapped, override with the door;
                                    veto anything heading back inside
                                        │
                                        ▼
      driving  ─────── turn to face it, then drive toward it burst-by-burst,
   (make it happen)     re-localizing after each burst; stop box + elevated
                        gate + exit ratchet can halt any burst
                                        │
                                        ▼
      mapping  ─────── capture a fresh snapshot at the new spot and stitch it
   (grow the map)       into the accumulated map; rebuild the viewer overlay
                                        │
                                        └──▶ (loop)

   boxed_in / reverse_escape: recovery paths taken when the robot is wedged or
   the pose is lost.
   imu_heading: consulted only during pose recovery, to break room symmetry.
```

---

## Reading order suggestion

1. `wander_types.py` — learn the vocabulary (what a `FrontierChoice` is, etc.).
2. `frontier.py` — how the robot decides where to go.
3. `driving.py` — how a decision becomes wheel motion.
4. `mapping.py` + `localization.py` — the two halves of SLAM.
5. The `main()` loop in `ldlidar_wander_snapshot_stitch.py` — how it's all sequenced,
   plus the exit-run and elevated-edge logic that still lives there.
