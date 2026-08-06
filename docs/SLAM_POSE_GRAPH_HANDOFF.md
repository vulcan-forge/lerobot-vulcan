# Sourccey SLAM: Pose-Graph Overhaul — Handoff

**Date:** 2026-08-05 (late night). **Branch:** `SLAM-system`. **Status:** implemented and
offline-verified; **NOT yet field-tested**. Fallback point for everything below:
commit `0f66c0d9` ("Stabilize SLAM heading and pose acceptance; checkpoint before
Manhattan-world").

This document exists so the work can continue in another conversation/tool with
zero context loss.

---

## 1. How we got here (one day, compressed)

The explorer's long-standing disease: the map **curves** and the robot **meanders**.
A full day of field runs + log forensics established the causes in order:

1. **Gyro heading reference drift.** The gyro was the heading referee at the
   map-commit chokepoint. It drifts; walls don't. A ledger split (accepted 12.1°
   re-anchor without re-zeroing the gyro pair) made every later batch fight a
   stale ±10–13° offset: geometry committed at two headings → bowed walls,
   validation gates discarding everything → zero-progress meandering.
   *Fixed* (checkpoint commit): re-zero at every accepted re-anchor, 3-strike
   same-signed drift escape, wide-window drift acceptance in recovery.
2. **Bottom-camera odometry was accidentally OFF all day** (`--bottom-odometry off`
   was a stale flag carried through command reconstruction). With it on,
   stationary corrections dropped from 30–46 cm to 4–28 cm. It is the along-corridor
   translation sensor; lidar cannot observe that coordinate in a corridor.
   **Never run without it.**
3. **Partial acceptance smear.** Consecutive stationary batches twice proposed the
   *identical* over-cap offset (tight scatter, high score); the innovation cap
   discarded both; recoveries then accepted only *part* (30 of 46 cm); snapshots
   painted at half-corrected poses = doubled ghost walls.
   *Fixed* (checkpoint commit): persistent-offset 2-strike acceptance.
   (A scan-shifting "retro-correction" was tried the same night and **reverted** —
   with stops-only mapping every snapshot is individually validated at commit, so
   shifting past paint by the newest correction moves *good* walls. Do not re-add.)
4. **The structural admission:** all of the above is gatekeeping at an
   irreversible-write chokepoint. The professional answer is poses-as-variables
   with global optimization. The user green-lit the overhaul. That is what's now
   implemented.

## 2. What is now implemented (this working tree, uncommitted until now)

### 2.1 `scripts/sourccey_pose_graph.py` (new module, pure numpy)

- **`PoseGraph2D`** — dense Gauss-Newton SE(2) pose graph (Grisetti tutorial
  formulation, analytic Jacobians, angle wrapping, fixed-node gauge). ~100
  stationary keyframes per mission ⇒ dense 3N×3N solves are microseconds.
  Factors: `add_between` (odometry), `add_pose_prior` (absolute fix with
  **anisotropic 2×2 world-frame position information**), `add_heading_prior`.
- **Manhattan axis machinery:**
  - `dominant_axis_deg(points)` → building wall axis mod 90 + mass fraction.
    Uses **5-return chords**, not consecutive pairs: adjacent returns are ~3 cm
    apart with ~0.5 cm noise ⇒ single-step orientations are ±14° noise; 15–30 cm
    chords are ±2°. (Found by test, matters on real data.)
  - `axis_snap_delta_deg(points, axis, tolerance_deg=7)` → heading correction to
    align a scan's walls to the building axis, or None. 7° tolerance is
    deliberate: drift-sized errors only; genuinely diagonal geometry refused.
  - `wall_direction_masses(points, axis)` → wall-length mass parallel to each
    axis. A wall **parallel to axis1 constrains translation along axis0**
    (perpendicular-to-wall observability); this mapping is unit-tested.

### 2.2 `WorldMap` becomes scans-as-source-of-truth (`sourccey_explore.py`)

- `MapScan` gained `node_id` / `node_pose` (defaulted; the navigator's keyword
  construction is unaffected).
- `attach_recent_scans_to_node(node_id, node_pose, since_index)` — scans ride
  rigidly on their node. Scans committed *between* stops (corridor checkpoints)
  attach to the **next** node via `graph_state["scans_watermark"]`.
- `apply_node_poses({node: (x, y, θ)})` — retroactively moves attached scans by
  their node's optimized delta, **re-renders the occupancy grid from the scan
  list**, invalidates the GOLD reference cache when gold moved, bumps
  `pose_revision`. Skips sub-0.1 mm deltas (no pointless rebuilds).
- Cache correctness: `_match_support_mask` now keys on
  `(n_scans, pose_revision)` — it previously keyed on count alone, which would
  have been a stale-cache bug the moment poses could change retroactively.
  `analysis_cache` / `_support_cache` key on `grid.version`, which re-render bumps.
- Anchor-spin scans keep `node_id = -1`: they are the **fixed gauge** and never move.
- Planner marks (`mark_hits`) are deliberately not replayed on re-render — they
  were inserted under the transform that was just corrected.

### 2.3 Explorer integration (`sourccey_explore.py`, all near the
`imu_heading_ref` block and the stationary-batch accept branches)

- **Building axis locked once** from the anchor GOLD map after anchor validation
  (`[graph] building axis locked …`). Non-Manhattan room ⇒ compass stays silent,
  gyro machinery (all of it, still present and tested) remains the referee.
- **`_axis_snap_pose`** = the Manhattan compass. Runs FIRST in both heading
  chokepoints: `_imu_map_heading_align` (now `(pose, local_xy)`; `WorldMap.add`
  passes the scan) and `_gyro_clamped_consensus` (optional `local_xy`, callers
  pass the best inlier scan). Walls own heading whenever confident; the gyro
  ledger is re-synced to the snapped result.
- **`_register_stationary_fix(predicted, accepted, local, score, axis_locked,
  odometry_broken)`** — called at **every accepted absolute fix**: validated
  stationary transaction, stationary re-anchor (odometry_broken when
  correction > 25 cm), continuity recovery, opposite-view reacquisition,
  panorama re-anchor (recoveries: `odometry_broken=True` ⇒ wide between edge so a
  broken chain is not averaged into good history). It:
  1. adds an odometry between-factor from the previous node (σ scales with leg
     distance),
  2. adds the scan-match pose prior with **anisotropic position information**
     (corridor: lateral tight, along-corridor weak ⇒ the graph lets bottom-camera
     odometry own the along-corridor coordinate — the exact observability split
     whose absence caused the 30–46 cm oscillations),
  3. adds an axis heading prior when the compass fired (σ = 1°),
  4. attempts **loop closure** (below),
  5. attaches new scans, optimizes, applies node poses (map re-render), and sets
     `cur_pose` to the refined node pose.
  Log line: `[graph] node N registered (odo X.XXm): … largest retroactive map
  correction Xcm; grid re-rendered.`

### 2.4 Loop closure — odometry-gated (the user's guardrail, verbatim)

Design rule: **geometry proposes, odometry disposes.**

- Candidate: an old node (≥ 5 stops back — a revisit, not a continuation) that
  the *current optimized graph* already places within 0.9 m of the new node.
- Verification: the new stationary scan is matched against the candidate node's
  own attached scan points (`_localize_against_points`, 0.45 m search, ±4°).
  Requires match ≥ max(11, min_match_score) and support ≥ 55 %.
- **The guardrail:** the correction the closure implies must fit the accumulated
  drift budget: `15 cm + 3 % × distance driven between the two visits`
  (per-node cumulative odometry in `graph_state["node_odo"]`). A hallway that
  merely *looks* identical demands a correction far outside that budget and is
  refused with an explicit log line:
  `[graph] loop-closure candidate REFUSED by the odometry guardrail: … a
  similar-looking place, not the same place.`
- Accepted closures add a between-factor (σ = 6 cm / 1.5°); the next optimize
  distributes the correction over the whole intervening chain and the map
  re-renders. Closures are soft graph edges — nothing is irreversibly painted.

## 3. Verification status

Offline (all passing; scratchpad tests, re-runnable):
- 7 pose-graph unit tests: axis recovery on rotated corridors (0/17/44/89°),
  clutter ⇒ silent compass, snap accepts 4° / refuses 22°, anisotropy mapping,
  odometry chain + end fix ⇒ distributed correction, **curved corridor
  15° bend / 1.43 m bow ⇒ 0.7° / 0.01 m**, anisotropic prior ownership split.
- End-to-end through the real `WorldMap` + graph: 2°/stop heading drift over six
  stops (12° raw) ⇒ final pose **0.08° / 0 cm** from truth, painted walls
  straight to 0.9 cm (95th pct), `pose_revision` advancing.
- Gyro fallback paths re-verified with the compass stubbed silent.
- `py_compile` clean on explorer, navigator, new module.

**Not verified:** any of this on the physical robot. Loop closure in particular
is compile-clean and gate-reviewed but has never fired on real data.

## 4. How to run + what to watch

```
uv run python scripts/sourccey_explore.py --remote-ip 192.168.1.237
```

(Bottom odometry defaults ON — never pass `--bottom-odometry off`. `rerun-sdk`
is installed in the env.)

Startup must show BOTH:
- `[explore] bottom-camera ground odometry ENABLED; …`
- `[graph] building axis locked from the anchor map: …`

During the run, the new signal lines:
- `[graph] axis compass: heading snapped ±X.Xdeg …` — walls correcting heading.
- `[graph] node N registered … largest retroactive map correction Xcm; grid
  re-rendered.` — the graph fixing history.
- `[graph] loop closure: node N <-> node C …` / `… REFUSED by the odometry
  guardrail …`.

Healthy expectations: no `consensus heading clamped` wars, straight walls,
stationary corrections in the single-digit cm after the first few nodes.

## 5. Next steps (in order)

1. **Field run.** If the map is straight and progress is clean → commit becomes
   the new baseline. If not → the log + `git diff 0f66c0d9` isolates the overhaul.
2. Tune closure gates from real data (0.9 m candidate radius, 55 % support,
   3 %/m drift budget are first guesses).
3. Consider registering *frontier-transit* checkpoints as lightweight nodes
   (currently they ride on the next stationary node).
4. Only after closures prove reliable: allow multi-view verified closures
   against distinctive GOLD anchors at longer ranges.
5. Known simplification: `completed_transitions` (doorway ratchet coordinates)
   are not re-posed on re-render; corrections are expected small. Revisit if
   doorway logic misbehaves after large closures.

## 6. Hard rules learned this day (do not relearn them the hard way)

- The lidar owns mapping; the gyro is an advisory veto (directive #9). Walls
  beat both when visible.
- Never shift already-validated painted scans by a newer correction outside the
  graph (the reverted "retro-correction" mistake) — retroactive moves are only
  legitimate as *optimized node deltas*.
- One aliased batch must never move the pose; two independent agreeing batches
  are corroboration (persistent-offset acceptance / the closure guardrail are
  both instances of this principle).
- `--bottom-odometry off` must never appear in a run command again.
