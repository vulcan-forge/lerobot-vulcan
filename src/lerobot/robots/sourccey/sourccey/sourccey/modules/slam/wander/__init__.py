"""Sourccey wander/SLAM subsystem, split by concern out of the former 8,300-line
``ldlidar_wander_snapshot_stitch.py`` monolith so each piece can be read on its own.

Read order (each module depends only on the ones above it, never below):

  wander_types    shared dataclasses (Pose targets, drive-result records,
                  StopZoneConfig lives in stop_zone) + pure geometry math.
  imu_heading     the IMU gyro-yaw heading prior (advisory: breaks room symmetry
                  during pose recovery; never authors the map).
  lidar_feed      SelfMaskedLidarFeed — filters the robot's own arms/shell out of
                  every LiDAR frame at one choke point.
  stop_zone       the collision stop-box: is a scan point inside the "don't drive
                  here" wedge, and how many points trip the stop.
  frontier        wander PATH GENERATION — where is there unmapped space to go
                  toward, from the live scan and from the stitched occupancy grid.
  boxed_in        recovery: a full in-place rotation survey to escape when walled in.
  mapping         rotation-snapshot capture + map STITCHING (append a new scan to
                  the growing map, rebuild, write the debug HTML).
  localization    relocalize the robot against the stitched map + accept/reject a
                  candidate pose + log pose health.
  driving         the low-level MOTION primitives: one forward burst, a tracked
                  drive, an arc-tracked turn, the reverse escape.

The orchestration — the main loop, the "now look for an exit" state machine, and
the "I see an elevated object" edge-mapping — still lives in
``ldlidar_wander_snapshot_stitch.py`` and imports everything above.
See ``ARCHITECTURE.md`` in this folder for how it all connects.
"""
