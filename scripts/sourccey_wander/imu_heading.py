"""IMU gyro-yaw heading prior for the LiDAR SLAM loop.

This is the robot's cheap sense of "which way am I facing." The Pi host
integrates the vertical gyro axis and publishes a running yaw angle on its own
ZMQ socket; ``ImuYawClient`` subscribes to it on a background thread. The value
is noisy in magnitude and drifts without bound, so ONLY yaw DELTAS are ever
consumed, and it is purely ADVISORY — it never writes the map. Its one real job
is to break the room's rotational symmetry during pose recovery (a rectangular
room looks the same at 0/90/180/270deg to a scan-matcher; the gyro says which one
you're actually in). If the feed is dead or stale every accessor returns None.
Mapping can remain LiDAR-only, while heading-controlled base motion enters a
safe recovery hold until the stream returns.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

# is supplying yaw. Tight (the gyro is accurate over the few seconds of a turn),
# so the wide-theta recovery tiebreaker can reject a room-symmetric wrong mode
# that a lidar-only, drifted anchor would wave through.
IMU_DEAD_RECK_SLACK_DEG = 20.0


class ImuYawClient:
    """Subscribes to the host's integrated-yaw PUB socket and exposes the latest
    heading in degrees (sign-corrected to the SLAM CCW convention).

    Wholly optional and non-fatal: if zmq is unavailable, the socket never
    connects, or samples stop arriving, ``deg()`` returns ``None``. Mapping
    callers fall back to LiDAR-only behavior; heading-controlled motion waits
    for stream recovery instead of issuing a blind command. Only yaw DELTAS are
    consumed downstream, so unbounded gyro drift in the absolute value is fine.
    """

    def __init__(self, endpoint: str, *, sign: float = 1.0, stale_after_s: float = 4.0) -> None:
        """Store connection settings and init the shared state; does NOT connect.

        Records the ZMQ ``endpoint`` to subscribe to, the ``sign`` used to flip
        the gyro into the SLAM CCW convention, and how long a sample stays
        "fresh". Sets up the lock-protected latest-yaw slot, the background-thread
        handles, and the sign self-check counters. Connecting and the receive
        loop don't start until ``start()`` is called.
        """
        # stale_after_s is generous on purpose: the samples arrive on a background
        # thread, but the main loop holds the GIL through the heavy stitch/solve
        # compute (rebuilds up to ~1.5s), starving that thread. Crucially, that
        # compute runs while the robot is STATIONARY, so a yaw sample from just
        # before it is still valid at the next turn's start (heading unchanged).
        # A truly dead feed is still caught — no messages at all -> stale ->
        # clean lidar fallback.
        self._endpoint = str(endpoint)
        self._sign = float(sign)
        self._stale_after_s = float(stale_after_s)
        self._lock = threading.Lock()
        self._yaw_deg: float | None = None
        self._last_rx_monotonic: float | None = None
        # Host-time history lets LiDAR consumers ask for the yaw that belonged
        # to a completed revolution instead of pairing an old scan with the yaw
        # at some later point in a slow scan-match. Both publishers use the Pi's
        # wall clock, so their timestamps are directly comparable.
        self._history: deque[tuple[float, float]] = deque(maxlen=2048)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock = None
        # Sign self-check: compare IMU deltas against well-tracked lidar turns.
        self._sign_agree = 0
        self._sign_disagree = 0
        self._sign_warned = False

    def start(self) -> None:
        """Connect the subscriber socket and spin up the background receiver.

        Opens a ZMQ SUB socket with CONFLATE (keep only the newest message) and a
        200ms receive timeout, connects it to the endpoint, and launches the
        daemon ``_run`` thread that fills the yaw slot. If anything fails
        (zmq missing, connect error) it prints a notice, leaves the socket None,
        and the whole client stays inert — every reader then falls back to
        lidar-only, so a missing IMU never breaks the run.
        """
        try:
            import zmq

            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.CONFLATE, 1)
            sock.setsockopt(zmq.RCVTIMEO, 200)
            sock.connect(self._endpoint)
            self._sock = sock
        except Exception as exc:  # noqa: BLE001
            print(
                f"[wander] IMU yaw client disabled (connect failed: {exc}); "
                "heading is lidar-only"
            )
            self._sock = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="wander_imu_yaw")
        self._thread.start()
        print(
            f"[wander] IMU yaw heading prior ENABLED (endpoint={self._endpoint}, "
            f"sign={self._sign:+.0f}); waiting for first sample"
        )

    def _run(self) -> None:
        """Background loop: receive yaw messages and store the latest one.

        Blocks on the socket (200ms timeout). On each message it parses the JSON,
        multiplies the yaw by the configured sign, and stores it with a receive
        timestamp under the lock. If nothing arrives it counts empty polls and
        prints a THROTTLED warning (~5s, then ~35s, then every ~5min) so a dead
        feed is visible once without flooding the log — the IMU is advisory, so
        silence must not spam. Exits when ``stop()`` sets the stop event.
        """
        import zmq

        first = True
        ever_received = False
        # RCVTIMEO is 200ms, so 25 empty polls ~= 5s. Warn when nothing is
        # arriving, but THROTTLED (at ~5s, ~35s, then every ~5min): the IMU is
        # advisory-only (lidar owns mapping), so a dead feed degrades the
        # recovery veto but must not flood the log for the whole session
        # (field 2026-07-18: the every-5s version drowned an otherwise clean
        # 12-capture run). Almost always: the robot host is running old code
        # (no 'IMU yaw publisher bound' at startup) or its IMU failed to
        # connect ('IMU reporter disabled' warning) — the host console says.
        empty_polls = 0
        next_warn_at = 25
        while not self._stop.is_set():
            try:
                msg = self._sock.recv()
            except zmq.Again:
                if ever_received:
                    empty_polls += 1
                    if empty_polls >= next_warn_at:
                        next_warn_at = empty_polls + (
                            150 if next_warn_at == 25 else 1500
                        )
                        print(
                            f"[wander] WARNING: IMU yaw stream on {self._endpoint} "
                            f"has been silent for ~{empty_polls // 5}s; navigation "
                            "will hold safely until ZMQ reconnects."
                        )
                    continue
                if not ever_received:
                    empty_polls += 1
                    if empty_polls >= next_warn_at:
                        next_warn_at = empty_polls + (150 if next_warn_at == 25 else 1500)
                        print(
                            f"[wander] WARNING: no IMU yaw samples on {self._endpoint} after "
                            f"~{empty_polls // 5}s — the advisory heading check is inactive "
                            "(mapping is lidar-only regardless). Check the Pi robot host "
                            "console: it must print 'IMU yaw publisher bound' at startup; "
                            "'IMU reporter disabled' means the IMU wiring/connection failed."
                        )
                continue
            except Exception:  # noqa: BLE001
                continue
            try:
                data = json.loads(msg.decode("utf-8"))
                yaw_deg = float(data["yaw_deg"]) * self._sign
                host_ts_s = float(data["ts_ns"]) / 1e9 if data.get("ts_ns") is not None else None
            except Exception:  # noqa: BLE001
                continue
            if ever_received and empty_polls >= 25:
                print("[wander] IMU yaw stream restored after a transport outage")
            ever_received = True
            empty_polls = 0
            next_warn_at = 25
            with self._lock:
                self._yaw_deg = yaw_deg
                self._last_rx_monotonic = time.monotonic()
                if host_ts_s is not None:
                    if self._history and host_ts_s < self._history[-1][0]:
                        # Do not interpolate across a host restart/clock correction.
                        self._history.clear()
                    self._history.append((host_ts_s, yaw_deg))
            if first:
                first = False
                print("[wander] IMU yaw feed live (heading prior active)")

    def deg(self) -> float | None:
        """Latest sign-corrected yaw in degrees, or None if absent/stale."""
        with self._lock:
            if self._yaw_deg is None or self._last_rx_monotonic is None:
                return None
            if (time.monotonic() - self._last_rx_monotonic) > self._stale_after_s:
                return None
            return float(self._yaw_deg)

    def deg_at_wall_time(self, host_wall_ts_s: float, *, max_gap_s: float = 0.20) -> float | None:
        """Interpolate continuous yaw at a Pi-host wall-clock timestamp.

        This fuses with ``ScanFrame`` revolution timestamps. It refuses to
        bridge a large sample gap and returns ``None`` for older host streams
        without timestamps, so callers can safely fall back to :meth:`deg`.
        """
        target = float(host_wall_ts_s)
        with self._lock:
            samples = tuple(self._history)
        if not samples:
            return None
        gap = float(max_gap_s)
        if target <= samples[0][0]:
            return float(samples[0][1]) if samples[0][0] - target <= gap else None
        if target >= samples[-1][0]:
            return float(samples[-1][1]) if target - samples[-1][0] <= gap else None

        lo = 0
        hi = len(samples) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if samples[mid][0] <= target:
                lo = mid
            else:
                hi = mid
        t0, y0 = samples[lo]
        t1, y1 = samples[hi]
        if target - t0 > gap or t1 - target > gap or t1 <= t0:
            return None
        fraction = (target - t0) / (t1 - t0)
        # Yaw is continuous/unwrapped, so ordinary interpolation is correct.
        return float(y0 + fraction * (y1 - y0))

    def deg_fresh(self, *, wait_up_to_s: float = 0.4, max_age_s: float = 0.3) -> float | None:
        """Yaw from a sample received within ``max_age_s``, waiting up to
        ``wait_up_to_s`` for one. Use this to BRACKET a maneuver: a sample that
        is merely "not too stale" (``deg()``'s 4s window) but was captured
        mid-turn under-reports the rotation. Call it at a moment the robot is
        stationary (just before a turn / after it settles) — the short sleep
        yields the GIL so the background receiver catches up if heavy main-loop
        compute starved it. Returns None only if the feed delivers nothing fresh
        in time (genuinely dead)."""
        deadline = time.monotonic() + max(0.0, float(wait_up_to_s))
        while True:
            with self._lock:
                yaw = self._yaw_deg
                rx = self._last_rx_monotonic
            now = time.monotonic()
            if yaw is not None and rx is not None and (now - rx) <= float(max_age_s):
                return float(yaw)
            if now >= deadline:
                return None
            time.sleep(0.02)

    def sample_age_s(self) -> float | None:
        """Age of the newest received sample, or ``None`` before first contact."""
        with self._lock:
            received = self._last_rx_monotonic
        if received is None:
            return None
        return max(0.0, time.monotonic() - float(received))

    def receiver_running(self) -> bool:
        """Whether the background subscriber thread is still available."""
        thread = self._thread
        return bool(thread is not None and thread.is_alive() and not self._stop.is_set())

    def delta_since(self, yaw_before: float | None) -> float | None:
        """RAW yaw change since ``yaw_before`` (both sign-corrected), or None.

        Raw, never normalized: the published yaw integrates continuously and is
        never wrapped, so the plain difference is the exact rotation — including
        multi-revolution accumulations that ±180 normalization would corrupt."""
        if yaw_before is None:
            return None
        now = self.deg()
        if now is None:
            return None
        return float(now) - float(yaw_before)

    def note_tracked_turn(self, tracked_deg: float, imu_delta_deg: float | None) -> None:
        """Cross-check IMU sign against a confidently lidar-tracked turn."""
        if imu_delta_deg is None or abs(tracked_deg) < 8.0 or abs(imu_delta_deg) < 4.0:
            return
        if (tracked_deg >= 0.0) == (imu_delta_deg >= 0.0):
            self._sign_agree += 1
        else:
            self._sign_disagree += 1
        if (
            not self._sign_warned
            and self._sign_disagree >= 3
            and self._sign_disagree > self._sign_agree * 2
        ):
            self._sign_warned = True
            print(
                "[wander] WARNING: IMU yaw sign looks INVERTED versus the lidar-tracked turns "
                f"(agree={self._sign_agree}, disagree={self._sign_disagree}). Re-run with "
                "--imu-yaw-sign -1 (or flip the host imu_yaw_gyro_sign). Until then the IMU "
                "heading prior is IGNORED so it cannot fight the solver."
            )

    def sign_trustworthy(self) -> bool:
        """False once the self-check is confident the configured sign is wrong."""
        return not (self._sign_disagree >= 3 and self._sign_disagree > self._sign_agree * 2)

    def stop(self) -> None:
        """Cleanly shut the client down at end of run.

        Signals the background thread to exit, joins it (up to 1s), then closes
        the socket, swallowing any close error. Safe to call even if ``start()``
        never connected.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:  # noqa: BLE001
                pass
        self._sock = None


def _imu_anchor(imu: "ImuYawClient | None", theta_deg: float) -> tuple[float, float] | None:
    """Snapshot (gyro_yaw_now, dead_reck_theta) so the heading can later be
    re-derived from the gyro. Set at stationary relocalization-accept moments, so
    a fresh read is both safe and accurate. Returns None if the IMU is dead."""
    if imu is None:
        return None
    yaw = imu.deg_fresh()
    if yaw is None:
        return None
    return (float(yaw), float(theta_deg))


def _imu_resolved_theta(imu: "ImuYawClient | None", anchor: tuple[float, float] | None) -> float | None:
    """Dead-reckoned heading propagated from ``anchor`` by the gyro delta since
    it was set. Uses a FRESH read (the recovery tiebreaker fires right after a
    possibly-blind turn, when a stale sample would still show the pre-turn
    heading). None if the IMU is dead.

    The delta is RAW, never ±180-normalized: the published yaw is continuous, and
    the cumulative rotation across a long lost episode routinely exceeds 180deg —
    wrapping it silently corrupts the recovered heading. Pose theta in this
    module runs unbounded too; consumers normalize final comparisons."""
    if imu is None or anchor is None:
        return None
    now = imu.deg_fresh()
    if now is None:
        return None
    return float(anchor[1]) + (float(now) - float(anchor[0]))
