#!/usr/bin/env python3
"""Standalone probe for the host's integrated-yaw PUB socket (port 8770).

Isolates "is the robot host publishing yaw at all" from the whole wander loop.
Run it on the Pi against localhost to test the host in isolation, or from the
client PC against the Pi's IP to also test the network path:

    # On the Pi (host-only test):
    python scripts/imu_yaw_probe.py --host 127.0.0.1

    # From the Windows client (host + network test):
    python scripts/imu_yaw_probe.py --host 192.168.1.237

Prints each yaw sample as it arrives. If it prints nothing, the robot host
(sourccey_host.py) is NOT publishing — it was not restarted with the new code,
or its IMU failed to connect (check its console for 'IMU yaw publisher bound'
vs 'IMU reporter disabled: failed to connect IMU').
"""

from __future__ import annotations

import argparse
import json
import time

import zmq


def main() -> int:
    p = argparse.ArgumentParser(description="Probe the integrated-yaw PUB socket")
    p.add_argument("--host", default="127.0.0.1", help="Host serving the yaw PUB socket")
    p.add_argument("--port", type=int, default=8770, help="Yaw PUB port (host imu_yaw_pub_endpoint)")
    p.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds (0 = run forever)")
    args = p.parse_args()

    endpoint = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 250)
    sock.connect(endpoint)
    print(f"[probe] subscribed to {endpoint}; waiting for yaw samples (Ctrl+C to stop)...")

    start = time.monotonic()
    received = 0
    last_warn = start
    try:
        while True:
            if args.seconds > 0 and (time.monotonic() - start) >= args.seconds:
                break
            try:
                msg = sock.recv()
            except zmq.Again:
                now = time.monotonic()
                if received == 0 and (now - last_warn) >= 3.0:
                    last_warn = now
                    print(
                        f"[probe] still NOTHING on {endpoint} after "
                        f"{now - start:.0f}s — host is not publishing yaw"
                    )
                continue
            received += 1
            try:
                data = json.loads(msg.decode("utf-8"))
                print(
                    f"[probe] seq={data.get('seq')} yaw_deg={float(data['yaw_deg']):+.2f} "
                    f"rate_rad_s={float(data.get('rate_rad_s', 0.0)):+.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[probe] received a message but could not parse it: {exc!r} raw={msg[:120]!r}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)
    print(f"[probe] done; received {received} samples total")
    return 0 if received > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
