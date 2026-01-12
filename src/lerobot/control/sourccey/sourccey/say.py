#!/usr/bin/env python3
"""
Send text to the Sourccey host text/TTS gateway (so the robot speaks what you type).

Requires the host-side `voice_listener.py` / `voice_listener_audio_stream.py` (or another host gateway) to be running
with the text gateway enabled (PULL on port 5557 by default).
"""

from __future__ import annotations

import argparse
import sys
import time
import os

import zmq


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    default_robot_ip = os.environ.get("SOURCCEY_ROBOT_IP", "192.168.1.213")
    p.add_argument(
        "--robot_ip",
        default=default_robot_ip,
        help="IP of the Sourccey host machine (default: 192.168.1.213). Can also be set via SOURCCEY_ROBOT_IP.",
    )
    p.add_argument("--text-in-port", type=int, default=5557, help="Host text input port (client -> host).")
    p.add_argument("--text-out-port", type=int, default=5558, help="Host text output port (host -> client).")
    p.add_argument("--text", default="", help="One-shot text to speak.")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Read lines from stdin and speak each line (type 'exit' or Ctrl+C to quit).",
    )
    p.add_argument(
        "--listen",
        action="store_true",
        help="Listen for host responses/events on text-out-port and print them.",
    )
    p.add_argument(
        "--listen-seconds",
        type=float,
        default=2.0,
        help="How long to listen after sending (non-interactive mode).",
    )
    args = p.parse_args(argv)

    ctx = zmq.Context()
    sock_out = ctx.socket(zmq.PUSH)
    sock_out.setsockopt(zmq.CONFLATE, 1)
    sock_out.connect(f"tcp://{args.robot_ip}:{int(args.text_in_port)}")

    sock_in = None
    if args.listen:
        sock_in = ctx.socket(zmq.PULL)
        sock_in.setsockopt(zmq.CONFLATE, 1)
        sock_in.connect(f"tcp://{args.robot_ip}:{int(args.text_out_port)}")

    def send(msg: str) -> None:
        s = (msg or "").strip()
        if not s:
            return
        try:
            sock_out.send_string(s, flags=zmq.NOBLOCK)
        except Exception:
            # best effort
            pass

    def _poll() -> None:
        if sock_in is None:
            return
        try:
            m = sock_in.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return
        except Exception:
            return
        print(m)

    try:
        if args.interactive:
            print("Type a line to speak. Type 'exit' to quit.", file=sys.stderr)
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line.lower() in {"exit", "quit"}:
                    break
                send(line)
                # show immediate ACK/events if listening
                t0 = time.time()
                while args.listen and (time.time() - t0) < 0.5:
                    _poll()
                    time.sleep(0.01)
        else:
            send(args.text)
            if args.listen:
                t0 = time.time()
                while (time.time() - t0) < float(args.listen_seconds):
                    _poll()
                    time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock_out.close()
            if sock_in is not None:
                sock_in.close()
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


