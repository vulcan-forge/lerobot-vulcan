#!/usr/bin/env python3
"""
Send text to the Sourccey host text/TTS gateway (so the robot speaks what you type).

Requires the host-side `voice_listener_audio_stream.py` (or another host gateway) to be running
with the text gateway enabled (PULL on port 5557 by default).
"""

from __future__ import annotations

import argparse
import sys

import zmq


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robot_ip", required=True, help="IP of the Sourccey host machine.")
    p.add_argument("--text-in-port", type=int, default=5557, help="Host text input port (client -> host).")
    p.add_argument("--text", default="", help="One-shot text to speak.")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Read lines from stdin and speak each line (type 'exit' or Ctrl+C to quit).",
    )
    args = p.parse_args(argv)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{args.robot_ip}:{int(args.text_in_port)}")

    def send(msg: str) -> None:
        s = (msg or "").strip()
        if not s:
            return
        try:
            sock.send_string(s, flags=zmq.NOBLOCK)
        except Exception:
            # best effort
            pass

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
        else:
            send(args.text)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


