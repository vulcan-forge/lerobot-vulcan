#!/usr/bin/env python3
"""
Compatibility alias for the voice + audio streaming entrypoint.

Use:
  python -m lerobot.robots.sourccey.sourccey.sourccey.voice_listener

This forwards to `voice_listener_audio_stream.py` (kept for backwards compatibility).
"""

from __future__ import annotations

from .voice_listener_audio_stream import main


if __name__ == "__main__":
    raise SystemExit(main())


