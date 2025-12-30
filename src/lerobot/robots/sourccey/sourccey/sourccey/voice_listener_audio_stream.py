"""
Voice listener that streams raw audio to client for remote speech recognition.

This version runs on the robot and publishes raw audio chunks over ZMQ (PUB socket).
Clients can subscribe to receive the audio, process it with Vosk, and send recognized
text back to the host.

Run (on robot):
  uv run python -m lerobot.robots.sourccey.sourccey.sourccey.voice_listener_audio_stream

The robot will wait for a client to connect before streaming audio.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing voice dependencies. Install `sounddevice` in the robot environment.") from e

from .config_sourccey import SourcceyHostConfig


class AudioStreamPublisher:
    def __init__(self, port: int):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)  # Only keep latest audio chunk
        self._sock.bind(f"tcp://*:{port}")
        # Give subscribers time to connect
        time.sleep(0.5)

    def send_audio(self, audio_data: bytes) -> bool:
        try:
            self._sock.send(audio_data, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False  # Shouldn't happen with PUB, but handle it

    def close(self):
        try:
            self._sock.close()
        finally:
            self._ctx.term()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sourccey voice listener - streams audio to client for remote recognition"
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index (optional)")
    parser.add_argument(
        "--audio-port",
        type=int,
        default=SourcceyHostConfig().port_zmq_audio,
        help="ZMQ port for publishing audio (default from SourcceyHostConfig)",
    )
    parser.add_argument(
        "--audio-threshold",
        type=float,
        default=200.0,
        help="Minimum audio level (RMS) to stream. Lower = more sensitive, Higher = less background noise. Default: 200.0",
    )
    parser.add_argument(
        "--debug-audio-levels",
        action="store_true",
        help="Print audio levels for debugging",
    )
    args = parser.parse_args(argv)

    publisher = AudioStreamPublisher(args.audio_port)
    # Use a bounded queue to prevent memory issues from overflow
    audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=10)

    def audio_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            # Only print overflow warnings, ignore other status messages
            if status.input_overflow:
                print(f"[voice_listener] Audio status: input overflow (queue full, dropping frames)", file=sys.stderr)
        # Non-blocking put - drop frame if queue is full
        try:
            audio_q.put_nowait(bytes(indata))
        except queue.Full:
            pass  # Drop frame if queue is full

    print(f"[voice_listener] Publishing audio on port {args.audio_port}")
    print("[voice_listener] Waiting for client to connect...")
    print("[voice_listener] Listening...")

    try:
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=1600,  # Smaller blocksize (0.1s at 16kHz) for lower latency and less overflow
            dtype="int16",
            channels=args.channels,
            callback=audio_callback,
            device=args.device,
        ):
            while True:
                try:
                    # Get audio with timeout to allow checking for interrupts
                    data = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Calculate audio level (RMS) to filter out background noise
                audio_array = np.frombuffer(data, dtype=np.int16)
                rms_level = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

                # Debug output
                if args.debug_audio_levels:
                    status = "✓" if rms_level >= args.audio_threshold else "✗"
                    print(f"[audio] RMS: {rms_level:.1f} (threshold: {args.audio_threshold:.1f}) {status}", file=sys.stderr)

                # Only stream audio above threshold
                if rms_level >= args.audio_threshold:
                    publisher.send_audio(data)

    except KeyboardInterrupt:
        print("[voice_listener] Stopping...")
        return 0
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())

