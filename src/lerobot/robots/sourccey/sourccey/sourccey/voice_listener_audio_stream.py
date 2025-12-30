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
import collections
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
        default=0.0,
        help="Minimum audio level (RMS) to stream. 0 = stream everything (recommended for best accuracy).",
    )
    parser.add_argument(
        "--preroll-s",
        type=float,
        default=0.3,
        help="Seconds of audio to buffer and send before speech starts (prevents clipped words).",
    )
    parser.add_argument(
        "--hangover-s",
        type=float,
        default=0.5,
        help="Seconds to keep streaming after audio falls below threshold (prevents chopped endings).",
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
        blocksize = 1600  # 0.1s at 16kHz
        bytes_per_frame = 2  # int16
        bytes_per_chunk = blocksize * args.channels * bytes_per_frame
        preroll_chunks = max(0, int(round(args.preroll_s / 0.1)))
        preroll_buf: "collections.deque[bytes]" = collections.deque(maxlen=max(1, preroll_chunks))

        stream_all = args.audio_threshold <= 0.0
        streaming_active = False
        last_above_ts = 0.0

        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=blocksize,  # Smaller blocksize (0.1s at 16kHz) for lower latency and less overflow
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
                    status = "✓" if (stream_all or rms_level >= args.audio_threshold) else "✗"
                    print(
                        f"[audio] RMS: {rms_level:.1f} (threshold: {args.audio_threshold:.1f}) {status}",
                        file=sys.stderr,
                    )

                # Normalize chunk size (defensive; should already be fixed-size)
                if len(data) != bytes_per_chunk:
                    # If device gives variable chunks, don't buffer/preroll; just forward best-effort
                    if stream_all or rms_level >= args.audio_threshold:
                        publisher.send_audio(data)
                    continue

                # Always keep a small preroll buffer (so we don't clip word starts)
                preroll_buf.append(data)

                now = time.time()
                above = stream_all or (rms_level >= args.audio_threshold)
                if above:
                    last_above_ts = now
                    if not streaming_active:
                        streaming_active = True
                        # Send buffered audio first
                        for chunk in list(preroll_buf):
                            publisher.send_audio(chunk)
                    publisher.send_audio(data)
                else:
                    # If we were streaming, keep going for hangover_s then stop
                    if streaming_active and (now - last_above_ts) <= max(0.0, args.hangover_s):
                        publisher.send_audio(data)
                    elif streaming_active:
                        streaming_active = False

    except KeyboardInterrupt:
        print("[voice_listener] Stopping...")
        return 0
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())

