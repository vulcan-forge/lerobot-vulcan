"""
Voice listener that streams raw audio to client for remote speech recognition.

This version runs on the robot and sends raw audio chunks over ZMQ to a client
that runs Vosk locally (with a large model). The client then sends recognized
text back to the host.

Run (on robot):
  uv run python -m lerobot.robots.sourccey.sourccey.sourccey.voice_listener_audio_stream --client-ip <CLIENT_IP>
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


class AudioStreamSender:
    def __init__(self, client_ip: str, port: int):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.CONFLATE, 1)  # Only keep latest audio chunk
        self._sock.connect(f"tcp://{client_ip}:{port}")

    def send_audio(self, audio_data: bytes) -> bool:
        try:
            self._sock.send(audio_data, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False  # Client not ready, drop audio

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
        "--client-ip",
        type=str,
        required=True,
        help="IP address of the client that will process the audio",
    )
    parser.add_argument(
        "--audio-port",
        type=int,
        default=SourcceyHostConfig().port_zmq_audio,
        help="ZMQ port for audio streaming (default from SourcceyHostConfig)",
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

    sender = AudioStreamSender(args.client_ip, args.audio_port)
    audio_q: "queue.Queue[bytes]" = queue.Queue()

    def audio_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[voice_listener] Audio status: {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    print(f"[voice_listener] Streaming audio to {args.client_ip}:{args.audio_port}")
    print("[voice_listener] Listening...")

    try:
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=args.channels,
            callback=audio_callback,
            device=args.device,
        ):
            while True:
                data = audio_q.get()

                # Calculate audio level (RMS) to filter out background noise
                audio_array = np.frombuffer(data, dtype=np.int16)
                rms_level = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

                # Debug output
                if args.debug_audio_levels:
                    status = "✓" if rms_level >= args.audio_threshold else "✗"
                    print(f"[audio] RMS: {rms_level:.1f} (threshold: {args.audio_threshold:.1f}) {status}", file=sys.stderr)

                # Only stream audio above threshold
                if rms_level >= args.audio_threshold:
                    sender.send_audio(data)

    except KeyboardInterrupt:
        print("[voice_listener] Stopping...")
        return 0
    finally:
        sender.close()


if __name__ == "__main__":
    raise SystemExit(main())

