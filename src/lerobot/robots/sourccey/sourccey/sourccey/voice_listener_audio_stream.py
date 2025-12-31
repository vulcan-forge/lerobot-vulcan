"""
Sourccey voice listener (FINAL BASELINE)

- Stereo capture (USB device)
- Single-mic selection (LEFT channel)
- High-pass filter (removes motor / chassis noise)
- Software gain normalization (no hardware AGC available)
- Clean mono int16 stream for ASR
- Preroll + hangover gating
"""

from __future__ import annotations

import argparse
import collections
import queue
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import zmq

from .config_sourccey import SourcceyHostConfig


# -----------------------------
# DSP helpers
# -----------------------------
def rms_i16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


def highpass_i16(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """
    Simple 1st-order high-pass filter.
    Critical for robots (removes motor rumble).
    """
    if x.size == 0:
        return x

    y = np.empty_like(x)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)

    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y


def normalize_i16(x: np.ndarray, target_rms: float = 200.0) -> np.ndarray:
    """
    Software gain staging.
    Prevents 'always loud' microphones from breaking VAD/ASR.
    """
    cur_rms = rms_i16(x) + 1e-6
    gain = target_rms / cur_rms
    gain = float(np.clip(gain, 0.1, 3.0))  # do NOT over-amplify noise
    y = x.astype(np.float32) * gain
    return np.clip(y, -32768, 32767).astype(np.int16)


# -----------------------------
# ZMQ publisher
# -----------------------------
class AudioStreamPublisher:
    def __init__(self, port: int):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        time.sleep(0.3)

    def send(self, data: bytes):
        try:
            self.sock.send(data, zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self):
        self.sock.close()
        self.ctx.term()


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--blocksize", type=int, default=3200)

    # Gating
    p.add_argument("--audio-threshold", type=float, default=0.0)
    p.add_argument("--preroll-s", type=float, default=0.3)
    p.add_argument("--hangover-s", type=float, default=0.5)

    # Debug
    p.add_argument("--debug", action="store_true")

    args = p.parse_args(argv)

    pub = AudioStreamPublisher(SourcceyHostConfig().port_zmq_audio)
    q: queue.Queue[bytes] = queue.Queue(maxsize=64)

    def audio_cb(indata, frames, time_info, status):
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    block_s = args.blocksize / args.sample_rate
    preroll_n = max(1, int(args.preroll_s / block_s))
    preroll = collections.deque(maxlen=preroll_n)

    streaming = False
    last_voice_ts = 0.0

    try:
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=args.blocksize,
            dtype="int16",
            channels=2,  # stereo device
            device=args.device,
            callback=audio_cb,
        ):
            print("[voice] Listening (HPF + normalize, mono output)")

            while True:
                try:
                    raw = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                stereo = np.frombuffer(raw, dtype=np.int16)
                if stereo.size % 2:
                    continue
                stereo = stereo.reshape(-1, 2)

                # ---- CHANNEL SELECTION ----
                # Use LEFT mic only (most stable)
                mono = stereo[:, 0]

                # ---- DSP PIPELINE ----
                mono = highpass_i16(mono, cutoff_hz=120.0, sr=args.sample_rate)
                mono = normalize_i16(mono, target_rms=200.0)

                level = rms_i16(mono)
                now = time.time()
                above = args.audio_threshold <= 0 or level >= args.audio_threshold

                if args.debug:
                    print(f"rms={level:6.1f}", file=sys.stderr)

                mono_bytes = mono.tobytes()
                preroll.append(mono_bytes)

                if above:
                    last_voice_ts = now
                    if not streaming:
                        streaming = True
                        for b in preroll:
                            pub.send(b)
                    pub.send(mono_bytes)
                else:
                    if streaming and (now - last_voice_ts) <= args.hangover_s:
                        pub.send(mono_bytes)
                    else:
                        streaming = False

    except KeyboardInterrupt:
        print("[voice] Stopping...")
        return 0
    finally:
        pub.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
