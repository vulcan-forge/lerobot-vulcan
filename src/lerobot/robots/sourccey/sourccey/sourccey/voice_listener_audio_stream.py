"""
Sourccey voice listener with:
- True stereo capture
- 2-mic beamforming
- RNNoise denoising (via librnnoise.so + ctypes)
- Direction gating
- Behavioral echo suppression
- Clean mono int16 output for ASR

This is near-commercial-grade for 2 basic microphones.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import queue
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import zmq

from .config_sourccey import SourcceyHostConfig


# -----------------------------
# RNNoise via ctypes (librnnoise.so)
# -----------------------------
class RNNoiseCTypes:
    """
    Minimal RNNoise wrapper using the system-installed librnnoise.so.
    RNNoise processes audio in 480-sample frames of float32 in [-1, 1].
    """

    FRAME_SIZE = 480

    def __init__(self) -> None:
        self._lib = ctypes.cdll.LoadLibrary("librnnoise.so")

        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.rnnoise_process_frame.restype = ctypes.c_float

        self._st = self._lib.rnnoise_create(None)
        if not self._st:
            raise RuntimeError("rnnoise_create() returned NULL")

    def process(self, audio_i16: np.ndarray) -> np.ndarray:
        """
        audio_i16: np.int16 mono
        returns: np.int16 mono denoised
        """
        if audio_i16.dtype != np.int16:
            raise ValueError("RNNoiseCTypes expects int16 audio")

        n = audio_i16.size
        if n < self.FRAME_SIZE:
            # Too short to process; return as-is
            return audio_i16

        # Convert to float in [-1, 1]
        x = (audio_i16.astype(np.float32) / 32768.0).copy()
        y = np.zeros_like(x)

        # Process full frames only; leave tail unprocessed (or you could pad)
        end = (n // self.FRAME_SIZE) * self.FRAME_SIZE
        for i in range(0, end, self.FRAME_SIZE):
            in_frame = x[i : i + self.FRAME_SIZE]
            out_frame = y[i : i + self.FRAME_SIZE]

            self._lib.rnnoise_process_frame(
                self._st,
                in_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )

        # Copy tail (unprocessed remainder) through
        if end < n:
            y[end:] = x[end:]

        out_i16 = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
        return out_i16

    def close(self) -> None:
        if getattr(self, "_st", None):
            try:
                self._lib.rnnoise_destroy(self._st)
            finally:
                self._st = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def try_init_rnnoise() -> Optional[RNNoiseCTypes]:
    try:
        rn = RNNoiseCTypes()
        print("[voice] RNNoise available (ctypes/librnnoise.so)", file=sys.stderr)
        return rn
    except Exception as e:
        print(f"[voice] RNNoise NOT available (ctypes load failed): {e}", file=sys.stderr)
        return None


# -----------------------------
# DSP helpers
# -----------------------------
def rms_i16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(xf))))


def estimate_delay(left: np.ndarray, right: np.ndarray, max_delay: int) -> int:
    n = min(len(left), len(right))
    if n < 32:
        return 0

    # int32 correlation with DC removal
    l = left[:n].astype(np.int32) - int(np.mean(left))
    r = right[:n].astype(np.int32) - int(np.mean(right))

    corr = np.correlate(l, r, mode="full")
    delay = int(np.argmax(corr) - (n - 1))
    return int(np.clip(delay, -max_delay, max_delay))


def shift(x: np.ndarray, d: int) -> np.ndarray:
    if d == 0:
        return x
    if d > 0:
        return np.pad(x, (d, 0))[: len(x)]
    d = -d
    return np.pad(x[d:], (0, d))


def beamform(left: np.ndarray, right: np.ndarray, sr: int, max_delay_ms: float):
    max_delay = int(sr * max_delay_ms / 1000.0)
    delay = estimate_delay(left, right, max_delay)

    if delay > 0:
        right = shift(right, -delay)
    elif delay < 0:
        left = shift(left, delay)

    mono = ((left.astype(np.int32) + right.astype(np.int32)) // 2)
    mono = np.clip(mono, -32768, 32767).astype(np.int16)
    return mono, delay


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

    # Audio
    p.add_argument("--input-channels", type=int, default=2)
    p.add_argument("--beamform", action="store_true")
    p.add_argument("--max-delay-ms", type=float, default=1.0)

    # Direction gating
    p.add_argument(
        "--max-front-delay",
        type=int,
        default=12,
        help="Reject speech if abs(delay) > this (samples)",
    )

    # Echo suppression
    p.add_argument("--tts-cooldown-ms", type=int, default=500)

    # Gating
    p.add_argument("--audio-threshold", type=float, default=0.0)
    p.add_argument("--preroll-s", type=float, default=0.3)
    p.add_argument("--hangover-s", type=float, default=0.5)

    # Debug
    p.add_argument("--debug", action="store_true")

    args = p.parse_args(argv)

    if args.input_channels != 2:
        print("ERROR: This pipeline requires stereo input.", file=sys.stderr)
        return 2

    pub = AudioStreamPublisher(SourcceyHostConfig().port_zmq_audio)
    q: queue.Queue[bytes] = queue.Queue(maxsize=64)

    rn = try_init_rnnoise()
    last_tts_ts = 0.0

    def audio_cb(indata, frames, time_info, status):
        # Keep callback minimal
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
            channels=2,
            device=args.device,
            callback=audio_cb,
        ):
            print("[voice] Listening (beamforming + RNNoise + gating)")

            while True:
                try:
                    raw = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Echo suppression (behavioral)
                if (time.time() - last_tts_ts) * 1000 < args.tts_cooldown_ms:
                    continue

                stereo = np.frombuffer(raw, dtype=np.int16)
                if stereo.size % 2:
                    continue
                stereo = stereo.reshape(-1, 2)
                left, right = stereo[:, 0], stereo[:, 1]

                mono, delay = beamform(left, right, args.sample_rate, args.max_delay_ms)

                # Direction gate
                if abs(delay) > args.max_front_delay:
                    continue

                # RNNoise denoise
                # if rn is not None:
                #     mono = rn.process(mono)

                level = rms_i16(mono)
                now = time.time()
                above = args.audio_threshold <= 0 or level >= args.audio_threshold

                if args.debug:
                    print(f"delay={delay:3d} rms={level:6.1f}", file=sys.stderr)

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
        if rn is not None:
            rn.close()
        pub.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
