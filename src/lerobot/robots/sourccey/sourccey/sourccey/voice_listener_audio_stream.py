"""
Sourccey voice listener (FINAL BASELINE v2)

- Mono capture (PortAudio/ALSA device only allows mono here)
- Stateful high-pass filter (removes motor / chassis noise)
- Smooth software AGC (no hardware AGC available)
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


class HighPass1:
    """
    1st-order high-pass filter with persistent state across blocks.
    This is *critical* — resetting each block makes the signal "blocky" and hurts VAD/ASR.
    """

    def __init__(self, cutoff_hz: float, sr: int):
        self.cutoff_hz = float(cutoff_hz)
        self.sr = int(sr)

        rc = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        dt = 1.0 / float(self.sr)
        self.alpha = float(rc / (rc + dt))

        self.x_prev = 0.0
        self.y_prev = 0.0

    def process_i16(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        # y[n] = a*(y[n-1] + x[n] - x[n-1])
        xf = x.astype(np.float32)
        y = np.empty_like(xf)

        a = self.alpha
        x_prev = self.x_prev
        y_prev = self.y_prev

        for i in range(xf.size):
            xn = float(xf[i])
            yn = a * (y_prev + xn - x_prev)
            y[i] = yn
            x_prev = xn
            y_prev = yn

        self.x_prev = x_prev
        self.y_prev = y_prev

        return np.clip(y, -32768, 32767).astype(np.int16)


class SmoothAGC:
    """
    Gentle automatic gain control with smoothing.
    Avoids per-block pumping (which breaks VAD and causes Whisper hallucinations).
    """

    def __init__(
        self,
        target_rms: float = 220.0,
        min_gain: float = 0.2,
        max_gain: float = 3.0,
        attack_s: float = 0.05,   # faster when signal too quiet
        release_s: float = 0.25,  # slower when signal too loud
        sr: int = 16000,
        blocksize: int = 3200,
    ):
        self.target_rms = float(target_rms)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)

        self.sr = int(sr)
        self.blocksize = int(blocksize)
        self.block_s = self.blocksize / float(self.sr)

        # Convert time constants to smoothing coefficients per block
        # gain += (desired - gain) * k
        self.k_attack = 1.0 - float(np.exp(-self.block_s / max(1e-6, self.attack_s)))
        self.k_release = 1.0 - float(np.exp(-self.block_s / max(1e-6, self.release_s)))

        self.gain = 1.0

    def process_i16(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x

        cur = rms_i16(x) + 1e-6
        desired = self.target_rms / cur
        desired = float(np.clip(desired, self.min_gain, self.max_gain))

        # If we need MORE gain (signal too quiet) -> attack (faster)
        # If we need LESS gain (signal too loud) -> release (slower)
        k = self.k_attack if desired > self.gain else self.k_release
        self.gain = float(self.gain + (desired - self.gain) * k)

        y = x.astype(np.float32) * self.gain
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

    # DSP
    p.add_argument("--hpf-hz", type=float, default=120.0)
    p.add_argument("--agc-target-rms", type=float, default=220.0)
    p.add_argument("--agc-min-gain", type=float, default=0.2)
    p.add_argument("--agc-max-gain", type=float, default=3.0)

    # Gating
    p.add_argument("--audio-threshold", type=float, default=0.0)
    p.add_argument("--preroll-s", type=float, default=0.25)
    p.add_argument("--hangover-s", type=float, default=0.5)

    # Debug
    p.add_argument("--debug", action="store_true")

    args = p.parse_args(argv)

    pub = AudioStreamPublisher(SourcceyHostConfig().port_zmq_audio)
    q: queue.Queue[bytes] = queue.Queue(maxsize=64)

    hpf = HighPass1(args.hpf_hz, args.sample_rate)
    agc = SmoothAGC(
        target_rms=args.agc_target_rms,
        min_gain=args.agc_min_gain,
        max_gain=args.agc_max_gain,
        sr=args.sample_rate,
        blocksize=args.blocksize,
    )

    def audio_cb(indata, frames, time_info, status):
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    block_s = args.blocksize / float(args.sample_rate)
    preroll_n = max(1, int(args.preroll_s / max(1e-6, block_s)))
    preroll = collections.deque(maxlen=preroll_n)

    streaming = False
    last_voice_ts = 0.0

    try:
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=args.blocksize,
            dtype="int16",
            channels=2,   # MUST match ALSA device
            device=args.device,
            callback=audio_cb,
        ):

            print("[voice] Listening (stateful HPF + smooth AGC, mono output)")

            while True:
                try:
                    raw = q.get(timeout=0.25)
                except queue.Empty:
                    continue

                # Decode stereo PCM16
                stereo = np.frombuffer(raw, dtype=np.int16)

                # Safety check (must be even number of samples)
                if stereo.size % 2 != 0:
                    continue

                stereo = stereo.reshape(-1, 2)

                # Pick left mic (stable, lower noise on USB mics)
                mono = stereo[:, 0]

                # Alternative (optional): average both mics
                # mono = ((stereo[:, 0].astype(np.int32) + stereo[:, 1].astype(np.int32)) // 2).astype(np.int16)

                if mono.size == 0:
                    continue

                # ---- DSP PIPELINE ----
                mono = hpf.process_i16(mono)
                mono = agc.process_i16(mono)

                level = rms_i16(mono)
                now = time.time()
                above = args.audio_threshold <= 0 or level >= args.audio_threshold

                if args.debug:
                    print(f"rms={level:6.1f} gain={agc.gain:4.2f}", file=sys.stderr)

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
