"""
Voice listener that streams (optionally beamformed) raw audio to client for remote speech recognition.

This version runs on the robot and publishes int16 mono audio chunks over ZMQ (PUB socket).
It supports true 2-mic capture on Raspberry Pi (stereo interleaved) and can do simple
delay-and-sum beamforming in software to improve far-field speech pickup.

Run (on robot):
  uv run python -m lerobot.robots.sourccey.sourccey.sourccey.voice_listener_audio_stream

Typical (force stereo input + beamforming -> mono out):
  uv run python -m lerobot.robots.sourccey.sourccey.sourccey.voice_listener_audio_stream \
    --device 2 --input-channels 2 --beamform

Notes:
- Output is ALWAYS mono int16 (best for ASR).
- If you pass --input-channels 2, your hardware MUST support stereo capture (you confirmed it does).
- ZMQ CONFLATE is optional but NOT recommended for STT.
"""

from __future__ import annotations

import argparse
import collections
import queue
import sys
import time
from typing import Optional

import numpy as np
import zmq

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing voice dependencies. Install `sounddevice` in the robot environment.") from e

from .config_sourccey import SourcceyHostConfig


# -----------------------------
# Audio DSP helpers (2-mic)
# -----------------------------
def _rms(x_i16: np.ndarray) -> float:
    if x_i16.size == 0:
        return 0.0
    xf = x_i16.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


def estimate_delay_samples(left_i16: np.ndarray, right_i16: np.ndarray, max_delay_samp: int) -> int:
    """
    Estimate integer sample delay between L/R using cross-correlation.
    Positive delay means "right lags left" (right arrives later).
    """
    n = min(left_i16.size, right_i16.size)
    if n <= 8:
        return 0

    # Use int32 to avoid overflow in correlation products.
    l = left_i16[:n].astype(np.int32)
    r = right_i16[:n].astype(np.int32)

    # Remove DC offset (helps correlation)
    l = l - int(np.mean(l))
    r = r - int(np.mean(r))

    corr = np.correlate(l, r, mode="full")
    delay = int(np.argmax(corr) - (n - 1))
    if max_delay_samp > 0:
        delay = int(np.clip(delay, -max_delay_samp, max_delay_samp))
    return delay


def apply_delay(x: np.ndarray, delay: int) -> np.ndarray:
    """
    Shift x by integer samples:
      delay > 0 => prepend zeros (shift right)
      delay < 0 => drop from front (shift left)
    """
    if delay == 0:
        return x
    if delay > 0:
        return np.pad(x, (delay, 0), mode="constant")[: x.size]
    # delay < 0
    d = -delay
    if d >= x.size:
        return np.zeros_like(x)
    return np.pad(x[d:], (0, d), mode="constant")


def beamform_delay_and_sum(
    left_i16: np.ndarray,
    right_i16: np.ndarray,
    *,
    sample_rate: int,
    max_delay_ms: float = 1.0,
    doa_debug: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Simple 2-mic delay-and-sum beamformer.
    Returns (mono_i16, estimated_delay_samples).

    max_delay_ms bounds the search/acceptance of delay to avoid wild jumps.
    """
    max_delay_samp = int(round((max_delay_ms / 1000.0) * float(sample_rate)))
    delay = estimate_delay_samples(left_i16, right_i16, max_delay_samp=max_delay_samp)

    # Align the later channel back in time
    if delay > 0:
        # right lags left => shift right backwards (left forward) OR right forward?
        # easiest: shift right leftwards by delay (apply_delay with -delay)
        right_aligned = apply_delay(right_i16, -delay)
        left_aligned = left_i16
    elif delay < 0:
        # left lags right => shift left leftwards by -delay
        left_aligned = apply_delay(left_i16, delay)  # delay is negative => shift left
        right_aligned = right_i16
    else:
        left_aligned = left_i16
        right_aligned = right_i16

    mono32 = (left_aligned.astype(np.int32) + right_aligned.astype(np.int32)) // 2
    mono_i16 = np.clip(mono32, -32768, 32767).astype(np.int16)

    if doa_debug:
        # Not a true angle; just a useful sign/magnitude indicator for “which side”
        print(f"[beamform] delay_samples={delay}", file=sys.stderr)

    return mono_i16, delay


def agc_simple(
    mono_i16: np.ndarray,
    *,
    target_rms: float,
    max_gain: float,
    min_rms: float = 50.0,
) -> np.ndarray:
    """
    Lightweight AGC: scale chunk to target RMS with a gain cap.
    - min_rms prevents cranking silence/noise.
    """
    if target_rms <= 0:
        return mono_i16

    current = _rms(mono_i16)
    if current < min_rms:
        return mono_i16

    gain = float(target_rms) / float(current)
    gain = float(np.clip(gain, 1.0 / max_gain if max_gain > 0 else gain, max_gain if max_gain > 0 else gain))

    y = (mono_i16.astype(np.float32) * gain).astype(np.int32)
    return np.clip(y, -32768, 32767).astype(np.int16)


# -----------------------------
# ZMQ publisher
# -----------------------------
class AudioStreamPublisher:
    def __init__(self, port: int, *, conflate: bool, sndhwm: int):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)

        # IMPORTANT: Do NOT conflate audio by default.
        # Conflation keeps only the newest message and drops intermediate chunks, which can
        # destroy speech recognition (missing phonemes, choppy audio). Keep it optional.
        if conflate:
            self._sock.setsockopt(zmq.CONFLATE, 1)
        try:
            self._sock.setsockopt(zmq.SNDHWM, int(sndhwm))
        except Exception:
            pass

        self._sock.bind(f"tcp://*:{port}")
        # Give subscribers time to connect
        time.sleep(0.5)

    def send_audio(self, audio_data: bytes) -> bool:
        try:
            self._sock.send(audio_data, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def close(self):
        try:
            self._sock.close()
        finally:
            self._ctx.term()


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sourccey voice listener - streams mono audio to client for recognition")

    # Audio capture
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--input-channels",
        type=int,
        default=1,
        help="Input channels from device. Use 2 for your L/R mics (stereo interleaved).",
    )
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index (optional)")

    parser.add_argument(
        "--blocksize",
        type=int,
        default=3200,
        help="Frames per chunk. Increase if you see input overflow. Default: 3200 (~0.2s at 16kHz).",
    )

    # Beamforming / DSP
    parser.add_argument(
        "--beamform",
        action="store_true",
        help="Enable 2-mic delay-and-sum beamforming (requires --input-channels 2).",
    )
    parser.add_argument(
        "--max-delay-ms",
        type=float,
        default=1.0,
        help="Max allowed inter-mic delay in ms for beamforming correlation clamp (default: 1.0).",
    )
    parser.add_argument(
        "--agc-target-rms",
        type=float,
        default=0.0,
        help="Enable simple AGC by setting a target RMS (e.g., 3000). 0 disables AGC.",
    )
    parser.add_argument(
        "--agc-max-gain",
        type=float,
        default=6.0,
        help="Max AGC gain multiplier (default: 6x).",
    )
    parser.add_argument(
        "--debug-doa",
        action="store_true",
        help="Print estimated delay samples for each chunk (noisy).",
    )

    # ZMQ
    parser.add_argument(
        "--audio-port",
        type=int,
        default=SourcceyHostConfig().port_zmq_audio,
        help="ZMQ port for publishing audio (default from SourcceyHostConfig)",
    )
    parser.add_argument(
        "--zmq-conflate",
        action="store_true",
        help="Enable ZMQ CONFLATE (keep only latest audio chunk). Not recommended for STT.",
    )
    parser.add_argument("--zmq-sndhwm", type=int, default=100, help="ZMQ send high-water mark for audio PUB socket.")

    # Streaming control / VAD-ish gating
    parser.add_argument(
        "--audio-threshold",
        type=float,
        default=0.0,
        help="Minimum mono RMS to stream. 0 = stream everything (recommended for best accuracy).",
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
    parser.add_argument("--debug-audio-levels", action="store_true", help="Print mono RMS audio levels for debugging")

    args = parser.parse_args(argv)

    if args.beamform and args.input_channels != 2:
        print("[voice_listener] ERROR: --beamform requires --input-channels 2", file=sys.stderr)
        return 2

    publisher = AudioStreamPublisher(args.audio_port, conflate=args.zmq_conflate, sndhwm=args.zmq_sndhwm)

    # Queue holds raw capture bytes (interleaved int16) to keep callback fast
    audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=80)
    last_overflow_log_ts = 0.0

    def audio_callback(indata, frames, time_info, status):  # noqa: ANN001
        nonlocal last_overflow_log_ts
        if status and getattr(status, "input_overflow", False):
            now = time.time()
            if (now - last_overflow_log_ts) >= 1.0:
                last_overflow_log_ts = now
                print("[voice_listener] Audio status: input overflow (device buffer overflow)", file=sys.stderr)

        # Avoid work in callback; just enqueue raw bytes
        try:
            audio_q.put_nowait(bytes(indata))
        except queue.Full:
            # Drop if overloaded (better than blocking callback and causing more overflows)
            pass

    print(f"[voice_listener] Publishing audio on port {args.audio_port}")
    print("[voice_listener] Listening... (output is MONO int16)")
    if args.device is None:
        print("[voice_listener] Device: (default)", file=sys.stderr)
    else:
        print(f"[voice_listener] Device index: {args.device}", file=sys.stderr)
    print(f"[voice_listener] Input channels: {args.input_channels}", file=sys.stderr)
    print(f"[voice_listener] Beamforming: {'ON' if args.beamform else 'OFF'}", file=sys.stderr)

    try:
        blocksize = int(args.blocksize)
        sample_rate = int(args.sample_rate)

        bytes_per_frame_i16 = 2  # int16
        # Capture chunk size in bytes (raw input)
        bytes_per_chunk_in = blocksize * int(args.input_channels) * bytes_per_frame_i16

        block_s = float(blocksize) / float(sample_rate)
        preroll_chunks = max(0, int(round(args.preroll_s / max(1e-6, block_s))))
        preroll_buf: "collections.deque[bytes]" = collections.deque(maxlen=max(1, preroll_chunks))

        stream_all = args.audio_threshold <= 0.0
        streaming_active = False
        last_above_ts = 0.0

        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype="int16",
            channels=int(args.input_channels),
            callback=audio_callback,
            device=args.device,
        ):
            while True:
                try:
                    data_in = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Defensive: allow variable chunk sizes, but prefer fixed-size behavior
                if len(data_in) % 2 != 0:
                    # must be multiple of 2 bytes for int16
                    continue

                # Convert to mono int16 (optionally beamformed)
                if args.input_channels == 1:
                    mono = np.frombuffer(data_in, dtype=np.int16)
                else:
                    # stereo interleaved: [L, R, L, R, ...]
                    stereo = np.frombuffer(data_in, dtype=np.int16)
                    if stereo.size % 2 != 0:
                        continue
                    stereo = stereo.reshape(-1, 2)
                    left = stereo[:, 0]
                    right = stereo[:, 1]

                    if args.beamform:
                        mono, _delay = beamform_delay_and_sum(
                            left,
                            right,
                            sample_rate=sample_rate,
                            max_delay_ms=float(args.max_delay_ms),
                            doa_debug=bool(args.debug_doa),
                        )
                    else:
                        # Simple downmix (still better than picking one mic)
                        mono32 = (left.astype(np.int32) + right.astype(np.int32)) // 2
                        mono = np.clip(mono32, -32768, 32767).astype(np.int16)

                # Optional AGC
                if args.agc_target_rms and args.agc_target_rms > 0.0:
                    mono = agc_simple(mono, target_rms=float(args.agc_target_rms), max_gain=float(args.agc_max_gain))

                # Compute mono RMS only if needed
                if stream_all and (not args.debug_audio_levels):
                    publisher.send_audio(mono.tobytes())
                    continue

                rms_level = _rms(mono)

                if args.debug_audio_levels:
                    status = "✓" if (stream_all or rms_level >= args.audio_threshold) else "✗"
                    print(
                        f"[audio] mono RMS: {rms_level:.1f} (threshold: {args.audio_threshold:.1f}) {status}",
                        file=sys.stderr,
                    )

                # Gating + preroll/hangover logic (operates on MONO bytes)
                mono_bytes = mono.tobytes()

                # If capture gives variable chunk sizes, still do best-effort (but preroll assumes consistent timing)
                fixed_size = (len(data_in) == bytes_per_chunk_in)

                if fixed_size:
                    preroll_buf.append(mono_bytes)

                now = time.time()
                above = stream_all or (rms_level >= args.audio_threshold)
                if above:
                    last_above_ts = now
                    if not streaming_active:
                        streaming_active = True
                        if fixed_size:
                            for chunk in list(preroll_buf):
                                publisher.send_audio(chunk)
                    publisher.send_audio(mono_bytes)
                else:
                    if streaming_active and (now - last_above_ts) <= max(0.0, args.hangover_s):
                        publisher.send_audio(mono_bytes)
                    elif streaming_active:
                        streaming_active = False

    except KeyboardInterrupt:
        print("[voice_listener] Stopping...")
        return 0
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
