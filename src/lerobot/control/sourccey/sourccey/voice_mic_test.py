#!/usr/bin/env python3
"""
Local microphone Whisper test (Windows/client).

This script is a debugging harness to answer:
"Is my robot mic bad, or is the STT pipeline bad?"

It runs the same style of pipeline as `voice_client.py`, but reads audio from
YOUR COMPUTER MIC (sounddevice) instead of the robot ZMQ audio stream.

Examples:
  # List microphone devices
  uv run python -m lerobot.control.sourccey.sourccey.voice_mic_test --list-devices

  # GPU large model (prints what it hears)
  uv run python -m lerobot.control.sourccey.sourccey.voice_mic_test --model large-v3 --device cuda --compute-type float16 --debug

  # CPU test
  uv run python -m lerobot.control.sourccey.sourccey.voice_mic_test --model small.en --device cpu --compute-type int8 --debug
"""

from __future__ import annotations

import argparse
import queue
import re
import sys
import time
from typing import Optional

import numpy as np


def _configure_windows_cuda_dll_search() -> None:
    """Same Windows DLL-path fix as in voice_client.py (must run before importing faster_whisper)."""

    if not sys.platform.startswith("win"):
        return
    try:
        import importlib.util
        import os

        def _add_nvidia_bin(mod: str) -> None:
            spec = importlib.util.find_spec(mod)
            if not spec or not spec.submodule_search_locations:
                return
            pkg_dir = spec.submodule_search_locations[0]
            bin_dir = os.path.join(pkg_dir, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

        _add_nvidia_bin("nvidia.cudnn")
        _add_nvidia_bin("nvidia.cublas")
        _add_nvidia_bin("nvidia.cuda_runtime")
    except Exception:
        return


_configure_windows_cuda_dll_search()

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing sounddevice. Install: pip install sounddevice") from e

try:
    from faster_whisper import WhisperModel
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing faster-whisper. Install: pip install faster-whisper") from e


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local microphone Whisper test (prints what it hears)")
    parser.add_argument("--list-devices", action="store_true", help="Print sounddevice devices and exit.")
    parser.add_argument("--device-index", type=int, default=None, help="sounddevice input device index (optional)")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Input sample rate. Default: use microphone's default samplerate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Input channels. Default: use microphone's max input channels.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=None,
        help="Frames per block. Default: ~200ms at the chosen input samplerate.",
    )

    parser.add_argument("--model", type=str, default="small.en")
    parser.add_argument("--device", type=str, default="auto", help="Whisper device: auto, cpu, cuda")
    parser.add_argument("--compute-type", type=str, default="auto", help="Compute type: auto, int8, int8_float16, float16, float32")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--beam-size", type=int, default=5)

    # Energy VAD knobs
    parser.add_argument("--min-utterance-s", type=float, default=0.4)
    parser.add_argument("--start-ms", type=int, default=200)
    parser.add_argument("--silence-ms", type=int, default=600)
    parser.add_argument("--preroll-ms", type=int, default=300)
    parser.add_argument("--speech-rms-mult", type=float, default=1.6)
    parser.add_argument("--min-speech-rms", type=float, default=300.0, help="Lower default than robot since local mic is cleaner.")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        print(sd.query_devices())
        return 0

    print(f"Loading Whisper model: {args.model}")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded!")

    whisper_sr = 16000

    def _downmix_to_mono_i16(x: np.ndarray) -> np.ndarray:
        # x is (frames, channels) or (frames,)
        if x.ndim == 1:
            return x.astype(np.int16, copy=False)
        if x.shape[1] == 1:
            return x[:, 0].astype(np.int16, copy=False)
        # mean across channels
        y = np.mean(x.astype(np.float32), axis=1)
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)

    def _resample_i16(x: np.ndarray, in_sr: float, out_sr: float) -> np.ndarray:
        if int(in_sr) == int(out_sr):
            return x
        if x.size == 0:
            return x
        ratio = float(out_sr) / float(in_sr)
        out_len = max(1, int(round(x.size * ratio)))
        xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
        fp = x.astype(np.float32)
        x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
        y = np.interp(x_new, xp, fp)
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)

    def _transcribe(audio_i16_16k: np.ndarray, model_obj: WhisperModel, *, beam_size: int) -> str:
        if audio_i16_16k.size == 0:
            return ""
        audio_f32 = (audio_i16_16k.astype(np.float32) / 32768.0).copy()
        segments, _info = model_obj.transcribe(
            audio_f32,
            language=args.language,
            beam_size=int(beam_size),
            vad_filter=False,  # we do our own segmentation
        )
        return " ".join((getattr(seg, "text", "") or "").strip() for seg in segments).strip()

    # Audio capture (use mic defaults unless user overrides)
    input_device = args.device_index if args.device_index is not None else sd.default.device[0]
    dev_info = sd.query_devices(input_device, "input")
    in_sr = float(args.sample_rate) if args.sample_rate else float(dev_info["default_samplerate"])
    in_ch = int(args.channels) if args.channels else int(dev_info["max_input_channels"] or 1)
    blocksize = int(args.blocksize) if args.blocksize else max(256, int(round(in_sr * 0.2)))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=100)
    last_overflow_ts = 0.0

    def audio_callback(indata, frames, time_info, status):  # noqa: ANN001
        nonlocal last_overflow_ts
        if status and getattr(status, "input_overflow", False):
            now = time.time()
            if (now - last_overflow_ts) > 1.0:
                last_overflow_ts = now
                print("[mic] input overflow (dropping frames)", file=sys.stderr)
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass

    # VAD state
    noise_rms = 200.0
    noise_alpha = 0.02
    in_speech = False
    silence_s = 0.0
    speech_run_s = 0.0
    utter_s = 0.0

    chunk_s_assumed = float(blocksize) / float(in_sr)
    preroll_max_chunks = max(1, int(round((args.preroll_ms / 1000.0) / max(1e-6, chunk_s_assumed))))
    preroll: list[np.ndarray] = []
    utter_chunks: list[np.ndarray] = []

    print(f"[mic] Device: {dev_info.get('name')}")
    print(f"[mic] Using input samplerate={in_sr:.0f}Hz channels={in_ch} blocksize={blocksize} (~{chunk_s_assumed:.2f}s)")
    print("[mic] Listening... (Ctrl+C to stop)")
    try:
        with sd.InputStream(
            samplerate=in_sr,
            channels=in_ch,
            dtype="int16",
            blocksize=blocksize,
            device=input_device,
            callback=audio_callback,
        ):
            while True:
                try:
                    block = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                mono_i16 = _downmix_to_mono_i16(block)
                chunk_s = float(mono_i16.size) / float(in_sr)

                rms = float(np.sqrt(np.mean(mono_i16.astype(np.float32) ** 2)))
                if not in_speech:
                    noise_rms = (1.0 - noise_alpha) * noise_rms + noise_alpha * rms

                thr = max(float(args.min_speech_rms), float(noise_rms) * float(args.speech_rms_mult))
                is_speech = rms >= thr

                preroll.append(mono_i16.copy())
                if len(preroll) > preroll_max_chunks:
                    preroll.pop(0)

                start_s = float(args.start_ms) / 1000.0

                if not in_speech:
                    if is_speech:
                        speech_run_s += chunk_s
                        if speech_run_s >= start_s:
                            in_speech = True
                            silence_s = 0.0
                            utter_s = 0.0
                            utter_chunks = list(preroll)
                            if args.debug:
                                print(f"[VAD] start (rms={rms:.1f} thr={thr:.1f} noise={noise_rms:.1f})", file=sys.stderr)
                    else:
                        speech_run_s = 0.0

                if in_speech:
                    utter_chunks.append(mono_i16.copy())
                    utter_s += chunk_s
                    if is_speech:
                        silence_s = 0.0
                    else:
                        silence_s += chunk_s

                    if silence_s >= (float(args.silence_ms) / 1000.0):
                        in_speech = False
                        speech_run_s = 0.0
                        if utter_s >= float(args.min_utterance_s):
                            utter_audio = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)
                            if args.debug:
                                print(f"[VAD] end utterance (len={utter_s:.2f}s)", file=sys.stderr)

                            utter_audio_16k = _resample_i16(utter_audio, in_sr=in_sr, out_sr=whisper_sr)
                            main_text = _transcribe(utter_audio_16k, model, beam_size=int(args.beam_size))
                            if main_text:
                                print(f"[HEARD] {main_text}")

                        utter_chunks = []
                        silence_s = 0.0
                        utter_s = 0.0

    except KeyboardInterrupt:
        print("\n[mic] Stopping...")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


