#!/usr/bin/env python3
"""
Local microphone Whisper test (Windows/client).

This script is a debugging harness to answer:
"Is my robot mic bad, or is the STT pipeline bad?"

It runs the same style of pipeline as `voice_client.py`, but reads audio from
YOUR COMPUTER MIC (sounddevice) instead of the robot ZMQ audio stream.

Can optionally send recognized text to the robot (like `voice_client.py` does).

Examples:
  # List microphone devices
  uv run python -m lerobot.control.sourccey.sourccey.voice_mic_test --list-devices

  # GPU large model (prints what it hears, no robot)
  uv run python -m lerobot.control.sourccey.sourccey.voice_mic_test --model large-v3 --device cuda --compute-type float16 --debug

  # GPU large model (sends text to robot)
  uv run python -m lerobot.control.sourccey.sourccey.voice_mic_test --robot-ip 192.168.1.213 --model large-v3 --device cuda --compute-type float16 --debug

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

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient


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

    parser.add_argument("--robot-ip", type=str, default=None, help="IP address of the robot (optional - if provided, sends text to robot)")
    parser.add_argument("--model", type=str, default="small.en")
    parser.add_argument("--device", type=str, default="auto", help="Whisper device: auto, cpu, cuda")
    parser.add_argument("--compute-type", type=str, default="auto", help="Compute type: auto, int8, int8_float16, float16, float32")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--beam-size", type=int, default=5)

    # Energy VAD knobs
    parser.add_argument("--min-utterance-s", type=float, default=0.4)
    parser.add_argument("--start-ms", type=int, default=200)
    parser.add_argument("--silence-ms", type=int, default=3000, help="Silence duration (ms) that ends an utterance. Default: 3000ms (3 seconds)")
    parser.add_argument("--preroll-ms", type=int, default=300)
    parser.add_argument("--speech-rms-mult", type=float, default=1.6)
    parser.add_argument("--min-speech-rms", type=float, default=300.0, help="Lower default than robot since local mic is cleaner.")

    # Text filtering
    parser.add_argument("--min-text-chars", type=int, default=2, help="Drop transcriptions shorter than this many non-space characters.")
    parser.add_argument("--drop-repetitions", action="store_true", help="Drop highly repetitive transcriptions (common for background noise).")
    parser.add_argument("--min-interval-s", type=float, default=0.7, help="Minimum seconds between sending recognized text")
    parser.add_argument("--mute-after-send-s", type=float, default=1.0, help="After sending text to the robot (which it will speak), ignore audio for this many seconds to prevent feedback loops.")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        print(sd.query_devices())
        return 0

    print(f"Loading Whisper model: {args.model}")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded!")

    # Connect to robot if IP provided
    robot = None
    if args.robot_ip:
        print(f"Connecting to robot at {args.robot_ip}...")
        robot_config = SourcceyClientConfig(remote_ip=args.robot_ip, id="sourccey")
        robot = SourcceyClient(robot_config)
        try:
            robot.connect()
            print("✓ Connected to robot!")
        except Exception as e:
            print(f"ERROR: Failed to connect to robot: {e}", file=sys.stderr)
            return 1

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

    # Text sending state
    last_sent = ""
    last_sent_ts = 0.0
    IDENTICAL_TEXT_COOLDOWN = 5.0  # Ignore identical text for 5 seconds
    muted_until_ts = 0.0

    def is_similar_text(text1: str, text2: str) -> bool:
        """Check if two texts are similar (simple word overlap check)"""
        if not text1 or not text2:
            return False
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return False
        # If more than 50% of words overlap, consider them similar
        overlap = len(words1 & words2)
        return overlap >= min(len(words1), len(words2)) * 0.5

    def _send_text(text: str) -> None:
        nonlocal last_sent, last_sent_ts, muted_until_ts
        if not text:
            return

        # Basic sanity filtering
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned.replace(" ", "")) < int(args.min_text_chars):
            if args.debug:
                print(f"[DROP] too short: {cleaned!r}", file=sys.stderr)
            return

        # Drop pure punctuation / non-letters
        if not re.search(r"[A-Za-z]", cleaned):
            if args.debug:
                print(f"[DROP] no letters: {cleaned!r}", file=sys.stderr)
            return

        if args.drop_repetitions:
            toks = [t for t in re.split(r"\W+", cleaned.lower()) if t]
            if len(toks) >= 6:
                uniq = len(set(toks))
                # If fewer than ~35% unique tokens, it's usually "oh oh oh ..." style noise
                if uniq / float(len(toks)) < 0.35:
                    if args.debug:
                        print(f"[DROP] repetitive: {cleaned!r}", file=sys.stderr)
                    return

        now = time.time()
        time_since_last = now - last_sent_ts
        if cleaned == last_sent and time_since_last < IDENTICAL_TEXT_COOLDOWN:
            if args.debug:
                print(f"[DROP] duplicate cooldown: {cleaned}", file=sys.stderr)
            return
        if is_similar_text(cleaned, last_sent) and time_since_last < IDENTICAL_TEXT_COOLDOWN:
            if args.debug:
                print(f"[DROP] similar cooldown: {cleaned}", file=sys.stderr)
            return
        if time_since_last < args.min_interval_s:
            if args.debug:
                print(f"[DROP] min interval: {cleaned}", file=sys.stderr)
            return

        if robot:
            success = robot.send_text(cleaned)
            if success:
                print(f"[RECOGNIZED] {cleaned}")
                last_sent = cleaned
                last_sent_ts = now
                # Prevent the robot's own TTS from being re-captured and re-sent.
                old_muted_until = muted_until_ts
                muted_until_ts = max(muted_until_ts, now + float(args.mute_after_send_s))
                if muted_until_ts > old_muted_until:
                    mute_duration = muted_until_ts - now
                    print(f"[MUTE] Entering cooldown period for {mute_duration:.1f}s (ignoring audio input)")
            else:
                print(f"[DROP] failed to send: {cleaned}", file=sys.stderr)
        else:
            # No robot connected, just print
            print(f"[HEARD] {cleaned}")
            last_sent = cleaned
            last_sent_ts = now

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
    was_muted = False  # Track muted state for transition detection

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
                    # Poll robot for messages if connected
                    if robot:
                        robot.poll_text_message()
                    continue

                # Check if we're muted (after sending text to robot)
                now_ts = time.time()
                is_muted = now_ts < muted_until_ts
                if is_muted:
                    # Drop everything while muted to avoid feedback loops.
                    if not was_muted:
                        remaining = muted_until_ts - now_ts
                        print(f"[MUTE] In cooldown period ({remaining:.1f}s remaining) - ignoring audio input")
                    was_muted = True
                    in_speech = False
                    silence_s = 0.0
                    utter_s = 0.0
                    utter_chunks = []
                    continue
                else:
                    if was_muted:
                        print(f"[MUTE] Cooldown period ended - resuming audio processing")
                    was_muted = False

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
                            print(f"[PROCESSING] Started processing audio (rms={rms:.1f} thr={thr:.1f} noise={noise_rms:.1f})")
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
                        print(f"[PROCESSING] Stopped processing audio (utterance length: {utter_s:.2f}s)")
                        if utter_s >= float(args.min_utterance_s):
                            utter_audio = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)
                            if args.debug:
                                print(f"[VAD] end utterance (len={utter_s:.2f}s)", file=sys.stderr)

                            print(f"[PROCESSING] Transcribing audio...")
                            utter_audio_16k = _resample_i16(utter_audio, in_sr=in_sr, out_sr=whisper_sr)
                            main_text = _transcribe(utter_audio_16k, model, beam_size=int(args.beam_size))
                            if main_text:
                                _send_text(main_text)
                            else:
                                print(f"[PROCESSING] Transcription completed (no text recognized)")

                        utter_chunks = []
                        silence_s = 0.0
                        utter_s = 0.0

    except KeyboardInterrupt:
        print("\n[mic] Stopping...")
        if robot:
            robot.disconnect()
            print("Disconnected from robot.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


