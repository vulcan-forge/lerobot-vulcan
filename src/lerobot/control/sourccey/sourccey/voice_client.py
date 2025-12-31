#!/usr/bin/env python3
"""
Client-side voice recognition for Sourccey (Whisper).

Receives raw audio from the robot (ZMQ SUB), segments it into utterances with an
adaptive RMS VAD (with hysteresis), transcribes it locally using faster-whisper,
and sends recognized text back to the robot host.

Run (on Windows/client machine):
  uv run python -m lerobot.control.sourccey.sourccey.voice_client --robot_ip <ROBOT_IP>
"""

import argparse
import re
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import zmq


def _configure_windows_cuda_dll_search() -> None:
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
    from faster_whisper import WhisperModel
except Exception as e:
    raise RuntimeError("Missing faster-whisper. Install: pip install faster-whisper") from e

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient


def _rms_i16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Client-side voice recognition for Sourccey (Whisper)")
    parser.add_argument("--robot_ip", type=str, required=True, help="IP address of the robot")
    parser.add_argument("--audio-port", type=int, default=5559, help="ZMQ port for subscribing to audio from robot")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (must match robot)")

    parser.add_argument("--model", type=str, default="small.en", help="Whisper model name or local path")
    parser.add_argument("--device", type=str, default="auto", help="Whisper device: auto, cpu, cuda")
    parser.add_argument("--compute-type", type=str, default="auto", help="Compute type: auto, int8, float16, float32")
    parser.add_argument("--language", type=str, default="en", help="Language code (default: en)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")

    # VAD
    parser.add_argument("--min-utterance-s", type=float, default=0.35)
    parser.add_argument("--max-utterance-s", type=float, default=5.0)
    parser.add_argument("--start-ms", type=int, default=120)
    parser.add_argument("--silence-ms", type=int, default=700)
    parser.add_argument("--preroll-ms", type=int, default=250)

    parser.add_argument("--speech-rms-mult", type=float, default=1.6, help="Start threshold = noise_rms * mult")
    parser.add_argument("--end-mult", type=float, default=0.65, help="End threshold = start_thr * end_mult (hysteresis)")
    parser.add_argument("--min-speech-rms", type=float, default=120.0)

    # Text filtering
    parser.add_argument("--min-text-chars", type=int, default=2)
    parser.add_argument("--drop-repetitions", action="store_true")
    parser.add_argument("--whisper-vad-filter", action="store_true")

    # Rate limiting / echo avoidance
    parser.add_argument("--min-interval-s", type=float, default=0.7)
    parser.add_argument("--mute-after-send-s", type=float, default=2.0)

    # Debug audio dumps (OFF by default; NEVER writes into repo implicitly)
    parser.add_argument("--dump-raw-path", type=str, default="", help="If set, append PCM16 .raw here")
    parser.add_argument("--dump-wav-once", type=str, default="", help="If set, write a single WAV file here (requires soundfile)")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")

    args = parser.parse_args(argv)

    def _load_model(name: str, *, device: str, compute_type: str) -> WhisperModel:
        try:
            return WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as e:
            if device == "cuda" and args.allow_cpu_fallback:
                print(f"WARNING: CUDA init failed ({e}). Falling back to CPU int8.", file=sys.stderr)
                return WhisperModel(name, device="cpu", compute_type="int8")
            raise

    print(f"Loading Whisper model: {args.model}")
    model = _load_model(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded!")

    print(f"Connecting to robot at {args.robot_ip}...")
    robot_config = SourcceyClientConfig(remote_ip=args.robot_ip, id="sourccey")
    robot = SourcceyClient(robot_config)

    try:
        robot.connect()
        print("✓ Connected to robot!")
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}", file=sys.stderr)
        return 1

    ctx = zmq.Context()
    audio_socket = ctx.socket(zmq.SUB)
    audio_socket.connect(f"tcp://{args.robot_ip}:{args.audio_port}")
    audio_socket.setsockopt(zmq.SUBSCRIBE, b"")
    print(f"Subscribed to audio stream from {args.robot_ip}:{args.audio_port}...")

    poller = zmq.Poller()
    poller.register(audio_socket, zmq.POLLIN)

    last_sent = ""
    last_sent_ts = 0.0
    IDENTICAL_TEXT_COOLDOWN = 5.0
    muted_until_ts = 0.0

    # Adaptive noise floor
    noise_rms = 220.0
    noise_alpha_idle = 0.02
    noise_alpha_silence = 0.05

    # VAD state
    in_speech = False
    speech_run_s = 0.0
    silence_s = 0.0
    utter_s = 0.0

    # Sample-accurate preroll buffer
    preroll_samples = int(args.sample_rate * (args.preroll_ms / 1000.0))
    preroll_buf: deque[np.ndarray] = deque()
    preroll_count = 0

    utter_chunks: list[np.ndarray] = []

    dump_raw_f = None
    wrote_wav_once = False

    if args.dump_raw_path:
        # user explicitly asked for it — keep it out of the repo by choosing a safe path
        dump_raw_f = open(args.dump_raw_path, "ab")

    def is_similar_text(text1: str, text2: str) -> bool:
        if not text1 or not text2:
            return False
        w1 = set(text1.lower().split())
        w2 = set(text2.lower().split())
        if not w1 or not w2:
            return False
        overlap = len(w1 & w2)
        return overlap >= min(len(w1), len(w2)) * 0.5

    def _transcribe(audio_i16: np.ndarray) -> str:
        if audio_i16.size == 0:
            return ""
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).copy()
        segments, _info = model.transcribe(
            audio_f32,
            language=args.language,
            beam_size=int(args.beam_size),
            vad_filter=bool(args.whisper_vad_filter),
        )
        return " ".join((getattr(seg, "text", "") or "").strip() for seg in segments).strip()

    def _send_text(text: str) -> None:
        nonlocal last_sent, last_sent_ts, muted_until_ts
        if not text:
            return

        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned.replace(" ", "")) < int(args.min_text_chars):
            if args.debug:
                print(f"[DROP] too short: {cleaned!r}", file=sys.stderr)
            return

        if not re.search(r"[A-Za-z]", cleaned):
            if args.debug:
                print(f"[DROP] no letters: {cleaned!r}", file=sys.stderr)
            return

        if args.drop_repetitions:
            toks = [t for t in re.split(r"\W+", cleaned.lower()) if t]
            if len(toks) >= 6:
                uniq = len(set(toks))
                if uniq / float(len(toks)) < 0.35:
                    if args.debug:
                        print(f"[DROP] repetitive: {cleaned!r}", file=sys.stderr)
                    return

        now = time.time()
        dt = now - last_sent_ts
        if cleaned == last_sent and dt < IDENTICAL_TEXT_COOLDOWN:
            return
        if is_similar_text(cleaned, last_sent) and dt < IDENTICAL_TEXT_COOLDOWN:
            return
        if dt < float(args.min_interval_s):
            return

        if robot.send_text(cleaned):
            print(f"[RECOGNIZED] {cleaned}")
            last_sent = cleaned
            last_sent_ts = now
            muted_until_ts = max(muted_until_ts, now + float(args.mute_after_send_s))

    try:
        while True:
            # Poll sockets so we don’t spam exceptions / busy loop
            events = dict(poller.poll(timeout=50))
            if audio_socket in events and events[audio_socket] & zmq.POLLIN:
                audio_data = audio_socket.recv()

                now_ts = time.time()
                if now_ts < muted_until_ts:
                    # reset VAD while muted
                    in_speech = False
                    speech_run_s = 0.0
                    silence_s = 0.0
                    utter_s = 0.0
                    utter_chunks = []
                    preroll_buf.clear()
                    preroll_count = 0
                    continue

                audio_i16 = np.frombuffer(audio_data, dtype=np.int16).copy()
                if audio_i16.size == 0:
                    continue

                # Optional debug dumps (explicit flags only)
                if dump_raw_f is not None:
                    dump_raw_f.write(audio_i16.tobytes())

                if args.dump_wav_once and not wrote_wav_once:
                    try:
                        import soundfile as sf  # optional dependency
                        sf.write(args.dump_wav_once, audio_i16, args.sample_rate)
                        wrote_wav_once = True
                        print(f"[debug] wrote wav: {args.dump_wav_once}", file=sys.stderr)
                    except Exception as e:
                        print(f"[debug] wav write failed (install soundfile): {e}", file=sys.stderr)
                        wrote_wav_once = True  # don’t keep trying

                chunk_s = float(audio_i16.size) / float(args.sample_rate)
                rms = _rms_i16(audio_i16)

                # Update noise estimate:
                # - when NOT in speech
                # - and also during silence while in speech (so the floor can recover)
                if not in_speech:
                    noise_rms = (1.0 - noise_alpha_idle) * noise_rms + noise_alpha_idle * rms
                else:
                    # if we’re “in speech” but this chunk is quiet, let noise track faster
                    if rms < max(noise_rms * 1.2, float(args.min_speech_rms)):
                        noise_rms = (1.0 - noise_alpha_silence) * noise_rms + noise_alpha_silence * rms

                start_thr = max(float(args.min_speech_rms), float(noise_rms) * float(args.speech_rms_mult))
                end_thr = max(float(args.min_speech_rms), start_thr * float(args.end_mult))

                is_speech_now = rms >= (start_thr if not in_speech else end_thr)
                start_s = float(args.start_ms) / 1000.0
                silence_end_s = float(args.silence_ms) / 1000.0

                if args.debug:
                    print(
                        f"[VAD] rms={rms:6.1f} noise={noise_rms:6.1f} start_thr={start_thr:6.1f} end_thr={end_thr:6.1f} in={in_speech} speech={is_speech_now}",
                        file=sys.stderr,
                    )

                # Maintain sample-accurate preroll
                preroll_buf.append(audio_i16)
                preroll_count += audio_i16.size
                while preroll_count > preroll_samples and preroll_buf:
                    popped = preroll_buf.popleft()
                    preroll_count -= popped.size

                if not in_speech:
                    if is_speech_now:
                        speech_run_s += chunk_s
                        if speech_run_s >= start_s:
                            in_speech = True
                            silence_s = 0.0
                            utter_s = 0.0
                            utter_chunks = list(preroll_buf)  # include preroll
                            if args.debug:
                                print("[VAD] start", file=sys.stderr)
                    else:
                        speech_run_s = 0.0
                else:
                    utter_chunks.append(audio_i16)
                    utter_s += chunk_s

                    if is_speech_now:
                        silence_s = 0.0
                    else:
                        silence_s += chunk_s

                    reached_silence_end = silence_s >= silence_end_s
                    reached_max_len = utter_s >= float(args.max_utterance_s)

                    if reached_silence_end or reached_max_len:
                        in_speech = False
                        speech_run_s = 0.0

                        if utter_s >= float(args.min_utterance_s):
                            utter_audio = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)
                            if args.debug:
                                print(f"[VAD] end len={utter_s:.2f}s silence={silence_s:.2f}s", file=sys.stderr)

                            text = _transcribe(utter_audio)
                            _send_text(text)

                        utter_chunks = []
                        silence_s = 0.0
                        utter_s = 0.0

            else:
                # No audio; poll robot for incoming messages, keep CPU low
                robot.poll_text_message()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if dump_raw_f is not None:
            try:
                dump_raw_f.close()
            except Exception:
                pass
        audio_socket.close()
        ctx.term()
        robot.disconnect()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
