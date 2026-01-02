#!/usr/bin/env python3
"""
Client-side voice recognition for Sourccey (Whisper).

Receives mono PCM16 audio from the robot (ZMQ SUB),
segments speech with adaptive RMS VAD,
transcribes using faster-whisper,
normalizes results,
and sends text back to the robot.
"""

import argparse
import re
import sys
import time
from typing import Optional

import numpy as np
import zmq

# -----------------------------
# Windows CUDA DLL helper
# -----------------------------
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
        pass


_configure_windows_cuda_dll_search()

from faster_whisper import WhisperModel
from lerobot.robots.sourccey.sourccey.sourccey import (
    SourcceyClientConfig,
    SourcceyClient,
)

# -----------------------------
# Text normalization
# -----------------------------
def normalize_robot_terms(text: str) -> str:
    replacements = {
        r"\b(sorcy|sorsi|orsi|doris|sourcey|sourcy|isourcing)\b": "Sourccey",
    }
    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def is_similar_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return False
    return len(wa & wb) >= min(len(wa), len(wb)) * 0.5


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robot_ip", required=True)
    p.add_argument("--audio-port", type=int, default=5559)
    p.add_argument("--sample-rate", type=int, default=16000)

    p.add_argument("--model", default="medium.en")
    p.add_argument("--device", default="auto")
    p.add_argument("--compute-type", default="auto")
    p.add_argument("--language", default="en")

    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--min-utterance-s", type=float, default=0.5)
    p.add_argument("--max-utterance-s", type=float, default=5.5)
    p.add_argument("--start-ms", type=int, default=150)
    p.add_argument("--silence-ms", type=int, default=1800)

    p.add_argument("--speech-rms-mult", type=float, default=1.15)
    p.add_argument(
        "--end-mult",
        type=float,
        default=0.6,
        help="Multiplier for end-of-speech threshold (hysteresis)"
    )
    p.add_argument("--min-speech-rms", type=float, default=160.0)

    p.add_argument("--min-text-chars", type=int, default=2)
    p.add_argument("--min-interval-s", type=float, default=0.7)
    p.add_argument("--mute-after-send-s", type=float, default=3.0)

    p.add_argument("--drop-repetitions", action="store_true")
    p.add_argument("--whisper-vad-filter", action="store_true")
    p.add_argument("--allow-cpu-fallback", action="store_true")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args(argv)

    # -----------------------------
    # Load Whisper
    # -----------------------------
    def load_model():
        try:
            return WhisperModel(
                args.model,
                device=args.device,
                compute_type=args.compute_type,
            )
        except Exception as e:
            if args.allow_cpu_fallback:
                print("[WARN] CUDA failed, falling back to CPU int8", file=sys.stderr)
                return WhisperModel(args.model, device="cpu", compute_type="int8")
            raise

    print(f"Loading Whisper model: {args.model}")
    model = load_model()
    print("Model loaded!")

    # -----------------------------
    # Robot connection
    # -----------------------------
    robot = SourcceyClient(
        SourcceyClientConfig(remote_ip=args.robot_ip, id="sourccey")
    )
    robot.connect()
    print("✓ Connected to robot")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.robot_ip}:{args.audio_port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")

    # -----------------------------
    # VAD state
    # -----------------------------
    noise_rms = 600.0
    noise_alpha = 0.02

    preroll = []
    preroll_max = int(0.3 / 0.2)

    utter_chunks = []
    in_speech = False
    silence_s = 0.0
    utter_s = 0.0

    last_sent = ""
    last_sent_ts = 0.0
    muted_until = 0.0

    # -----------------------------
    # Transcription helper
    # -----------------------------
    def transcribe(audio_i16: np.ndarray) -> str:
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        segments, info = model.transcribe(
            audio_f32,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=args.whisper_vad_filter,
            initial_prompt=(
                "This is a conversation with a robot named Sourccey. "
                "Common words include: Sourccey, robot, move, follow, stop, hello, task."
            ),
        )

        return " ".join(seg.text.strip() for seg in segments).strip()

    # -----------------------------
    # Main loop
    # -----------------------------
    try:
        while True:
            try:
                data = sock.recv(zmq.NOBLOCK)
                now = time.time()

                if now < muted_until:
                    continue

                audio = np.frombuffer(data, dtype=np.int16)
                if audio.size == 0:
                    continue

                rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

                if not in_speech:
                    noise_rms = (1 - noise_alpha) * noise_rms + noise_alpha * rms

                thr = max(args.min_speech_rms, noise_rms * args.speech_rms_mult)
                start_thr = max(args.min_speech_rms, noise_rms * args.speech_rms_mult)
                end_thr = max(args.min_speech_rms * 0.5, noise_rms * args.end_mult)

                if not in_speech:
                    is_speech = rms >= start_thr
                else:
                    is_speech = rms >= end_thr

                chunk_s = audio.size / args.sample_rate

                preroll.append(audio)
                if len(preroll) > preroll_max:
                    preroll.pop(0)

                if not in_speech:
                    if is_speech:
                        utter_s += chunk_s
                        if utter_s >= args.start_ms / 1000:
                            in_speech = True
                            utter_chunks = list(preroll)
                            silence_s = 0.0
                            utter_s = 0.0
                    else:
                        utter_s = 0.0

                if in_speech:
                    utter_chunks.append(audio)
                    utter_s += chunk_s

                    if is_speech:
                        silence_s = 0.0
                    else:
                        silence_s += chunk_s
                        noise_rms = (1 - noise_alpha) * noise_rms + noise_alpha * rms

                    if (
                        silence_s >= args.silence_ms / 1000
                        or utter_s >= args.max_utterance_s
                    ):
                        in_speech = False
                        if utter_s >= args.min_utterance_s:
                            full = np.concatenate(utter_chunks)
                            text = transcribe(full)
                            text = normalize_robot_terms(text)

                            if (
                                text
                                and len(text.replace(" ", "")) >= args.min_text_chars
                                and now - last_sent_ts >= args.min_interval_s
                                and not is_similar_text(text, last_sent)
                            ):
                                if robot.send_text(text):
                                    print(f"[RECOGNIZED] {text}")
                                    last_sent = text
                                    last_sent_ts = now
                                    muted_until = now + args.mute_after_send_s

                        utter_chunks = []
                        silence_s = 0.0
                        utter_s = 0.0

            except zmq.Again:
                robot.poll_text_message()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()
        ctx.term()
        robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
