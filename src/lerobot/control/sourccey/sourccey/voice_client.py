#!/usr/bin/env python3
"""
Client-side voice recognition for Sourccey (Whisper).

Receives raw audio from the robot (ZMQ SUB), transcribes it locally using
`faster-whisper`, and sends recognized text back to the robot host.

Run (on Windows/client machine):
  uv run python -m lerobot.control.sourccey.sourccey.voice_client --robot_ip <ROBOT_IP>
"""

import argparse
import re
import sys
import time
from typing import Optional

import numpy as np
import zmq

def _configure_windows_cuda_dll_search() -> None:
    """
    On Windows Python 3.8+, DLL search paths are restricted.

    CUDA-enabled `ctranslate2` (used by `faster-whisper`) loads cuDNN/cuBLAS DLLs
    at runtime. If these DLLs are installed via pip (e.g. `nvidia-cudnn-cu12`),
    we must add their `.../bin` directories to the DLL search path *before*
    importing `faster_whisper` / `ctranslate2`.
    """

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
                # Add to Windows DLL search (Python 3.8+) AND prepend to PATH.
                # Some native loaders still rely on PATH rather than the user DLL dirs.
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

        # These packages come from pip wheels like:
        # - nvidia-cudnn-cu12  -> nvidia.cudnn
        # - nvidia-cublas-cu12 -> nvidia.cublas
        # - nvidia-cuda-runtime-cu12 -> nvidia.cuda_runtime
        _add_nvidia_bin("nvidia.cudnn")
        _add_nvidia_bin("nvidia.cublas")
        _add_nvidia_bin("nvidia.cuda_runtime")
    except Exception:
        # Best-effort only; if CUDA init fails, you'll see a clear error.
        return


_configure_windows_cuda_dll_search()

try:
    from faster_whisper import WhisperModel
except Exception as e:
    raise RuntimeError("Missing faster-whisper. Install: pip install faster-whisper") from e

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Client-side voice recognition for Sourccey (Whisper)")
    parser.add_argument(
        "--robot_ip",
        type=str,
        required=True,
        help="IP address of the robot",
    )
    parser.add_argument(
        "--audio-port",
        type=int,
        default=5559,
        help="ZMQ port for subscribing to audio from robot",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (must match robot)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small.en",
        help="Whisper model name (e.g. tiny.en, base.en, small.en, medium.en) or a local path.",
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default=None,
        help="Wake word/phrase (e.g. Hello). If set: nothing is sent until wake word is heard; then ONE command is captured and sent.",
    )
    parser.add_argument(
        "--wake-model",
        type=str,
        default=None,
        help="Whisper model used for wake-word detection. Default: same as --model.",
    )
    parser.add_argument(
        "--wake-beam-size",
        type=int,
        default=1,
        help="Beam size for wake-word decoding (default: 1).",
    )
    parser.add_argument(
        "--wake-timeout-s",
        type=float,
        default=10.0,
        help="If wake word is heard but no command follows within this many seconds, return to waiting.",
    )
    parser.add_argument(
        "--wake-max-words",
        type=int,
        default=3,
        help="Max words allowed in wake transcription to count as a wake hit (reduces false wake from noise).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Whisper device: auto, cpu, cuda",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="auto",
        help="Compute type: auto, int8, int8_float16, float16, float32",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    parser.add_argument(
        "--min-utterance-s",
        type=float,
        default=0.5,
        help="Minimum utterance length (seconds) before transcribing.",
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=200,
        help="Require this much sustained speech (ms) before starting an utterance (reduces noise triggers).",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=2400,
        help="Silence duration (ms) that ends an utterance.",
    )
    parser.add_argument(
        "--preroll-ms",
        type=int,
        default=300,
        help="Audio to keep before speech start (ms) to avoid clipping word starts.",
    )
    parser.add_argument(
        "--speech-rms-mult",
        type=float,
        default=1.2,
        help="Speech threshold = noise_rms * multiplier (adaptive energy VAD).",
    )
    parser.add_argument(
        "--min-speech-rms",
        type=float,
        default=150.0,
        help="Absolute minimum RMS to treat as speech (helps in noisy rooms).",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=2,
        help="Drop transcriptions shorter than this many non-space characters.",
    )
    parser.add_argument(
        "--drop-repetitions",
        action="store_true",
        help="Drop highly repetitive transcriptions (common for background noise).",
    )
    parser.add_argument(
        "--whisper-vad-filter",
        action="store_true",
        help="Enable faster-whisper internal VAD filter (in addition to our VAD).",
    )
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=0.7,
        help="Minimum seconds between sending recognized text",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info (VAD + transcription)",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If CUDA init fails (missing cuDNN/CUDA), fall back to CPU int8 automatically.",
    )
    args = parser.parse_args(argv)

    def _load_model(name: str, *, device: str, compute_type: str) -> WhisperModel:
        try:
            return WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as e:
            # Most common Windows failure: missing CUDA/cuDNN DLLs (e.g. cudnn_ops64_9.dll)
            if device == "cuda" and args.allow_cpu_fallback:
                print(
                    f"WARNING: Failed to initialize CUDA model '{name}' ({e}). Falling back to CPU int8.",
                    file=sys.stderr,
                )
                return WhisperModel(name, device="cpu", compute_type="int8")
            raise

    print(f"Loading Whisper model: {args.model}")
    model = _load_model(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded!")

    wake_model: Optional[WhisperModel] = None
    if args.wake_word:
        wake_model_name = args.wake_model or args.model
        if wake_model_name == args.model:
            wake_model = model
        else:
            print(f"Loading wake-word model: {wake_model_name}")
            wake_model = _load_model(wake_model_name, device=args.device, compute_type=args.compute_type)
            print("Wake-word model loaded!")

    # Connect to robot for sending text back
    print(f"Connecting to robot at {args.robot_ip}...")
    robot_config = SourcceyClientConfig(remote_ip=args.robot_ip, id="sourccey")
    robot = SourcceyClient(robot_config)

    try:
        robot.connect()
        print("✓ Connected to robot!")
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}", file=sys.stderr)
        return 1

    # Set up ZMQ socket to subscribe to audio from robot
    ctx = zmq.Context()
    audio_socket = ctx.socket(zmq.SUB)
    audio_socket.connect(f"tcp://{args.robot_ip}:{args.audio_port}")
    audio_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
    print(f"Subscribed to audio stream from {args.robot_ip}:{args.audio_port}...")

    last_sent = ""
    last_sent_ts = 0.0
    IDENTICAL_TEXT_COOLDOWN = 5.0  # Ignore identical text for 5 seconds

    # Adaptive energy VAD
    noise_rms = 600.0
    noise_alpha = 0.02  # slow update
    chunk_s = 0.2  # best-effort; depends on robot streamer blocksize

    silence_s = 0.0
    utter_s = 0.0
    in_speech = False
    preroll_max_chunks = max(1, int(round((args.preroll_ms / 1000.0) / max(1e-6, chunk_s))))
    preroll: list[np.ndarray] = []
    utter_chunks: list[np.ndarray] = []

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

    def _contains_wake(text: str) -> bool:
        if not args.wake_word:
            return False
        ww = args.wake_word.strip().lower()
        return re.search(rf"\b{re.escape(ww)}\b", text.lower()) is not None

    def _wake_is_clean_hit(wake_text: str) -> bool:
        """Be strict: wake must contain the wake word and be short (usually just 'hello')."""
        if not _contains_wake(wake_text):
            return False
        words = [w for w in re.split(r"\W+", wake_text.lower()) if w]
        if not words:
            return False
        return len(words) <= int(args.wake_max_words)

    def _looks_like_real_command(text: str) -> bool:
        """Reject punctuation/noise like '?' so we arm for next utterance instead of 'sending' junk."""
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return False
        # Must have at least one letter to be considered a command.
        return re.search(r"[A-Za-z]", cleaned) is not None

    def _strip_wake(text: str) -> str:
        if not args.wake_word:
            return text
        ww = args.wake_word.strip().lower()
        pattern = re.compile(rf"\b{re.escape(ww)}\b", re.IGNORECASE)
        m = pattern.search(text)
        if not m:
            return text
        stripped = (text[: m.start()] + " " + text[m.end() :]).strip()
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped

    def _transcribe(audio_i16: np.ndarray, model_obj: WhisperModel, *, beam_size: int, vad_filter: bool) -> str:
        if audio_i16.size == 0:
            return ""
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).copy()
        segments, _info = model_obj.transcribe(
            audio_f32,
            language=args.language,
            beam_size=int(beam_size),
            vad_filter=bool(vad_filter),
        )
        text = " ".join((getattr(seg, "text", "") or "").strip() for seg in segments).strip()
        return text

    def _send_text(text: str) -> None:
        nonlocal last_sent, last_sent_ts
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

        success = robot.send_text(cleaned)
        if success:
            print(f"[RECOGNIZED] {cleaned}")
            last_sent = cleaned
            last_sent_ts = now
        else:
            print(f"[DROP] failed to send: {cleaned}", file=sys.stderr)

    # Wake-word gating state:
    # - waiting_for_wake: look for wake word in each utterance (cheap tiny model)
    # - waiting_for_command: next utterance is treated as command and sent, then back to wake mode
    waiting_for_wake = bool(args.wake_word)
    waiting_for_command = False
    armed_ts = 0.0

    try:
        while True:
            try:
                # Receive audio chunk from robot (non-blocking)
                audio_data = audio_socket.recv(zmq.NOBLOCK)

                # Decode PCM16 chunk
                audio_i16 = np.frombuffer(audio_data, dtype=np.int16).copy()
                if audio_i16.size == 0:
                    continue

                # Estimate chunk duration
                chunk_s = float(audio_i16.size) / float(args.sample_rate)

                rms = float(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))
                if not in_speech:
                    # update noise floor while not in speech
                    noise_rms = (1.0 - noise_alpha) * noise_rms + noise_alpha * rms

                thr = max(float(args.min_speech_rms), float(noise_rms) * float(args.speech_rms_mult))
                is_speech = rms >= thr
                start_s = float(args.start_ms) / 1000.0

                # Maintain preroll ring
                preroll.append(audio_i16)
                if len(preroll) > preroll_max_chunks:
                    preroll.pop(0)

                # Require sustained speech to start (reduces random spikes becoming "speech")
                if not in_speech:
                    if is_speech:
                        utter_s += chunk_s  # reuse as "speech-run" accumulator while idle
                        if utter_s >= start_s:
                            in_speech = True
                            silence_s = 0.0
                            # reset utter length and start collecting
                            utter_s = 0.0
                            utter_chunks = list(preroll)
                            if args.debug:
                                print(
                                    f"[VAD] start (rms={rms:.1f} thr={thr:.1f} noise={noise_rms:.1f})",
                                    file=sys.stderr,
                                )
                    else:
                        # decay accumulator when not speech
                        utter_s = 0.0

                if in_speech:
                    utter_chunks.append(audio_i16)
                    utter_s += chunk_s
                    if is_speech:
                        silence_s = 0.0
                    else:
                        silence_s += chunk_s

                    if silence_s >= (float(args.silence_ms) / 1000.0):
                        # end utterance
                        in_speech = False
                        if utter_s >= float(args.min_utterance_s):
                            utter_audio = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)
                            if args.debug:
                                print(f"[VAD] end utterance (len={utter_s:.2f}s)", file=sys.stderr)

                            if waiting_for_wake:
                                assert wake_model is not None
                                # For wake-word detection, disable faster-whisper's internal VAD.
                                # We already performed energy-based VAD and built an utterance.
                                # The internal VAD can sometimes discard short wake words ("Hello")
                                # and return empty text.
                                wake_text = _transcribe(
                                    utter_audio,
                                    wake_model,
                                    beam_size=int(args.wake_beam_size),
                                    vad_filter=False,
                                )
                                if args.debug and wake_text:
                                    print(f"[WAKE] {wake_text!r}", file=sys.stderr)

                                if _wake_is_clean_hit(wake_text):
                                    # If "Hello <command>" is spoken in one utterance, send immediately.
                                    main_text = _transcribe(
                                        utter_audio,
                                        model,
                                        beam_size=int(args.beam_size),
                                        vad_filter=bool(args.whisper_vad_filter),
                                    )
                                    cmd_text = _strip_wake(main_text)
                                    cmd_text = cmd_text.strip()
                                    if _looks_like_real_command(cmd_text):
                                        _send_text(cmd_text)
                                        waiting_for_command = False
                                        armed_ts = 0.0
                                    else:
                                        waiting_for_command = True
                                        armed_ts = time.time()
                                        if args.debug:
                                            print("[WAKE] armed for next utterance", file=sys.stderr)

                            elif waiting_for_command:
                                if (time.time() - armed_ts) > float(args.wake_timeout_s):
                                    waiting_for_command = False
                                    waiting_for_wake = True
                                    if args.debug:
                                        print("[WAKE] timeout; back to waiting", file=sys.stderr)
                                else:
                                    main_text = _transcribe(
                                        utter_audio,
                                        model,
                                        beam_size=int(args.beam_size),
                                        vad_filter=bool(args.whisper_vad_filter),
                                    )
                                    main_text = _strip_wake(main_text).strip()
                                    if args.debug:
                                        print(f"[CMD] {main_text!r}", file=sys.stderr)
                                    _send_text(main_text)
                                    waiting_for_command = False
                                    waiting_for_wake = True
                                    armed_ts = 0.0

                            else:
                                # No wake-word gating; send every utterance.
                                main_text = _transcribe(
                                    utter_audio,
                                    model,
                                    beam_size=int(args.beam_size),
                                    vad_filter=bool(args.whisper_vad_filter),
                                )
                                _send_text(main_text)
                        utter_chunks = []
                        silence_s = 0.0
                        utter_s = 0.0

            except zmq.Again:
                # No audio available, poll robot for any incoming messages
                robot.poll_text_message()
                time.sleep(0.01)  # Small sleep to avoid busy loop
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                time.sleep(0.1)

    finally:
        audio_socket.close()
        ctx.term()
        robot.disconnect()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

