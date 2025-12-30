#!/usr/bin/env python3
"""
Client-side voice recognition for Sourccey (Whisper).

Receives raw audio from the robot (ZMQ SUB), transcribes it locally using
`faster-whisper`, and sends recognized text back to the robot host.

Run (on Windows/client machine):
  uv run python -m lerobot.control.sourccey.sourccey.voice_client --robot_ip <ROBOT_IP>
"""

import argparse
import sys
import time
from typing import Optional

import numpy as np
import zmq

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
        "--silence-ms",
        type=int,
        default=700,
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
        default=1.6,
        help="Speech threshold = noise_rms * multiplier (adaptive energy VAD).",
    )
    parser.add_argument(
        "--min-speech-rms",
        type=float,
        default=900.0,
        help="Absolute minimum RMS to treat as speech (helps in noisy rooms).",
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default=None,
        help="Optional wake word/phrase. If set, only sends text containing it.",
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
    args = parser.parse_args(argv)

    print(f"Loading Whisper model: {args.model}")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded!")

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

    def transcribe_and_send(audio_i16: np.ndarray) -> None:
        nonlocal last_sent, last_sent_ts
        if audio_i16.size == 0:
            return
        # Convert int16 PCM -> float32 [-1, 1]
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).copy()
        segments, _info = model.transcribe(
            audio_f32,
            language=args.language,
            beam_size=int(args.beam_size),
            vad_filter=True,
        )
        text = " ".join((seg.text or "").strip() for seg in segments).strip()
        if not text:
            return

        if args.wake_word and (args.wake_word.lower() not in text.lower()):
            if args.debug:
                print(f"[DROP] no wake word: {text}", file=sys.stderr)
            return

        now = time.time()
        time_since_last = now - last_sent_ts
        if text == last_sent and time_since_last < IDENTICAL_TEXT_COOLDOWN:
            if args.debug:
                print(f"[DROP] duplicate cooldown: {text}", file=sys.stderr)
            return
        if is_similar_text(text, last_sent) and time_since_last < IDENTICAL_TEXT_COOLDOWN:
            if args.debug:
                print(f"[DROP] similar cooldown: {text}", file=sys.stderr)
            return
        if time_since_last < args.min_interval_s:
            if args.debug:
                print(f"[DROP] min interval: {text}", file=sys.stderr)
            return

        success = robot.send_text(text)
        if success:
            print(f"[RECOGNIZED] {text}")
            last_sent = text
            last_sent_ts = now
        else:
            print(f"[DROP] failed to send: {text}", file=sys.stderr)

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

                # Maintain preroll ring
                preroll.append(audio_i16)
                if len(preroll) > preroll_max_chunks:
                    preroll.pop(0)

                if not in_speech and is_speech:
                    in_speech = True
                    silence_s = 0.0
                    utter_s = 0.0
                    utter_chunks = list(preroll)
                    if args.debug:
                        print(f"[VAD] start (rms={rms:.1f} thr={thr:.1f} noise={noise_rms:.1f})", file=sys.stderr)

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
                            transcribe_and_send(utter_audio)
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

