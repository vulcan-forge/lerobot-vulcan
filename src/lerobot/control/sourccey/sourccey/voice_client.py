#!/usr/bin/env python3
"""
Client-side voice recognition for Sourccey.

Receives raw audio from the robot, processes it with Vosk (large model),
and sends recognized text back to the robot host.

Run (on Windows/client machine):
  python voice_client.py --robot_ip <ROBOT_IP>
  python voice_client.py --robot_ip 192.168.1.100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import zmq

try:
    from vosk import KaldiRecognizer, Model
except Exception as e:
    raise RuntimeError("Missing vosk. Install: pip install vosk") from e

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient


def _default_model_path() -> Path:
    env = os.environ.get("VOSK_MODEL_PATH")
    if env:
        env_path = Path(env).expanduser().resolve()
        if env_path.exists():
            return env_path
    
    # Default location in the project: src/lerobot/model/vosk-model-en-us-0.42-gigaspeech
    # From voice_client.py at src/lerobot/control/sourccey/sourccey/, go up to src/lerobot/
    # Use resolve() to get absolute path
    script_file = Path(__file__).resolve()
    default_path = script_file.parent.parent.parent.parent / "model" / "vosk-model-en-us-0.42-gigaspeech"
    if default_path.exists():
        return default_path
    
    # Common locations on Windows
    for base in [Path.home() / ".cache" / "vosk", Path("C:/vosk")]:
        model_path = base / "vosk-model-en-us-0.42-gigaspeech"
        if model_path.exists():
            return model_path.resolve()
    
    # Fallback to 0.22 if 0.42 not found
    fallback = script_file.parent.parent.parent.parent / "model" / "vosk-model-en-us-0.22"
    if fallback.exists():
        return fallback
    
    return Path.home() / ".cache" / "vosk" / "vosk-model-en-us-0.22"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Client-side voice recognition for Sourccey")
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
        "--model-path",
        type=str,
        default=None,
        help="Path to Vosk model directory (default: auto-detect from project or ~/.cache/vosk)",
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default=None,
        help='Optional JSON list of phrases/words to constrain recognition (e.g. \'["hello sourccey","stop"]\')',
    )
    parser.add_argument(
        "--min-avg-conf",
        type=float,
        default=0.55,
        help="Minimum average word confidence required to send text to robot (0.0-1.0).",
    )
    parser.add_argument(
        "--max-alternatives",
        type=int,
        default=3,
        help="Ask Vosk for N alternatives (for debugging / better decisions).",
    )
    parser.add_argument(
        "--debug-alternatives",
        action="store_true",
        help="Print Vosk alternatives/confidence info to stderr.",
    )
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=0.7,
        help="Minimum seconds between sending recognized text",
    )
    args = parser.parse_args(argv)

    # Use default model path if not specified
    if args.model_path is None:
        model_path = _default_model_path()
    else:
        model_path = Path(args.model_path).expanduser().resolve()
    
    # Load Vosk model
    if not model_path.exists():
        print(f"ERROR: Vosk model not found at {model_path}", file=sys.stderr)
        print(f"Expected location: {Path(__file__).resolve().parent.parent.parent.parent / 'model' / 'vosk-model-en-us-0.42-gigaspeech'}", file=sys.stderr)
        print(f"Download from: https://alphacephei.com/vosk/models", file=sys.stderr)
        return 1

    print(f"Loading Vosk model: {model_path}")
    model = Model(str(model_path))
    if args.grammar:
        try:
            grammar_list = json.loads(args.grammar)
            if not isinstance(grammar_list, list):
                raise ValueError("grammar must be a JSON list")
            recognizer = KaldiRecognizer(model, args.sample_rate, json.dumps(grammar_list))
            print(f"Using grammar: {grammar_list}")
        except Exception as e:
            print(f"WARNING: invalid --grammar, falling back to free-form STT: {e}", file=sys.stderr)
            recognizer = KaldiRecognizer(model, args.sample_rate)
    else:
        recognizer = KaldiRecognizer(model, args.sample_rate)

    # Ask Vosk to include per-word confidences where possible
    try:
        recognizer.SetWords(True)
    except Exception:
        pass
    try:
        recognizer.SetMaxAlternatives(int(args.max_alternatives))
    except Exception:
        pass
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
    silence_count = 0
    SILENCE_THRESHOLD = 50  # Reset recognizer after this many empty results

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

    def avg_confidence(result_obj: dict) -> Optional[float]:
        words = result_obj.get("result")
        if not isinstance(words, list) or not words:
            return None
        confs = [w.get("conf") for w in words if isinstance(w, dict) and isinstance(w.get("conf"), (int, float))]
        if not confs:
            return None
        return float(sum(confs) / len(confs))

    try:
        while True:
            try:
                # Receive audio chunk from robot (non-blocking)
                audio_data = audio_socket.recv(zmq.NOBLOCK)

                # Process with Vosk
                if recognizer.AcceptWaveform(audio_data):
                    result = json.loads(recognizer.Result())
                    text = (result.get("text", "") or "").strip()

                    if not text:
                        silence_count += 1
                        # Reset recognizer after prolonged silence
                        if silence_count >= SILENCE_THRESHOLD:
                            recognizer.Reset()
                            silence_count = 0
                        continue

                    silence_count = 0  # Reset silence counter on valid text
                    now = time.time()
                    time_since_last = now - last_sent_ts

                    conf = avg_confidence(result)
                    if conf is not None and conf < max(0.0, float(args.min_avg_conf)):
                        if args.debug_alternatives:
                            print(f"[DROP] Low confidence {conf:.2f}: {text}", file=sys.stderr)
                            if "alternatives" in result:
                                print(f"[DEBUG] alternatives={result.get('alternatives')}", file=sys.stderr)
                        recognizer.Reset()
                        continue

                    # Strong debounce: ignore if same text sent recently
                    if text == last_sent and time_since_last < IDENTICAL_TEXT_COOLDOWN:
                        print(f"[DROP] Duplicate text (cooldown): {text}", file=sys.stderr)
                        recognizer.Reset()
                        continue

                    # Check for similar text (fuzzy matching)
                    if is_similar_text(text, last_sent) and time_since_last < IDENTICAL_TEXT_COOLDOWN:
                        print(f"[DROP] Similar text (cooldown): {text}", file=sys.stderr)
                        recognizer.Reset()
                        continue

                    # General debounce: minimum interval between any sends
                    if time_since_last < args.min_interval_s:
                        print(f"[DROP] Too soon (min interval): {text}", file=sys.stderr)
                        recognizer.Reset()
                        continue

                    # Send recognized text back to robot host
                    success = robot.send_text(text)
                    if success:
                        print(f"[RECOGNIZED] {text}")
                        last_sent = text
                        last_sent_ts = now
                        # Reset recognizer after sending to prevent echo
                        recognizer.Reset()
                    else:
                        print(f"[DROP] Failed to send: {text}", file=sys.stderr)
                else:
                    # Check partial results to keep recognizer active
                    if args.debug_alternatives:
                        partial = json.loads(recognizer.PartialResult())
                        partial_text = (partial.get("partial", "") or "").strip()
                        if partial_text:
                            print(f"[PARTIAL] {partial_text}", file=sys.stderr)

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

