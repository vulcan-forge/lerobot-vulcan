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
    recognizer = KaldiRecognizer(model, args.sample_rate)
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
                        continue

                    now = time.time()
                    time_since_last = now - last_sent_ts

                    # Debounce: ignore if same text sent recently
                    if text == last_sent and time_since_last < 3.0:
                        recognizer.Reset()
                        continue

                    # General debounce
                    if time_since_last < args.min_interval_s:
                        recognizer.Reset()
                        continue

                    # Send recognized text back to robot host
                    success = robot.send_text(text)
                    if success:
                        print(f"[RECOGNIZED] {text}")
                        last_sent = text
                        last_sent_ts = now
                        recognizer.Reset()
                else:
                    # Check partial results to keep recognizer active
                    partial = json.loads(recognizer.PartialResult())
                    # partial_text = partial.get("partial", "")

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

