"""
Voice listener for Sourccey (runs on the robot).

Listens to microphone audio using Vosk, and sends recognized text to the local
Sourccey host process via ZMQ text channel (host's `port_zmq_text_in`).

Run (on robot):
  uv run python -m lerobot.robots.sourccey.sourccey.sourccey.voice_listener

Notes:
- Requires `vosk` and `sounddevice` to be installed in the robot python env.
- Requires a Vosk model directory (default can be overridden via args/env).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from pathlib import Path
from typing import Optional

import zmq

try:
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing voice dependencies. Install `vosk` and `sounddevice` in the robot environment."
    ) from e

from .config_sourccey import SourcceyHostConfig


def _default_model_path() -> Path:
    # Allow override from env (useful on the robot)
    env = os.environ.get("VOSK_MODEL_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "vosk" / "vosk-model-small-en-us-0.15"


class VoiceToHostSender:
    def __init__(self, host_ip: str, port: int):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(f"tcp://{host_ip}:{port}")

    def send(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        try:
            self._sock.send_string(text, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def close(self):
        try:
            self._sock.close()
        finally:
            self._ctx.term()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sourccey voice listener (speech -> host text channel)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index (optional)")
    parser.add_argument("--host-ip", type=str, default="127.0.0.1", help="Host IP to send text to (default: localhost)")
    parser.add_argument(
        "--host-text-in-port",
        type=int,
        default=SourcceyHostConfig().port_zmq_text_in,
        help="Host ZMQ text_in port (default from SourcceyHostConfig)",
    )
    parser.add_argument("--model-path", type=str, default=str(_default_model_path()))
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=0.7,
        help="Minimum seconds between sends (debounce) to reduce spam.",
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        print(f"[voice_listener] ERROR: Vosk model not found at: {model_path}", file=sys.stderr)
        return 1

    host_cfg = SourcceyHostConfig()
    sender = VoiceToHostSender(args.host_ip, args.host_text_in_port or host_cfg.port_zmq_text_in)

    audio_q: "queue.Queue[bytes]" = queue.Queue()

    def audio_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[voice_listener] Audio status: {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    print(f"[voice_listener] Loading Vosk model: {model_path}")
    model = Model(str(model_path))
    recognizer = KaldiRecognizer(model, args.sample_rate)

    print(f"[voice_listener] Sending text to tcp://{args.host_ip}:{args.host_text_in_port}")
    print("[voice_listener] Listening...")

    last_sent = ""
    last_sent_ts = 0.0
    silence_count = 0
    SILENCE_THRESHOLD = 50  # Reset recognizer after this many empty results
    IDENTICAL_TEXT_COOLDOWN = 3.0  # Ignore identical text for this many seconds
    recent_words = []  # Track recent words to detect partial recognition bursts
    RECENT_WORDS_WINDOW = 1.5  # Seconds to track recent words
    MAX_RECENT_WORDS = 3  # If we see this many different words in the window, it's likely partial recognition

    try:
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=args.channels,
            callback=audio_callback,
            device=args.device,
        ):
            while True:
                data = audio_q.get()
                
                # Check for final results
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = (result.get("text", "") or "").strip()
                    
                    if not text:
                        silence_count += 1
                        # Reset recognizer after prolonged silence to clear stale state
                        if silence_count >= SILENCE_THRESHOLD:
                            recognizer.Reset()
                            silence_count = 0
                        continue
                    
                    # Reset silence counter on valid text
                    silence_count = 0
                    
                    now = time.time()
                    time_since_last = now - last_sent_ts
                    
                    # Clean up old entries from recent_words
                    recent_words = [(word, ts) for word, ts in recent_words if (now - ts) < RECENT_WORDS_WINDOW]
                    
                    # Strong debounce for identical text - ignore for longer period
                    if text == last_sent:
                        if time_since_last < IDENTICAL_TEXT_COOLDOWN:
                            # Reset recognizer immediately to stop processing same audio
                            recognizer.Reset()
                            # Drain a few audio chunks to clear buffer
                            for _ in range(3):
                                try:
                                    audio_q.get_nowait()
                                except queue.Empty:
                                    break
                            continue
                    
                    # Detect partial recognition bursts: if we've seen many different words recently,
                    # this is likely partial recognition and we should wait for the final result
                    unique_recent = len(set(word for word, _ in recent_words))
                    if unique_recent >= MAX_RECENT_WORDS and time_since_last < RECENT_WORDS_WINDOW:
                        # Too many different words recently - likely partial recognition
                        recognizer.Reset()
                        continue
                    
                    # General debounce: ignore any text sent very recently (increased from 0.3 to 0.8)
                    if time_since_last < 0.8:
                        recognizer.Reset()
                        continue
                    
                    ok = sender.send(text)
                    if ok:
                        print(f"[voice_listener] SENT: {text}")
                        last_sent = text
                        last_sent_ts = now
                        # Track this word in recent words
                        recent_words.append((text, now))
                        # Reset recognizer after sending to prevent repeated recognition
                        recognizer.Reset()
                        # Drain audio queue to clear any buffered audio
                        while True:
                            try:
                                audio_q.get_nowait()
                            except queue.Empty:
                                break
                    else:
                        print(f"[voice_listener] DROP (socket busy): {text}", file=sys.stderr)
                else:
                    # Check partial results to clear them and avoid stale final results
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = (partial.get("partial", "") or "").strip()
                    # If we have a partial result, reset silence counter
                    if partial_text:
                        silence_count = 0
    except KeyboardInterrupt:
        print("[voice_listener] Stopping...")
        return 0
    finally:
        sender.close()


if __name__ == "__main__":
    raise SystemExit(main())


