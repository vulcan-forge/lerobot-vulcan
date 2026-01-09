#!/usr/bin/env python3
"""
Client-side voice recognition for Sourccey (Whisper).

Receives mono PCM16 audio from the robot (ZMQ SUB),
segments speech with adaptive RMS VAD,
transcribes using faster-whisper,
wake-word gates commands,
and sends text back to the robot.

Includes early and repeating "Thinking..." feedback
once the wake word is confirmed, without freezing the main loop.
"""

import argparse
import re
import sys
import time
import threading
from typing import Optional, List

import numpy as np
import zmq
import requests

LLM_URL = "http://localhost:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are Sourccey, a friendly helpful robot. "
    "You speak in short, clear sentences. "
    "You do not ramble. "
    "You keep responses under three sentences. "
    "If you do not know something, say so."
)

_llm_history = [{"role": "system", "content": SYSTEM_PROMPT}]

PHYSICAL_VERBS = {
    "pick up", "lift", "carry", "take out", "wash",
    "drive", "move", "come to", "go to", "grab", "clean",
}

WAKE_WORD = "sourccey"

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
# Text helpers
# -----------------------------
def normalize_robot_terms(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(
        r"\b(sorcy|sorsi|orsi|doris|sourcey|sourcy|isourcing)\b",
        "Sourccey",
        text,
        flags=re.IGNORECASE,
    ).strip()


def has_wake_word(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    prefixes = (
        WAKE_WORD,
        f"{WAKE_WORD},",
        f"hey {WAKE_WORD}",
        f"ok {WAKE_WORD}",
        f"okay {WAKE_WORD}",
    )
    return any(t.startswith(p) for p in prefixes)


def strip_wake_word(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"^(hey|ok|okay)?\s*sourccey[:,]?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()


def sounds_like_physical_action(text: str) -> bool:
    t = (text or "").lower()
    return any(v in t for v in PHYSICAL_VERBS)


def is_similar_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return False
    return len(wa & wb) >= min(len(wa), len(wb)) * 0.5


# -----------------------------
# LLM
# -----------------------------
def ask_llm(user_text: str, timeout_s: float = 15.0) -> str:
    """
    Send user text to the local LLM and get a response.
    Keeps short rolling context to avoid drift.
    """
    global _llm_history

    _llm_history.append({"role": "user", "content": user_text})
    while len(_llm_history) > 7:
        _llm_history.pop(1)

    r = requests.post(
        LLM_URL,
        json={
            "model": "qwen",
            "messages": _llm_history,
            "temperature": 0.3,
            "max_tokens": 80,
        },
        timeout=float(timeout_s),
    )
    r.raise_for_status()

    reply = (r.json()["choices"][0]["message"]["content"] or "").strip()
    if not reply:
        reply = "Sorry — I didn’t get a reply from my brain box."
    _llm_history.append({"role": "assistant", "content": reply})
    return reply


# -----------------------------
# Transcription helper
# -----------------------------
def whisper_transcribe(
    model: WhisperModel,
    audio_i16: np.ndarray,
    language: str,
    beam_size: int,
    vad_filter: bool,
    initial_prompt: str,
) -> str:
    """
    Always returns a concrete string (consumes generator).
    """
    if audio_i16.size == 0:
        return ""
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    segments, _info = model.transcribe(
        audio_f32,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )

    out: List[str] = []
    for seg in segments:
        s = (getattr(seg, "text", "") or "").strip()
        if s:
            out.append(s)
    return " ".join(out).strip()


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
    p.add_argument("--whisper-vad-filter", action="store_true")
    p.add_argument("--allow-cpu-fallback", action="store_true")

    # VAD
    p.add_argument("--min-utterance-s", type=float, default=0.5)
    p.add_argument("--max-utterance-s", type=float, default=5.5)
    p.add_argument("--start-ms", type=int, default=150)
    p.add_argument("--silence-ms", type=int, default=1800)
    p.add_argument("--speech-rms-mult", type=float, default=1.15)
    p.add_argument("--end-mult", type=float, default=0.6)
    p.add_argument("--min-speech-rms", type=float, default=160.0)
    p.add_argument("--noise-alpha", type=float, default=0.02)

    # Rate limiting + mute
    p.add_argument("--min-text-chars", type=int, default=2)
    p.add_argument("--min-interval-s", type=float, default=0.7)
    p.add_argument("--mute-after-send-s", type=float, default=3.0)

    # Thinking system
    p.add_argument("--thinking-text", default="Thinking...")
    p.add_argument("--thinking-interval-s", type=float, default=3.0)
    p.add_argument("--thinking-first-delay-s", type=float, default=0.0)
    p.add_argument("--disable-thinking", action="store_true")

    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    initial_prompt = (
        "This is a conversation with a robot named Sourccey. "
        "Common words include: Sourccey, robot, move, follow, stop, hello, task."
    )

    # -----------------------------
    # Load Whisper
    # -----------------------------
    def load_model() -> WhisperModel:
        try:
            return WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        except Exception as e:
            if args.allow_cpu_fallback:
                print("[WARN] Whisper init failed; falling back to CPU int8", file=sys.stderr)
                return WhisperModel(args.model, device="cpu", compute_type="int8")
            raise

    print(f"Loading Whisper model: {args.model} ({args.device}/{args.compute_type})")
    model = load_model()
    print("Model loaded!")

    # -----------------------------
    # Robot connection
    # -----------------------------
    robot = SourcceyClient(SourcceyClientConfig(remote_ip=args.robot_ip, id="sourccey"))
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
    in_speech = False
    silence_s = 0.0
    utter_s = 0.0
    utter_chunks: List[np.ndarray] = []

    # preroll ~300ms (based on actual chunk duration)
    preroll: List[np.ndarray] = []
    preroll_max_chunks = 6

    # send gating
    last_cmd = ""
    last_cmd_ts = 0.0

    # mic mute handling
    muted_until = 0.0

    # thinking + LLM async
    llm_done = threading.Event()
    llm_reply: dict[str, str] = {"text": ""}
    thinking_active = False
    next_think_ts = 0.0
    first_think_ts = 0.0
    pending_cmd: Optional[str] = None

    def _reset_utterance() -> None:
        nonlocal in_speech, silence_s, utter_s, utter_chunks
        nonlocal preroll
        in_speech = False
        silence_s = 0.0
        utter_s = 0.0
        utter_chunks = []
        preroll = []

    def _safe_send(text: str) -> None:
        # robot.send_text() can throw; never let it kill the loop
        try:
            robot.send_text(text)
        except Exception as e:
            if args.debug:
                print(f"[ROBOT SEND ERROR] {e}", file=sys.stderr)

    def _start_llm(cmd: str) -> None:
        nonlocal thinking_active, next_think_ts, first_think_ts, pending_cmd, muted_until
        nonlocal llm_reply

        pending_cmd = cmd
        llm_reply["text"] = ""
        llm_done.clear()

        # Activate thinking loop (but first think can be delayed)
        thinking_active = (not args.disable_thinking) and bool((args.thinking_text or "").strip())
        now = time.time()
        first_think_ts = now + max(0.0, float(args.thinking_first_delay_s))
        next_think_ts = first_think_ts  # first emission time

        def _worker() -> None:
            try:
                if sounds_like_physical_action(cmd):
                    llm_reply["text"] = "I can explain how to do that, but I can't physically perform it."
                else:
                    llm_reply["text"] = ask_llm(cmd, timeout_s=15.0)
            except Exception as e:
                llm_reply["text"] = "Sorry, I had trouble thinking just now."
                if args.debug:
                    print(f"[LLM ERROR] {e}", file=sys.stderr)
            finally:
                llm_done.set()

        threading.Thread(target=_worker, name="sourccey-llm", daemon=True).start()

        # While thinking/LLM is active, keep mic muted to avoid self-hearing.
        # We'll extend this as we emit "Thinking..." and when we speak the final reply.
        muted_until = max(muted_until, now + 0.25)

    def _maybe_emit_thinking(now: float) -> None:
        nonlocal next_think_ts, muted_until
        if not thinking_active:
            return
        if llm_done.is_set():
            return
        if now < next_think_ts:
            return

        txt = (args.thinking_text or "").strip()
        if not txt:
            return

        _safe_send(txt)
        # mute briefly so the robot doesn't hear itself
        muted_until = max(muted_until, now + 1.2)

        # schedule next
        interval = max(0.5, float(args.thinking_interval_s))
        next_think_ts = now + interval

    # -----------------------------
    # Main loop
    # -----------------------------
    try:
        while True:
            now = time.time()

            # Keep thinking loop alive even while we're not receiving audio.
            _maybe_emit_thinking(now)

            # If LLM finished, speak reply immediately and stop thinking.
            if pending_cmd is not None and llm_done.is_set():
                reply = (llm_reply.get("text") or "").strip()
                if reply:
                    _safe_send(reply)
                    muted_until = max(muted_until, time.time() + float(args.mute_after_send_s))
                    if args.debug:
                        print(f"[USER ] {pending_cmd}")
                        print(f"[ROBOT] {reply}")
                # clear pending
                pending_cmd = None
                thinking_active = False

            # Receive audio
            try:
                data = sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                try:
                    robot.poll_text_message()
                except Exception:
                    pass
                time.sleep(0.005)
                continue

            # If mic muted, discard audio quickly.
            if now < muted_until:
                continue

            audio = np.frombuffer(data, dtype=np.int16)
            if audio.size == 0:
                continue

            chunk_s = audio.size / float(args.sample_rate)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

            if not in_speech:
                noise_rms = (1.0 - args.noise_alpha) * noise_rms + args.noise_alpha * rms

            start_thr = max(args.min_speech_rms, noise_rms * args.speech_rms_mult)
            end_thr = max(args.min_speech_rms * 0.5, noise_rms * args.end_mult)

            is_speech = rms >= (start_thr if not in_speech else end_thr)

            # preroll buffer
            preroll.append(audio)
            if len(preroll) > preroll_max_chunks:
                preroll.pop(0)

            # Speech start detection with hysteresis + start-ms hold
            if not in_speech:
                if is_speech:
                    utter_s += chunk_s
                    if utter_s >= args.start_ms / 1000.0:
                        in_speech = True
                        utter_chunks = list(preroll)
                        silence_s = 0.0
                        utter_s = 0.0
                else:
                    utter_s = 0.0
                continue

            # In speech: accumulate
            utter_chunks.append(audio)
            utter_s += chunk_s

            if is_speech:
                silence_s = 0.0
            else:
                silence_s += chunk_s
                # let noise track a bit during trailing silence
                noise_rms = (1.0 - args.noise_alpha) * noise_rms + args.noise_alpha * rms

            # End of utterance
            if (silence_s >= args.silence_ms / 1000.0) or (utter_s >= args.max_utterance_s):
                full = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)
                _reset_utterance()

                if full.size == 0:
                    continue
                full_dur = full.size / float(args.sample_rate)
                if full_dur < args.min_utterance_s:
                    continue

                # Transcribe full utterance
                try:
                    text = whisper_transcribe(
                        model,
                        full,
                        language=args.language,
                        beam_size=args.beam_size,
                        vad_filter=args.whisper_vad_filter,
                        initial_prompt=initial_prompt,
                    )
                except Exception as e:
                    if args.debug:
                        print(f"[WHISPER ERROR] {e}", file=sys.stderr)
                    continue

                text = normalize_robot_terms(text)
                if args.debug:
                    print(f"[HEARD] {text}")

                if not has_wake_word(text):
                    if args.debug:
                        print(f"[IGNORED] No wake word: {text}")
                    continue

                cmd = strip_wake_word(text)
                cmd = cmd.strip()
                if not cmd:
                    continue

                # Rate limit + repetition drop
                now2 = time.time()
                if len(cmd.replace(" ", "")) < args.min_text_chars:
                    continue
                if (now2 - last_cmd_ts) < float(args.min_interval_s):
                    continue
                if is_similar_text(cmd, last_cmd):
                    continue

                # Start async LLM + thinking loop
                last_cmd = cmd
                last_cmd_ts = now2
                _start_llm(cmd)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
