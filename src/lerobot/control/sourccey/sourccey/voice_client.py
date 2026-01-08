#!/usr/bin/env python3
"""
Low-latency voice client for Sourccey.

Key features (fixed + hardened):
- Adaptive RMS VAD with hysteresis (start/end thresholds) + noise floor tracking
- Optional early intent detection using tiny.en on partial audio (throttled)
- Full transcription fallback using main Whisper model when utterance ends
- LLM call with short context + robust error handling
- Never silently dies on exceptions; keeps running
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

LLM_URL_DEFAULT = "http://localhost:8080/v1/chat/completions"

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

WAKE_WORD_DEFAULT = "sourccey"

# -----------------------------
# Windows CUDA DLL helper
# -----------------------------
def _configure_windows_cuda_dll_search() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import importlib.util
        import os

        def _add(mod: str) -> None:
            spec = importlib.util.find_spec(mod)
            if not spec or not spec.submodule_search_locations:
                return
            p = spec.submodule_search_locations[0]
            b = os.path.join(p, "bin")
            if os.path.isdir(b):
                try:
                    os.add_dll_directory(b)
                except Exception:
                    pass
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")

        for m in ("nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_runtime"):
            _add(m)
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
    replacements = {
        r"\b(sorcy|sorsi|orsi|doris|sourcey|sourcy|isourcing)\b": "Sourccey",
    }
    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text.strip()


def has_wake_word(text: str, wake_word: str) -> bool:
    if not text:
        return False
    t = text.lower().lstrip(" ,.!?…uhum")
    w = wake_word.lower()
    return (
        t.startswith(w)
        or t.startswith(f"hey {w}")
        or t.startswith(f"ok {w}")
        or t.startswith(f"okay {w}")
    )

def strip_wake_word(text: str, wake_word: str) -> str:
    if not text:
        return ""
    w = re.escape(wake_word)
    return re.sub(
        rf"^(hey|ok|okay)?\s*{w}[:,]?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()


def sounds_like_physical_action(text: str) -> bool:
    t = (text or "").lower()
    return any(v in t for v in PHYSICAL_VERBS)


def intent_confident(text: str) -> bool:
    """
    Heuristic: early-commit only if it's clearly a question/command.
    Keep this conservative to avoid misfires.
    """
    t = (text or "").lower().strip()
    return any(
        t.startswith(p)
        for p in (
            "can you",
            "could you",
            "do you",
            "what is",
            "what's",
            "why",
            "how do",
            "how to",
            "should i",
            "tell me",
            "help me",
        )
    )


def is_similar_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return False
    return len(wa & wb) >= min(len(wa), len(wb)) * 0.6


# -----------------------------
# LLM
# -----------------------------
def ask_llm_fast(user_text: str, llm_url: str, llm_model: str, max_tokens: int, temperature: float) -> str:
    """
    Request a short reply; return first sentence to reduce perceived latency.
    Robust to server-down situations.
    """
    global _llm_history

    _llm_history.append({"role": "user", "content": user_text})
    # Keep context small (system + last 3 turns)
    while len(_llm_history) > 7:
        _llm_history.pop(1)

    try:
        r = requests.post(
            llm_url,
            json={
                "model": llm_model,
                "messages": _llm_history,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            },
            timeout=10,
        )
        r.raise_for_status()
        full = (r.json()["choices"][0]["message"]["content"] or "").strip()
        if not full:
            raise RuntimeError("Empty LLM response")
    except Exception as e:
        return "Sorry — my brain box isn’t responding right now."

    # first "sentence" (very naive, but fast)
    first = full.split("\n")[0].strip()
    if "." in first:
        first = first.split(".", 1)[0].strip()
    if not first:
        first = "Okay."

    _llm_history.append({"role": "assistant", "content": first})
    return first


# -----------------------------
# Transcription
# -----------------------------
def _whisper_to_text(model: WhisperModel, audio_i16: np.ndarray, language: str, beam_size: int, vad_filter: bool, initial_prompt: str) -> str:
    """
    Always returns a real string. Never returns a generator.
    """
    if audio_i16.size == 0:
        return ""
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio_f32,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )

    texts: List[str] = []
    for seg in segments:
        try:
            s = (seg.text or "").strip()
        except Exception:
            s = ""
        if s:
            texts.append(s)
    return " ".join(texts).strip()


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()

    # Robot/audio
    p.add_argument("--robot_ip", required=True)
    p.add_argument("--audio-port", type=int, default=5559)
    p.add_argument("--sample-rate", type=int, default=16000)

    # Models
    p.add_argument("--model", default="small.en")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--beam-size", type=int, default=2)
    p.add_argument("--language", default="en")
    p.add_argument("--whisper-vad-filter", action="store_true")

    # Wake/early commit
    p.add_argument("--wake-word", default=WAKE_WORD_DEFAULT)
    p.add_argument("--enable-early-intent", action="store_true")
    p.add_argument("--scout-model", default="tiny.en")
    p.add_argument("--scout-beam-size", type=int, default=1)
    p.add_argument("--early-commit-min-s", type=float, default=0.8)
    p.add_argument("--early-commit-max-s", type=float, default=1.6)
    p.add_argument("--early-check-interval-ms", type=int, default=200)

    # VAD params
    p.add_argument("--start-ms", type=int, default=80)
    p.add_argument("--silence-ms", type=int, default=500)
    p.add_argument("--max-utterance-s", type=float, default=3.5)
    p.add_argument("--min-utterance-s", type=float, default=0.35)

    p.add_argument("--speech-rms-mult", type=float, default=1.6)
    p.add_argument("--end-mult", type=float, default=0.7, help="End threshold multiplier vs noise (hysteresis)")
    p.add_argument("--min-speech-rms", type=float, default=120.0)
    p.add_argument("--noise-alpha", type=float, default=0.02)

    # Rate limiting
    p.add_argument("--min-interval-s", type=float, default=0.25)
    p.add_argument("--mute-after-send-s", type=float, default=1.2)
    p.add_argument("--min-text-chars", type=int, default=2)

    # LLM
    p.add_argument("--llm-url", default=LLM_URL_DEFAULT)
    p.add_argument("--llm-model", default="qwen")
    p.add_argument("--llm-max-tokens", type=int, default=80)
    p.add_argument("--llm-temp", type=float, default=0.3)

    # "Thinking..." chatter while waiting on LLM
    p.add_argument("--thinking-text", default="Thinking...Thinking...")
    p.add_argument("--thinking-interval-s", type=float, default=2.0)
    p.add_argument("--disable-thinking", action="store_true", help="Disable periodic 'Thinking...' speech while waiting for LLM.")

    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    # ---- Load models
    print(f"Loading main Whisper model: {args.model} ({args.device}/{args.compute_type})")
    main_model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    scout_model = None
    if args.enable_early_intent:
        print(f"Loading intent scout model: {args.scout_model} (cpu/int8)")
        scout_model = WhisperModel(args.scout_model, device="cpu", compute_type="int8")

    print("Models ready")

    # ---- Connect robot
    robot = SourcceyClient(SourcceyClientConfig(remote_ip=args.robot_ip, id="sourccey"))
    robot.connect()
    print("✓ Connected to robot")

    # ---- ZMQ audio
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.robot_ip}:{args.audio_port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")

    # ---- VAD state
    noise_rms = 600.0
    in_speech = False
    silence_s = 0.0
    utter_s = 0.0
    utter_chunks: List[np.ndarray] = []

    # preroll ~300ms so we don't chop leading phonemes
    preroll: List[np.ndarray] = []
    # estimate chunk duration dynamically, but cap buffer length to a few chunks
    preroll_max_chunks = 6

    last_sent_ts = 0.0
    muted_until = 0.0
    last_user_text = ""
    is_robot_speaking = False
    early_committed = False
    wake_acknowledged = False

    last_early_probe_ts = 0.0
    early_check_interval_s = max(0.05, args.early_check_interval_ms / 1000.0)

    initial_prompt = (
        "This is a conversation with a robot named Sourccey. "
        "Common words include: Sourccey, robot, move, follow, stop, hello, task."
    )

    def _reset_utterance():
        nonlocal in_speech
        nonlocal silence_s
        nonlocal utter_s
        nonlocal utter_chunks
        nonlocal preroll
        nonlocal last_early_probe_ts
        nonlocal early_committed
        nonlocal wake_acknowledged

        in_speech = False
        silence_s = 0.0
        utter_s = 0.0
        utter_chunks = []
        preroll = []
        last_early_probe_ts = 0.0
        early_committed = False
        wake_acknowledged = False

    def _say_thinking_immediately():
        """Say 'Thinking...' immediately on wake-word detection."""
        nonlocal muted_until, is_robot_speaking

        if args.disable_thinking:
            return

        txt = (args.thinking_text or "").strip()
        if not txt:
            return

        is_robot_speaking = True
        now = time.time()

        # Extend mute window so we don't listen during this short utterance
        muted_until = max(muted_until, now + 1.2)

        try:
            robot.send_text(txt)
        except Exception:
            pass

    def _say_thinking_once() -> None:
        """
        Legacy hook: thinking speech is now handled entirely inside `_send_reply()`.
        Keep this as a no-op so call sites don't need to change.
        """
        return
    
    def _send_reply(cmd_text: str, *, force: bool = False):
        nonlocal last_sent_ts, muted_until, last_user_text, is_robot_speaking

        now = time.time()

        if not force:
            if now - last_sent_ts < args.min_interval_s:
                return

        if not cmd_text:
            return

        cmd_text = cmd_text.strip()
        if len(cmd_text.replace(" ", "")) < args.min_text_chars:
            return

        if is_similar_text(cmd_text, last_user_text):
            return

        if sounds_like_physical_action(cmd_text):
            reply = "I can explain how to do that, but I can't physically perform it."
        else:
            # ZMQ sockets are not thread-safe; keep `robot.send_text(...)` on this thread.
            # Instead, run the LLM call in a worker thread and "tick" thinking speech here.
            done = threading.Event()
            llm_out: dict[str, str] = {"reply": ""}

            def _llm_worker() -> None:
                try:
                    llm_out["reply"] = ask_llm_fast(
                        cmd_text,
                        llm_url=args.llm_url,
                        llm_model=args.llm_model,
                        max_tokens=args.llm_max_tokens,
                        temperature=args.llm_temp,
                    )
                finally:
                    done.set()

            t = threading.Thread(target=_llm_worker, name="sourccey-llm", daemon=True)
            t.start()

            # Never allow faster than 2s (matches expected UX; avoids accidental spam).
            interval = max(2.0, float(args.thinking_interval_s))
            thinking_text = (args.thinking_text or "").strip()
            enable_thinking = (not args.disable_thinking) and bool(thinking_text)

            # Keep mic muted while we audibly "think"
            is_robot_speaking = True

            # First "Thinking..." is emitted at wake-word detection time; only repeat here.
            next_think_t = time.monotonic() + (interval * 3.0)
            while not done.is_set():
                now2_m = time.monotonic()

                if enable_thinking and now2_m >= next_think_t:
                    # Extend mute window so the main loop doesn't resume listening mid-thinking.
                    now_wall = time.time()
                    muted_until = max(muted_until, now_wall + interval + 0.75)
                    try:
                        robot.send_text(thinking_text)
                    except Exception:
                        pass
                    next_think_t = now2_m + interval

                # Short wait so we stop promptly when the LLM finishes.
                done.wait(timeout=0.05)

            reply = llm_out.get("reply", "")

        if not reply:
            return

        # ---- HARD MUTE MIC WHILE SPEAKING ----
        is_robot_speaking = True
        robot.send_text(reply)

        if args.debug:
            print(f"[USER ] {cmd_text}")
            print(f"[ROBOT] {reply}")

        last_user_text = cmd_text
        last_sent_ts = now
        muted_until = now + args.mute_after_send_s

    try:
        while True:
            try:
                data = sock.recv(zmq.NOBLOCK)
                now = time.time()

                # HARD MIC MUTE while robot is speaking
                if is_robot_speaking:
                    if now >= muted_until:
                        is_robot_speaking = False
                    else:
                        continue

                audio = np.frombuffer(data, dtype=np.int16)
                if audio.size == 0:
                    continue

                chunk_s = audio.size / args.sample_rate

                # RMS for VAD
                rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

                # Update noise floor when NOT in speech
                if not in_speech:
                    noise_rms = (1.0 - args.noise_alpha) * noise_rms + args.noise_alpha * rms

                # Hysteresis thresholds
                start_thr = max(args.min_speech_rms, noise_rms * args.speech_rms_mult)
                end_thr = max(args.min_speech_rms * 0.5, noise_rms * args.end_mult)

                is_speech = rms >= (start_thr if not in_speech else end_thr)

                # Maintain preroll buffer
                preroll.append(audio)
                if len(preroll) > preroll_max_chunks:
                    preroll.pop(0)

                # Transition into speech only after start_ms of sustained speech
                if not in_speech:
                    if is_speech:
                        utter_s += chunk_s
                        if utter_s >= args.start_ms / 1000.0:
                            in_speech = True
                            utter_chunks = list(preroll)
                            silence_s = 0.0
                            utter_s = 0.0
                            wake_acknowledged = False
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

                # ---- EARLY INTENT PROBE (throttled)
                # ---- WAKE WORD ACK (FAST PATH)
                if (
                    scout_model is not None
                    and in_speech
                    and not wake_acknowledged
                    and utter_s >= 0.2  # ~200 ms
                ):
                    try:
                        probe = np.concatenate(utter_chunks)
                        txt = _whisper_to_text(
                            scout_model,
                            probe,
                            language=args.language,
                            beam_size=1,
                            vad_filter=False,
                            initial_prompt=initial_prompt,
                        )
                        txt = normalize_robot_terms(txt)

                        if args.debug and txt:
                            print(f"[WAKE ] {txt}")

                        if has_wake_word(txt, args.wake_word):
                            wake_acknowledged = True
                            _say_thinking_immediately()

                    except Exception:
                        pass

                # ---- END OF UTTERANCE: silence or max length
                if (silence_s >= args.silence_ms / 1000.0) or (utter_s >= args.max_utterance_s):

                    # If we already early-committed, skip full ASR entirely
                    if early_committed:
                        _reset_utterance()
                        continue

                    full = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)
                    _reset_utterance()

                    if full.size == 0:
                        continue

                    full_dur = full.size / args.sample_rate
                    if full_dur < args.min_utterance_s:
                        continue

                    try:
                        txt = _whisper_to_text(
                            main_model,
                            full,
                            language=args.language,
                            beam_size=args.beam_size,
                            vad_filter=args.whisper_vad_filter,
                            initial_prompt=initial_prompt,
                        )
                        txt = normalize_robot_terms(txt)

                        if args.debug:
                            print(f"[HEARD] {txt}")

                        if not has_wake_word(txt, args.wake_word):
                            if args.debug:
                                print(f"[IGNORED] No wake word: {txt}")
                            continue

                        # Wake word confirmed → immediate feedback
                        _say_thinking_immediately()

                        cmd = strip_wake_word(txt, args.wake_word).strip()
                        if not cmd:
                            continue

                        _send_reply(cmd, force=True)

                    except Exception as e:
                        print(f"[VOICE ERROR] {e}", file=sys.stderr)
                        time.sleep(0.05)

            except zmq.Again:
                # keep the robot client responsive if it needs polling
                try:
                    robot.poll_text_message()
                except Exception:
                    pass
                time.sleep(0.005)

            except Exception as e:
                # Never die silently
                print(f"[LOOP ERROR] {e}", file=sys.stderr)
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[voice] Stopping...")
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
