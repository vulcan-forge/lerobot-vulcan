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

def reset_llm_history() -> None:
    """Reset rolling LLM context so each interaction can be truly stateless if desired."""
    global _llm_history
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
        # Separate connect/read timeouts behave better with local servers.
        timeout=(5.0, float(timeout_s)),
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
    p.add_argument("--min-utterance-s", type=float, default=0.5)
    p.add_argument("--max-utterance-s", type=float, default=5.5)
    p.add_argument("--start-ms", type=int, default=150)
    p.add_argument("--silence-ms", type=int, default=1800)

    p.add_argument("--speech-rms-mult", type=float, default=1.15)
    p.add_argument("--end-mult", type=float, default=0.6)
    p.add_argument("--min-speech-rms", type=float, default=160.0)
    # Helps prevent false triggers / hallucinated wake probes in quiet environments.
    p.add_argument("--noise-rms-floor", type=float, default=300.0)

    p.add_argument("--min-text-chars", type=int, default=2)
    p.add_argument("--min-interval-s", type=float, default=0.7)
    p.add_argument("--mute-after-send-s", type=float, default=3.0)

    # 🔔 THINKING CONFIG
    p.add_argument("--thinking-text", default="Thinking…")
    p.add_argument("--thinking-interval-s", type=float, default=3.0)
    p.add_argument("--disable-thinking", action="store_true")

    # Wake-probe tuning
    p.add_argument("--wake-probe-min-s", type=float, default=0.25)
    p.add_argument("--wake-probe-interval-ms", type=int, default=250)
    p.add_argument("--wake-probe-max-per-utterance", type=int, default=12)
    p.add_argument("--wake-probe-min-rms-mult", type=float, default=1.15)
    p.add_argument("--wake-probe-print-all", action="store_true")

    p.add_argument("--whisper-vad-filter", action="store_true")
    p.add_argument("--allow-cpu-fallback", action="store_true")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args(argv)

    # -----------------------------
    # Whisper model
    # -----------------------------
    def load_model() -> WhisperModel:
        try:
            return WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        except Exception:
            if not args.allow_cpu_fallback:
                raise
            print("[WARN] Failed to init requested device; falling back to CPU int8", file=sys.stderr)
            return WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"Loading Whisper model: {args.model} ({args.device}/{args.compute_type})")
    model = load_model()
    print("Model loaded!")

    # -----------------------------
    # Robot connection
    # -----------------------------
    robot = SourcceyClient(SourcceyClientConfig(remote_ip=args.robot_ip, id="sourccey"))
    robot.connect()
    print("✓ Connected to robot")

    # -----------------------------
    # ZMQ audio
    # -----------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.robot_ip}:{args.audio_port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")

    # -----------------------------
    # VAD + buffers
    # -----------------------------
    noise_rms = 600.0
    noise_alpha = 0.02

    preroll: list[np.ndarray] = []
    preroll_max_chunks = 6  # ~ small leading buffer; actual duration depends on chunk sizes

    utter_chunks: list[np.ndarray] = []
    in_speech = False
    silence_s = 0.0
    utter_s = 0.0

    # -----------------------------
    # Think / mute / rate limit
    # -----------------------------
    muted_until = 0.0
    thinking = False
    next_think_ts = 0.0
    awaiting_reply = False
    thinking_stop = threading.Event()
    thinking_thread: Optional[threading.Thread] = None
    reply_ready = threading.Event()
    reply_lock = threading.Lock()
    reply_text: str = ""
    reply_cmd: str = ""
    llm_inflight = False
    turn_id = 0  # increments on each "hard reset everything" to invalidate stale LLM worker results

    last_sent_text = ""
    last_sent_ts = 0.0

    # -----------------------------
    # Wake probe state
    # -----------------------------
    wake_detected = False
    wake_ack_spoken = False
    last_wake_probe_ts = 0.0
    wake_probe_count = 0
    wake_probe_interval_s = max(0.05, args.wake_probe_interval_ms / 1000.0)

    INITIAL_PROMPT = (
        "This is a conversation with a robot named Sourccey. "
        "Common words include: Sourccey, robot, move, follow, stop, hello, task."
    )

    # -----------------------------
    # Helpers
    # -----------------------------
    def reset_speech_state() -> None:
        """Reset speech/VAD/wake tracking only (keeps thinking/thread state)."""
        nonlocal in_speech, silence_s, utter_s, utter_chunks, preroll
        nonlocal wake_detected, wake_ack_spoken, last_wake_probe_ts, wake_probe_count

        in_speech = False
        silence_s = 0.0
        utter_s = 0.0
        utter_chunks.clear()
        preroll.clear()

        wake_detected = False
        wake_ack_spoken = False
        last_wake_probe_ts = 0.0
        wake_probe_count = 0

    def stop_thinking() -> None:
        """Stop the thinking repeater immediately."""
        nonlocal thinking, next_think_ts
        thinking = False
        next_think_ts = 0.0
        thinking_stop.set()

    def hard_reset_utterance() -> None:
        nonlocal in_speech, silence_s, utter_s, utter_chunks, preroll
        nonlocal wake_detected, wake_ack_spoken, thinking, next_think_ts, last_wake_probe_ts, awaiting_reply
        nonlocal llm_inflight
        reset_speech_state()

        stop_thinking()
        awaiting_reply = False
        llm_inflight = False

    def end_utterance(mute_s: float = 0.0) -> None:
        """Always clear speech buffers after deciding an utterance outcome."""
        nonlocal muted_until
        reset_speech_state()
        if mute_s and mute_s > 0:
            muted_until = max(muted_until, time.time() + float(mute_s))

    def hard_reset_everything(after_reply: bool = False) -> None:
        """Reset ALL runtime state so the next interaction behaves like a fresh start.

        When `after_reply=True`, we keep the post-reply mute window intact (caller sets muted_until).
        """
        nonlocal noise_rms, muted_until
        nonlocal thinking, next_think_ts, awaiting_reply, turn_id
        nonlocal reply_text, reply_cmd, llm_inflight, last_sent_text, last_sent_ts
        nonlocal thinking_thread

        # Stop any thinking loop immediately
        stop_thinking()
        awaiting_reply = False

        # Clear reply worker state
        llm_inflight = False
        reply_ready.clear()
        with reply_lock:
            reply_text = ""
            reply_cmd = ""
        turn_id += 1
        reset_llm_history()

        # Reset VAD/noise + speech buffers
        noise_rms = max(600.0, float(args.noise_rms_floor))
        reset_speech_state()

        # Clear anti-duplicate guards (prevents "works once then ignores the next command")
        last_sent_text = ""
        last_sent_ts = 0.0

        # Optionally drain any queued audio frames so we don't immediately re-trigger on stale audio.
        # Keep it short so we don't block shutdown.
        try:
            t0 = time.monotonic()
            while (time.monotonic() - t0) < 0.15:
                try:
                    sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
        except Exception:
            pass

        # Allow a new thinking thread to be created on next use.
        thinking_thread = None

    def transcribe(audio_i16: np.ndarray) -> str:
        if audio_i16.size == 0:
            return ""
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        # NOTE: faster-whisper returns an iterator; we must consume it inside try
        try:
            segments, _ = model.transcribe(
                audio_f32,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=args.whisper_vad_filter,
                initial_prompt=INITIAL_PROMPT,
            )
            parts = []
            for seg in segments:
                t = (seg.text or "").strip()
                if t:
                    parts.append(t)
            return " ".join(parts).strip()
        except Exception as e:
            # Never crash the main loop on Whisper failures
            print(f"[ASR ERROR] {e}", file=sys.stderr)
            return ""

    def speak_thinking_now() -> None:
        nonlocal muted_until, thinking, next_think_ts, thinking_thread, thinking_stop
        if args.disable_thinking:
            return
        txt = (args.thinking_text or "").strip()
        if not txt:
            return
        now = time.time()
        try:
            robot.send_text(txt)
        except Exception:
            return
        # NOTE: don't hard-mute the mic if we're already inside speech (otherwise we can chop the command).
        # The main loop will ignore `muted_until` only when not in_speech.
        muted_until = max(muted_until, now + 1.2)

        thinking = True
        next_think_ts = now + float(args.thinking_interval_s)

        # Ensure "Thinking..." repeats at a steady cadence even while Whisper/LLM blocks.
        # Always restart the thread cleanly to avoid stale Event/thread state after many turns.
        if thinking_thread is None or not thinking_thread.is_alive():
            thinking_stop.clear()
        else:
            # Stop old thread (if any) and start a fresh one.
            thinking_stop.set()
            thinking_stop = threading.Event()

            interval_s = max(0.25, float(args.thinking_interval_s))
            txt_copy = txt
            stop_event = thinking_stop

            def _thinking_loop(ev: threading.Event) -> None:
                nonlocal muted_until, next_think_ts, thinking
                # First "Thinking..." already sent; schedule from that moment.
                next_t = time.monotonic() + interval_s
                while not ev.is_set():
                    # Sleep until the next scheduled time, maintaining steady cadence (no drift).
                    now_m = time.monotonic()
                    sleep_s = next_t - now_m
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 0.25))
                        continue

                    if not thinking:
                        # stop quickly when reply is ready
                        break

                    try:
                        robot.send_text(txt_copy)
                    except Exception:
                        pass

                    wall_now = time.time()
                    muted_until = max(muted_until, wall_now + 1.2)
                    next_think_ts = wall_now + interval_s
                    next_t += interval_s

            thinking_thread = threading.Thread(target=_thinking_loop, args=(stop_event,), daemon=True)
            thinking_thread.start()

    def tick_thinking() -> None:
        nonlocal muted_until, next_think_ts, thinking
        # With the dedicated thinking thread, the main loop no longer needs to schedule repeats.
        if True:
            return
        txt = (args.thinking_text or "").strip()
        if not txt:
            return
        now = time.time()
        if now >= next_think_ts:
            try:
                robot.send_text(txt)
            except Exception:
                return
            muted_until = max(muted_until, now + 1.2)
            next_think_ts = now + float(args.thinking_interval_s)

    # -----------------------------
    # Main loop
    # -----------------------------
    try:
        while True:
            # periodic thinking while waiting on LLM, etc.
            tick_thinking()

            # If the LLM worker finished, speak the reply ASAP (without waiting for more audio).
            if reply_ready.is_set():
                with reply_lock:
                    r_txt = (reply_text or "").strip()
                    r_cmd = (reply_cmd or "").strip()
                reply_ready.clear()

                stop_thinking()
                awaiting_reply = False
                llm_inflight = False

                if r_txt:
                    robot.send_text(r_txt)
                    if args.debug:
                        print(f"[USER ] {r_cmd}")
                        print(f"[ROBOT] {r_txt}")

                    last_sent_text = r_cmd
                    last_sent_ts = time.time()
                    muted_until = last_sent_ts + args.mute_after_send_s

                # HARD RESET EVERYTHING after we finish speaking a reply.
                hard_reset_everything(after_reply=True)

            try:
                data = sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                try:
                    robot.poll_text_message()
                except Exception:
                    pass
                time.sleep(0.01)
                continue

            # While awaiting a confirmed command's reply, the mic is OFF: discard audio frames.
            # IMPORTANT: do NOT gate on `thinking` because wake-probe uses thinking as an early ACK
            # while the user is still speaking their command.
            if awaiting_reply:
                time.sleep(0.01)
                continue

            now = time.time()
            # If muted and not already tracking speech, ignore audio entirely (prevents robot/TTS echo triggering VAD).
            # But if we're already inside a user utterance, keep accumulating so we don't drop the rest of the command.
            if now < muted_until and not in_speech:
                continue

            audio = np.frombuffer(data, dtype=np.int16)
            if audio.size == 0:
                continue

            chunk_s = audio.size / args.sample_rate
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

            # Track noise when not speaking
            if not in_speech:
                noise_rms = (1 - noise_alpha) * noise_rms + noise_alpha * rms
                noise_rms = max(noise_rms, float(args.noise_rms_floor))

            start_thr = max(args.min_speech_rms, noise_rms * args.speech_rms_mult)
            end_thr = max(args.min_speech_rms * 0.5, noise_rms * args.end_mult)

            is_speech = rms >= (start_thr if not in_speech else end_thr)

            # preroll buffer (for not chopping wake word)
            preroll.append(audio)
            if len(preroll) > preroll_max_chunks:
                preroll.pop(0)

            # Enter speech only after sustained start_ms
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
                # let noise track a little during trailing silence
                noise_rms = (1 - noise_alpha) * noise_rms + noise_alpha * rms
                noise_rms = max(noise_rms, float(args.noise_rms_floor))

            # -----------------------------
            # FAST wake probe (only while in speech)
            # -----------------------------
            if (
                in_speech
                and not wake_detected
                and utter_s >= float(args.wake_probe_min_s)
                and (now - last_wake_probe_ts) >= wake_probe_interval_s
            ):
                last_wake_probe_ts = now
                # Avoid runaway probes on noise: require stronger audio and cap total probes/utterance.
                if wake_probe_count < int(args.wake_probe_max_per_utterance):
                    probe_gate = max(
                        float(args.min_speech_rms),
                        start_thr * float(args.wake_probe_min_rms_mult),
                    )
                    if rms >= probe_gate:
                        wake_probe_count += 1
                        probe_audio = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)
                        probe_text = normalize_robot_terms(transcribe(probe_audio)).strip()

                        if args.debug and probe_text:
                            if args.wake_probe_print_all or has_wake_word(probe_text):
                                print(f"[WAKE PROBE] {probe_text}")

                        if has_wake_word(probe_text):
                            wake_detected = True
                            # 🔥 IMMEDIATE "Thinking..." the moment wake word is detected
                            if not wake_ack_spoken:
                                speak_thinking_now()
                                wake_ack_spoken = True

            # -----------------------------
            # END OF UTTERANCE
            # -----------------------------
            if (silence_s >= args.silence_ms / 1000.0) or (utter_s >= args.max_utterance_s):
                full = np.concatenate(utter_chunks) if utter_chunks else np.array([], dtype=np.int16)

                # Too short? ignore
                if full.size == 0:
                    end_utterance(mute_s=0.05)
                    continue
                full_dur = full.size / args.sample_rate
                if full_dur < args.min_utterance_s:
                    end_utterance(mute_s=0.05)
                    continue

                # Full transcription
                text = normalize_robot_terms(transcribe(full)).strip()
                if args.debug:
                    print(f"[HEARD] {text}")

                if not has_wake_word(text):
                    end_utterance(mute_s=0.05)
                    continue

                cmd = strip_wake_word(text).strip()

                # =============================
                # STEP 4: EARLIEST THINKING START
                # =============================
                if cmd and not args.disable_thinking and not thinking:
                    if args.debug:
                        print("[THINK ] Command confirmed, starting thinking immediately")

                    speak_thinking_now()

                # ---- WAKE WORD ONLY (no command yet) ----
                if not cmd:
                    if args.debug:
                        print("[INFO ] Wake word detected, no command yet")

                    # Do NOT start thinking yet
                    awaiting_reply = False
                    thinking = False
                    thinking_stop.set()
                    end_utterance(mute_s=0.2)
                    continue

                # ---- VALID COMMAND CONFIRMED ----
                awaiting_reply = True

                # Always start the thinking repeater once a full command is confirmed.
                if not args.disable_thinking:
                    if args.debug:
                        print("[THINK ] Starting thinking loop")
                    speak_thinking_now()

                cmd = strip_wake_word(text).strip()

                # --- WAKE WORD ONLY (no command) ---
                if not cmd:
                    if args.debug:
                        print("[INFO ] Wake word only, awaiting command")

                    # Stop thinking immediately
                    stop_thinking()

                    # Brief mute to avoid instant re-trigger
                    muted_until = time.time() + 0.2
                    end_utterance(mute_s=0.2)
                    continue

                if len(cmd.replace(" ", "")) < args.min_text_chars:
                    end_utterance(mute_s=0.1)
                    continue
                if (now - last_sent_ts) < args.min_interval_s:
                    end_utterance(mute_s=0.1)
                    continue
                if is_similar_text(cmd, last_sent_text):
                    end_utterance(mute_s=0.1)
                    continue

                # Start/restart thinking loop (if wake was only confirmed late)
                if not args.disable_thinking:
                    speak_thinking_now()

                # Generate reply in a background worker so "Thinking..." cadence is stable.
                if not llm_inflight:
                    llm_inflight = True
                    my_turn = turn_id

                    def _llm_worker(command_text: str, worker_turn: int) -> None:
                        nonlocal reply_text, reply_cmd
                        try:
                            if sounds_like_physical_action(command_text):
                                out = "I can explain how to do that, but I can't physically perform it."
                            else:
                                out = ask_llm(command_text)
                        except Exception as e:
                            print(f"[LLM ERROR] {e}", file=sys.stderr)
                            out = "Sorry, I had trouble thinking just now."
                        # If we reset since this worker started, drop the result (prevents stale replies / stuck states).
                        if worker_turn != turn_id:
                            return
                        with reply_lock:
                            reply_text = out
                            reply_cmd = command_text
                        reply_ready.set()

                    threading.Thread(target=_llm_worker, args=(cmd, my_turn), daemon=True).start()

                # Clear speech buffers now; mic remains off while awaiting_reply/thinking.
                end_utterance(mute_s=0.0)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        thinking_stop.set()
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
