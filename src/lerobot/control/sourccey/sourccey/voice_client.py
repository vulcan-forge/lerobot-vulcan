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
import difflib
import json
import re
import sys
import time
import threading
from typing import Optional, List
from pathlib import Path
import subprocess
import os

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

        # Some Windows setups have multiple CUDA toolkit versions on PATH (v12.x, v13.x, etc.).
        # That can cause cuDNN/CT2 to load incompatible DLLs even when the correct pip-provided
        # NVIDIA runtime packages are installed, leading to CUDNN_STATUS_NOT_INITIALIZED.
        #
        # For *this process only*, prefer the venv's `site-packages/nvidia/**/bin` DLLs by
        # removing CUDA Toolkit paths from PATH (unless explicitly disabled).
        if os.environ.get("SOURCCEY_CUDA_PATH_SANITIZE", "1") != "0":
            parts = (os.environ.get("PATH", "") or "").split(os.pathsep)
            cleaned: list[str] = []
            for p in parts:
                pl = (p or "").lower()
                # Keep the pip-provided NVIDIA DLL dirs we just prepended.
                if "site-packages" in pl and (f"{os.sep}nvidia{os.sep}" in pl):
                    cleaned.append(p)
                    continue
                # Drop CUDA toolkit entries (bin/libnvvp) to avoid version conflicts.
                if "nvidia gpu computing toolkit" in pl and f"{os.sep}cuda{os.sep}" in pl:
                    continue
                if f"{os.sep}cuda{os.sep}v" in pl and (pl.endswith(f"{os.sep}bin") or "libnvvp" in pl):
                    continue
                cleaned.append(p)
            os.environ["PATH"] = os.pathsep.join(cleaned)
    except Exception:
        pass


_configure_windows_cuda_dll_search()

from faster_whisper import WhisperModel

# -----------------------------
# Text helpers
# -----------------------------
def normalize_robot_terms(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(
        # Common Whisper spellings / near-homophones for "Sourccey"
        r"\b(sorcy|sorsi|sorcie|sorci|sorsee|sorcey|"
        r"orsi|doris|"
        r"sourcey|sourcee|sourcy|soursee|sourcie|sourci|"
        r"saucey|saucy|"
        r"isourcing)\b",
        "Sourccey",
        text,
        flags=re.IGNORECASE,
    ).strip()

def _wake_cleanup(text: str) -> str:
    # Normalize punctuation/spacing for robust wake matching without affecting downstream command text.
    t = (text or "").strip().lower()
    # Normalize common unicode quotes
    t = re.sub(r"[\u2019\u2018\u201c\u201d]", "'", t)
    # Replace punctuation with spaces (keep letters/digits/_ and spaces and apostrophes)
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _wake_first_token(text: str) -> str:
    t = _wake_cleanup(text)
    if not t:
        return ""
    # Drop common attention prefixes
    t = re.sub(r"^(hey|ok|okay)\s+", "", t).strip()
    if not t:
        return ""
    return t.split(" ", 1)[0].strip()


def _wake_fuzzy_match(token: str, target: str, min_ratio: float) -> bool:
    if not token or not target:
        return False
    try:
        r = float(min_ratio)
    except Exception:
        r = 0.76
    r = max(0.0, min(1.0, r))
    token = token.lower().strip()
    target = target.lower().strip()
    if not token or not target:
        return False
    if token == target:
        return True
    return difflib.SequenceMatcher(None, token, target).ratio() >= r


def has_wake_word(text: str, *, enable_fuzzy: bool = True, fuzzy_min_ratio: float = 0.76) -> bool:
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
    if any(t.startswith(p) for p in prefixes):
        return True
    if not enable_fuzzy:
        return False
    tok = _wake_first_token(text)
    return _wake_fuzzy_match(tok, WAKE_WORD, fuzzy_min_ratio)


def strip_wake_word(text: str, *, enable_fuzzy: bool = True, fuzzy_min_ratio: float = 0.76) -> str:
    if not text:
        return ""
    s = text.strip()
    # Exact path: "Sourccey" (after optional hey/ok/okay)
    stripped = re.sub(
        r"^(hey|ok|okay)?\s*sourccey[:,]?\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    if stripped != s:
        return stripped
    if not enable_fuzzy:
        return s
    # Fuzzy path: if the first token sounds like the wake word, drop it.
    cleaned = _wake_cleanup(s)
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(hey|ok|okay)\s+", "", cleaned).strip()
    if not cleaned:
        return ""
    first, *rest = cleaned.split(" ")
    if _wake_fuzzy_match(first, WAKE_WORD, fuzzy_min_ratio):
        return " ".join(rest).strip()
    return s


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
    # Defaults: keep a working "no-args" path, but allow overrides via flag/env.
    default_robot_ip = os.environ.get("SOURCCEY_ROBOT_IP", "192.168.1.213")
    p.add_argument(
        "--robot_ip",
        default=default_robot_ip,
        help="Robot/host IP (default: 192.168.1.213). Can also be set via SOURCCEY_ROBOT_IP.",
    )
    p.add_argument("--audio-port", type=int, default=5559)
    p.add_argument("--text-in-port", type=int, default=5557, help="Host text input port (client -> host).")
    p.add_argument("--text-out-port", type=int, default=5558, help="Host text output port (host -> client).")
    p.add_argument("--sample-rate", type=int, default=16000)

    # These defaults are tuned for your working setup; override via flags if desired.
    p.add_argument("--model", default=os.environ.get("SOURCCEY_ASR_MODEL", "large-v3"))
    p.add_argument("--device", default=os.environ.get("SOURCCEY_ASR_DEVICE", "cuda"))
    p.add_argument("--compute-type", default=os.environ.get("SOURCCEY_ASR_COMPUTE", "float16"))
    p.add_argument("--language", default="en")

    p.add_argument("--beam-size", type=int, default=3)
    # Whisper decoding / hallucination controls
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Whisper sampling temperature (0.0 is most deterministic).",
    )
    p.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, allow Whisper to condition on previous text (can increase hallucinations in noisy audio).",
    )
    p.add_argument(
        "--no-speech-threshold",
        type=float,
        default=None,
        help="Optional Whisper no-speech threshold (higher can reduce hallucinated text on silence).",
    )
    p.add_argument(
        "--log-prob-threshold",
        type=float,
        default=None,
        help="Optional Whisper log-prob threshold (can reduce low-confidence hallucinations).",
    )
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
    p.add_argument(
        "--wake-probe-prompt",
        default="",
        help="Initial prompt used for wake-probe transcriptions. Default empty to reduce prompt-echo hallucinations.",
    )
    p.add_argument(
        "--thinking-on-wake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, say 'Thinking…' immediately when the wake word is detected (even before the command is finished).",
    )

    # Wake-word sensitivity (fuzzy matching)
    p.add_argument(
        "--wake-fuzzy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fuzzy wake-word matching (accepts near-homophones of 'Sourccey').",
    )
    p.add_argument(
        "--wake-fuzzy-min-ratio",
        type=float,
        default=0.90,
        help="Fuzzy match threshold in [0..1]. Lower = more sensitive, higher = fewer false wakes.",
    )

    # Mic unmute behavior after robot speaks a reply
    p.add_argument(
        "--unmute-on-tts-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If supported by the host, re-enable the mic exactly when robot TTS ends (instead of a fixed timer).",
    )
    p.add_argument(
        "--post-tts-mute-s",
        type=float,
        default=0.25,
        help="Small extra mute window after TTS end to avoid tail echo retrigger.",
    )
    p.add_argument(
        "--tts-fallback-wpm",
        type=float,
        default=165.0,
        help="Fallback TTS speed estimate (words per minute) used if host doesn't send TTS end events.",
    )
    p.add_argument(
        "--tts-fallback-base-s",
        type=float,
        default=0.4,
        help="Fallback constant seconds added to TTS duration estimate.",
    )

    p.add_argument("--whisper-vad-filter", action="store_true")
    p.add_argument("--allow-cpu-fallback", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--voice-eval-config",
        default=str(Path(__file__).resolve().parents[5] / "data" / "voice_eval_commands.json"),
        help="JSON mapping for 'execute command N' voice triggers.",
    )
    p.add_argument(
        "--enable-voice-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable voice-triggered 'execute command N' which launches lerobot evaluate runs.",
    )
    p.add_argument(
        "--block-while-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, ignore new audio/commands while an eval run is active.",
    )
    p.add_argument(
        "--require-command-mode-for-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, require a 'commands' keyword before allowing voice-triggered eval runs.",
    )
    p.add_argument(
        "--command-mode-word",
        default="commands",
        help="Second wake word for eval triggers (default: 'commands'). Example: 'Sourccey, commands, execute command 1'.",
    )
    p.add_argument(
        "--debug-audio",
        action="store_true",
        help="Print basic audio stats (duration/rms/peak) for each transcription (helps confirm audio is non-silent).",
    )

    args = p.parse_args(argv)

    def _load_eval_map() -> dict:
        if not args.enable_voice_eval:
            return {}
        try:
            pth = Path(args.voice_eval_config).expanduser()
            if not pth.is_absolute():
                # Make relative paths resolve from repo root (modules/lerobot-vulcan)
                pth = Path.cwd() / pth
            return json.loads(pth.read_text(encoding="utf-8"))
        except Exception:
            return {}

    eval_map = _load_eval_map()
    eval_proc: Optional[subprocess.Popen] = None
    eval_lock = threading.Lock()
    eval_active_id: Optional[str] = None

    def _build_evaluate_argv(entry: dict) -> list[str]:
        # Build args for: python -m lerobot.control.sourccey.sourccey.evaluate ...
        remote_ip = str(entry.get("remote_ip", "{robot_ip}")).replace("{robot_ip}", str(args.robot_ip))
        out: list[str] = [
            sys.executable,
            "-m",
            "lerobot.control.sourccey.sourccey.evaluate",
            f"--id={entry.get('id', 'sourccey')}",
            f"--remote_ip={remote_ip}",
            f"--model_path={entry['model_path']}",
        ]
        dataset = entry.get("dataset") or {}
        for k, v in dataset.items():
            out.append(f"--dataset.{k}={v}")
        # Make it automation-friendly by default
        out.append(f"--enable_keyboard={bool(entry.get('enable_keyboard', False))}")
        out.append(f"--enable_rerun={bool(entry.get('enable_rerun', False))}")
        return out

    def _maybe_finish_eval() -> None:
        nonlocal eval_proc, eval_active_id, awaiting_reply
        with eval_lock:
            p = eval_proc
            cid = eval_active_id
        if p is None:
            return
        rc = p.poll()
        if rc is None:
            return
        entry = eval_map.get(str(cid or "")) if cid is not None else None
        cmd_name = ""
        try:
            if entry:
                cmd_name = (entry.get("name") or entry.get("description") or "").strip()
        except Exception:
            cmd_name = ""
        with eval_lock:
            eval_proc = None
            eval_active_id = None
        awaiting_reply = False
        if cmd_name:
            send_text(f"Finished {cmd_name} (exit={rc}).")
        else:
            send_text(f"Eval command {cid} finished (exit={rc}).")

    def _start_eval_command(cmd_id: str) -> bool:
        nonlocal eval_proc, eval_active_id, awaiting_reply
        entry = eval_map.get(str(cmd_id))
        if not entry:
            send_text(f"I don't have an eval mapping for command {cmd_id}.")
            return False
        if "model_path" not in entry:
            send_text(f"Eval command {cmd_id} is missing model_path.")
            return False
        cmd_name = (entry.get("name") or entry.get("description") or f"command {cmd_id}").strip()
        with eval_lock:
            if eval_proc is not None and eval_proc.poll() is None:
                send_text("I'm already running an eval command.")
                return False
            argv2 = _build_evaluate_argv(entry)
            try:
                eval_proc = subprocess.Popen(
                    argv2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(Path.cwd()),
                )
                eval_active_id = str(cmd_id)
                if bool(args.block_while_eval):
                    awaiting_reply = True
                send_text(f"Executing {cmd_name}.")
                return True
            except Exception as e:
                eval_proc = None
                eval_active_id = None
                send_text(f"Failed to start eval command {cmd_id}: {e}")
                return False

    _NUM_WORDS = {
        "zero": "0",
        "one": "1",
        "won": "1",
        "two": "2",
        "to": "2",
        "too": "2",
        "three": "3",
        "tree": "3",
        "four": "4",
        "for": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "ate": "8",
        "nine": "9",
        "ten": "10",
    }

    def _parse_command_id(raw: str) -> Optional[str]:
        s = (raw or "").strip().lower()
        if not s:
            return None
        if s.isdigit():
            return s
        return _NUM_WORDS.get(s)

    def _strip_command_mode_prefix(text: str) -> tuple[bool, str]:
        """
        Returns (in_command_mode, remaining_text).
        Accepts: "commands ..." or "... commands, ..." etc.
        """
        t = (text or "").strip()
        if not t:
            return False, ""
        mode = (args.command_mode_word or "commands").strip().lower()
        # Be robust to ASR: treat "command" and "commands" as equivalent, and accept "command mode".
        modes = {m for m in [mode, mode.rstrip("s"), f"{mode.rstrip('s')}s", "command", "commands", "command mode"] if m}

        # Find the first occurrence of any mode token and take everything after it.
        for mword in sorted(modes, key=len, reverse=True):
            m = re.search(rf"\\b{re.escape(mword)}\\b[:,]?\\s*(.*)$", t, flags=re.IGNORECASE)
            if not m:
                continue
            return True, (m.group(1) or "").strip()

        return False, t

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
    asr_forced_cpu = False

    # -----------------------------
    # Direct ZMQ text transport (no SourcceyClient dependency)
    # -----------------------------
    text_ctx = zmq.Context()
    text_out_sock = text_ctx.socket(zmq.PUSH)  # client -> host (host binds PULL)
    text_out_sock.setsockopt(zmq.CONFLATE, 1)
    text_out_sock.connect(f"tcp://{args.robot_ip}:{int(args.text_in_port)}")

    text_in_sock = text_ctx.socket(zmq.PULL)  # host -> client (host binds PUSH)
    text_in_sock.setsockopt(zmq.CONFLATE, 1)
    text_in_sock.connect(f"tcp://{args.robot_ip}:{int(args.text_out_port)}")

    def send_text(message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        try:
            text_out_sock.send_string(msg, flags=zmq.NOBLOCK)
        except Exception:
            pass

    def poll_text_message() -> Optional[str]:
        try:
            return text_in_sock.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return None
        except Exception:
            return None

    print("✓ Connected to host text gateway")

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

    # Host TTS tracking (so we can unmute immediately when the robot finishes speaking)
    waiting_tts_end = False
    tts_current_seq = 0
    tts_fallback_unmute_ts = 0.0

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
        "Common words include: Sourccey."
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

    def hard_reset_everything(after_reply: bool = False, keep_awaiting_reply: bool = False) -> None:
        """Reset ALL runtime state so the next interaction behaves like a fresh start.

        When `after_reply=True`, we keep the post-reply mute window intact (caller sets muted_until).
        """
        nonlocal noise_rms, muted_until
        nonlocal thinking, next_think_ts, awaiting_reply, turn_id
        nonlocal reply_text, reply_cmd, llm_inflight, last_sent_text, last_sent_ts
        nonlocal thinking_thread

        # Stop any thinking loop immediately
        stop_thinking()
        if not keep_awaiting_reply:
            awaiting_reply = False

        # Clear reply worker state
        llm_inflight = False
        reply_ready.clear()
        with reply_lock:
            reply_text = ""
            reply_cmd = ""
        turn_id += 1
        reset_llm_history()

        # Reset speech buffers, but KEEP the learned noise estimate so the next utterance
        # doesn't start with miscalibrated thresholds (which can clip the wake word).
        noise_rms = max(float(noise_rms), float(args.noise_rms_floor))
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

    def _estimate_tts_duration_s(text: str) -> float:
        # Fallback only: estimate how long the host will speak, based on configured WPM.
        words = len(re.findall(r"\w+", text or ""))
        wpm = max(30.0, float(args.tts_fallback_wpm))
        return float(args.tts_fallback_base_s) + (words / (wpm / 60.0))

    def _drain_audio(max_s: float = 0.2) -> None:
        # Drop queued audio frames so we don't immediately re-trigger on stale buffered audio.
        t0 = time.monotonic()
        while (time.monotonic() - t0) < float(max_s):
            try:
                sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break

    def _handle_host_text_message(msg: str) -> None:
        nonlocal awaiting_reply, waiting_tts_end, tts_current_seq, muted_until
        if not msg:
            return
        # Host emits: "__TTS_EVENT__:start:<seq>" and "__TTS_EVENT__:end:<seq>"
        if not msg.startswith("__TTS_EVENT__:"):
            return
        try:
            _prefix, kind, seq_s = msg.split(":", 2)
            seq = int(seq_s)
        except Exception:
            return
        if kind == "start":
            tts_current_seq = seq
            return
        if kind == "end":
            if seq != int(tts_current_seq):
                return
            # We are ready for the next user turn immediately after TTS ends.
            waiting_tts_end = False
            awaiting_reply = False
            muted_until = max(muted_until, time.time() + float(args.post_tts_mute_s))
            _drain_audio(max_s=0.25)

    def _poll_host_text(limit: int = 5) -> None:
        for _ in range(int(limit)):
            m = poll_text_message()
            if not m:
                break
            _handle_host_text_message(m)
            # Always feed messages through our handler (covers the case where the client
            # doesn't support callbacks, and is harmless when it does).
            _handle_host_text_message(m)

    def transcribe(audio_i16: np.ndarray) -> str:
        nonlocal model, asr_forced_cpu
        if audio_i16.size == 0:
            return ""
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        if args.debug_audio:
            rms = float(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))
            peak = int(np.max(np.abs(audio_i16))) if audio_i16.size else 0
            dur_s = float(audio_i16.size) / float(args.sample_rate)
            print(f"[AUDIO] dur_s={dur_s:.2f} rms={rms:.1f} peak={peak}", file=sys.stderr)

        try:
            segments, _ = model.transcribe(
                audio_f32,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=args.whisper_vad_filter,
                initial_prompt=INITIAL_PROMPT,
                temperature=float(args.temperature),
                condition_on_previous_text=bool(args.condition_on_previous_text),
                no_speech_threshold=args.no_speech_threshold,
                log_prob_threshold=args.log_prob_threshold,
            )
            parts: list[str] = []
            for seg in segments:
                t = (seg.text or "").strip()
                if t:
                    parts.append(t)
            return " ".join(parts).strip()

        except Exception as e:
            # If CUDA/cuDNN is misconfigured at runtime, retry on CPU (best-effort).
            msg = str(e)
            is_cudnn_init = ("CUDNN_STATUS_NOT_INITIALIZED" in msg) or ("cuDNN" in msg) or ("cudnn" in msg)
            if is_cudnn_init and args.allow_cpu_fallback and not asr_forced_cpu:
                asr_forced_cpu = True
                try:
                    print("[WARN] ASR cuDNN error detected; switching Whisper to CPU int8 fallback.", file=sys.stderr)
                    model = WhisperModel(args.model, device="cpu", compute_type="int8")
                    segments, _ = model.transcribe(
                        audio_f32,
                        language=args.language,
                        beam_size=args.beam_size,
                        vad_filter=args.whisper_vad_filter,
                        initial_prompt=INITIAL_PROMPT,
                        temperature=float(args.temperature),
                        condition_on_previous_text=bool(args.condition_on_previous_text),
                        no_speech_threshold=args.no_speech_threshold,
                        log_prob_threshold=args.log_prob_threshold,
                    )
                    parts: list[str] = []
                    for seg in segments:
                        t = (seg.text or "").strip()
                        if t:
                            parts.append(t)
                    return " ".join(parts).strip()
                except Exception as e2:
                    print(f"[ASR ERROR] {e2}", file=sys.stderr)
                    return ""

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
        send_text(txt)
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

                    send_text(txt_copy)

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
            send_text(txt)
            muted_until = max(muted_until, now + 1.2)
            next_think_ts = now + float(args.thinking_interval_s)

    # -----------------------------
    # Main loop
    # -----------------------------
    try:
        while True:
            _maybe_finish_eval()
            _poll_host_text()

            # periodic thinking while waiting on LLM, etc.
            tick_thinking()

            # If the LLM worker finished, speak the reply ASAP (without waiting for more audio).
            if reply_ready.is_set():
                with reply_lock:
                    r_txt = (reply_text or "").strip()
                    r_cmd = (reply_cmd or "").strip()
                reply_ready.clear()

                stop_thinking()
                llm_inflight = False

                if r_txt:
                    send_text(r_txt)
                    if args.debug:
                        print(f"[USER ] {r_cmd}")
                        print(f"[ROBOT] {r_txt}")

                    last_sent_text = r_cmd
                    last_sent_ts = time.time()
                    if bool(args.unmute_on_tts_end):
                        # Stay muted until we get host TTS end (or fallback estimate, if host doesn't send events).
                        awaiting_reply = True
                        waiting_tts_end = True
                        tts_fallback_unmute_ts = last_sent_ts + max(
                            float(args.mute_after_send_s),
                            _estimate_tts_duration_s(r_txt),
                        )
                    else:
                        awaiting_reply = False
                        muted_until = last_sent_ts + args.mute_after_send_s

                # Drop any remaining work/buffers immediately.
                # If we're waiting for TTS end, keep `awaiting_reply=True` so we don't process audio.
                hard_reset_everything(after_reply=True, keep_awaiting_reply=awaiting_reply)

            try:
                data = sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                _poll_host_text()
                time.sleep(0.01)
                continue

            # While awaiting a confirmed command's reply, the mic is OFF: discard audio frames.
            # IMPORTANT: do NOT gate on `thinking` because wake-probe uses thinking as an early ACK
            # while the user is still speaking their command.
            if awaiting_reply:
                _poll_host_text()
                # Fallback: if we never got host TTS end events, unmute after an estimated duration.
                if waiting_tts_end and (time.time() >= float(tts_fallback_unmute_ts)):
                    waiting_tts_end = False
                    awaiting_reply = False
                    muted_until = max(muted_until, time.time() + float(args.post_tts_mute_s))
                    _drain_audio(max_s=0.25)
                else:
                    # Keep the audio socket fresh by draining queued frames quickly.
                    _drain_audio(max_s=0.01)
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
                        # Wake probe: use a minimal prompt (or none) to reduce prompt-echo hallucinations.
                        probe_text = normalize_robot_terms(
                            whisper_transcribe(
                                model=model,
                                audio_i16=probe_audio,
                                language=args.language,
                                beam_size=args.beam_size,
                                vad_filter=args.whisper_vad_filter,
                                initial_prompt=str(args.wake_probe_prompt or ""),
                            )
                        ).strip()

                        if args.debug and probe_text:
                            if args.wake_probe_print_all or has_wake_word(
                                probe_text,
                                enable_fuzzy=bool(args.wake_fuzzy),
                                fuzzy_min_ratio=float(args.wake_fuzzy_min_ratio),
                            ):
                                print(f"[WAKE PROBE] {probe_text}")

                        if has_wake_word(
                            probe_text,
                            enable_fuzzy=bool(args.wake_fuzzy),
                            fuzzy_min_ratio=float(args.wake_fuzzy_min_ratio),
                        ):
                            wake_detected = True
                            # 🔥 IMMEDIATE "Thinking..." the moment wake word is detected
                            if bool(args.thinking_on_wake) and not wake_ack_spoken:
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

                if not has_wake_word(
                    text,
                    enable_fuzzy=bool(args.wake_fuzzy),
                    fuzzy_min_ratio=float(args.wake_fuzzy_min_ratio),
                ):
                    end_utterance(mute_s=0.05)
                    continue

                cmd = strip_wake_word(
                    text,
                    enable_fuzzy=bool(args.wake_fuzzy),
                    fuzzy_min_ratio=float(args.wake_fuzzy_min_ratio),
                ).strip()

                # Voice-triggered eval commands: "execute command N"
                if bool(args.enable_voice_eval):
                    in_mode, rest = _strip_command_mode_prefix(cmd)
                    if (not bool(args.require_command_mode_for_eval)) or in_mode:
                        candidate = (rest if in_mode else cmd).strip()
                        
                        if args.debug:
                            print(f"[EVAL] cmd={cmd!r} in_mode={in_mode} candidate={candidate!r}", file=sys.stderr)

                        # If the user entered command mode but didn't say a command, prompt instead of sending to LLM.
                        if in_mode and not candidate:
                            send_text("Say: execute command one.")
                            end_utterance(mute_s=0.2)
                            continue

                        # Accept a few phrasings:
                        # - "execute command one"
                        # - "run command 1"
                        # - "command one"
                        # - "execute one"
                        # - "one" (if in command mode)
                        patterns = [
                            r"^(execute|run|start)\s+(?:eval\s+)?command\s+([a-z0-9]+)\b",
                            r"^command\s+([a-z0-9]+)\b",
                            r"^(execute|run|start)\s+([a-z0-9]+)\b",
                            r"^([a-z0-9]+)\b",  # Just a number/word if in command mode
                        ]
                        cmd_id_raw: Optional[str] = None
                        for pat in patterns:
                            mm = re.match(pat, candidate, flags=re.IGNORECASE)
                            if not mm:
                                continue
                            cmd_id_raw = mm.group(mm.lastindex or 1)
                            if args.debug:
                                print(f"[EVAL] pattern matched: {pat!r} -> {cmd_id_raw!r}", file=sys.stderr)
                            break

                        # Fallback: search for any number word in the candidate text
                        if cmd_id_raw is None and candidate:
                            words = re.findall(r"\b([a-z0-9]+)\b", candidate, flags=re.IGNORECASE)
                            for word in words:
                                parsed = _parse_command_id(word)
                                if parsed is not None:
                                    cmd_id_raw = word
                                    if args.debug:
                                        print(f"[EVAL] fallback found number word: {word!r} -> {parsed!r}", file=sys.stderr)
                                    break

                        if cmd_id_raw is not None:
                            cid = _parse_command_id(cmd_id_raw)
                            if cid is None:
                                send_text("I didn't understand which command number you meant.")
                            else:
                                if args.debug:
                                    print(f"[EVAL] executing command {cid}", file=sys.stderr)
                                _start_eval_command(cid)
                            end_utterance(mute_s=0.2)
                            continue

                        # If we required command mode and heard it, but didn't parse a command,
                        # do NOT forward to the LLM (prevents confusing responses like "What command one?").
                        if in_mode and bool(args.require_command_mode_for_eval):
                            if args.debug:
                                print(f"[EVAL] command mode detected but no command parsed from {candidate!r}", file=sys.stderr)
                            send_text("I didn't catch a command number. Try: execute command one.")
                            end_utterance(mute_s=0.2)
                            continue

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

                    # Ensure thinking is OFF (wake-only should not keep thinking running)
                    awaiting_reply = False
                    stop_thinking()
                    end_utterance(mute_s=0.2)
                    continue

                # ---- VALID COMMAND CONFIRMED ----
                awaiting_reply = True

                # Always start the thinking repeater once a full command is confirmed.
                if not args.disable_thinking:
                    if args.debug:
                        print("[THINK ] Starting thinking loop")
                    speak_thinking_now()

                cmd = strip_wake_word(
                    text,
                    enable_fuzzy=bool(args.wake_fuzzy),
                    fuzzy_min_ratio=float(args.wake_fuzzy_min_ratio),
                ).strip()

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
            text_out_sock.close()
            text_in_sock.close()
        except Exception:
            pass
        try:
            text_ctx.term()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
