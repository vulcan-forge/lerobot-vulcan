"""
Sourccey voice listener (FINAL BASELINE v2)

- Mono capture (PortAudio/ALSA device only allows mono here)
- Stateful high-pass filter (removes motor / chassis noise)
- Smooth software AGC (no hardware AGC available)
- Clean mono int16 stream for ASR
- Preroll + hangover gating

Optional:
- Can also run the Sourccey host gateway loop in-process, so you only need one command
  to bring up text + command/observation ZMQ ports *and* the audio stream.
"""

from __future__ import annotations

import argparse
import collections
import queue
import sys
import threading
import subprocess
import shutil
import time
from typing import Optional
import os

import numpy as np
import sounddevice as sd
import zmq

from .config_sourccey import SourcceyHostConfig


# -----------------------------
# DSP helpers
# -----------------------------
def rms_i16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


class HighPass1:
    """
    1st-order high-pass filter with persistent state across blocks.
    This is *critical* — resetting each block makes the signal "blocky" and hurts VAD/ASR.
    """

    def __init__(self, cutoff_hz: float, sr: int):
        self.cutoff_hz = float(cutoff_hz)
        self.sr = int(sr)

        rc = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        dt = 1.0 / float(self.sr)
        self.alpha = float(rc / (rc + dt))

        self.x_prev = 0.0
        self.y_prev = 0.0

    def process_i16(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        # y[n] = a*(y[n-1] + x[n] - x[n-1])
        xf = x.astype(np.float32)
        y = np.empty_like(xf)

        a = self.alpha
        x_prev = self.x_prev
        y_prev = self.y_prev

        for i in range(xf.size):
            xn = float(xf[i])
            yn = a * (y_prev + xn - x_prev)
            y[i] = yn
            x_prev = xn
            y_prev = yn

        self.x_prev = x_prev
        self.y_prev = y_prev

        return np.clip(y, -32768, 32767).astype(np.int16)


class SmoothAGC:
    """
    Gentle automatic gain control with smoothing.
    Avoids per-block pumping (which breaks VAD and causes Whisper hallucinations).
    """

    def __init__(
        self,
        target_rms: float = 220.0,
        min_gain: float = 0.2,
        max_gain: float = 3.0,
        attack_s: float = 0.05,   # faster when signal too quiet
        release_s: float = 0.25,  # slower when signal too loud
        sr: int = 16000,
        blocksize: int = 3200,
    ):
        self.target_rms = float(target_rms)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)

        self.sr = int(sr)
        self.blocksize = int(blocksize)
        self.block_s = self.blocksize / float(self.sr)

        # Convert time constants to smoothing coefficients per block
        # gain += (desired - gain) * k
        self.k_attack = 1.0 - float(np.exp(-self.block_s / max(1e-6, self.attack_s)))
        self.k_release = 1.0 - float(np.exp(-self.block_s / max(1e-6, self.release_s)))

        self.gain = 1.0

    def process_i16(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x

        cur = rms_i16(x) + 1e-6
        desired = self.target_rms / cur
        desired = float(np.clip(desired, self.min_gain, self.max_gain))

        # If we need MORE gain (signal too quiet) -> attack (faster)
        # If we need LESS gain (signal too loud) -> release (slower)
        k = self.k_attack if desired > self.gain else self.k_release
        self.gain = float(self.gain + (desired - self.gain) * k)

        y = x.astype(np.float32) * self.gain
        return np.clip(y, -32768, 32767).astype(np.int16)


# -----------------------------
# ZMQ publisher
# -----------------------------
class AudioStreamPublisher:
    def __init__(self, port: int):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        time.sleep(0.3)

    def send(self, data: bytes):
        try:
            self.sock.send(data, zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self):
        self.sock.close()
        self.ctx.term()


class TextTtsGateway:
    """
    Minimal text gateway + TTS (espeak-ng/espeak) for the voice pipeline.

    - Binds PULL on port_zmq_text_in (client -> host)
    - Binds PUSH on port_zmq_text_out (host -> client)
    - Speaks received text and emits __TTS_EVENT__ start/end messages so the client
      can unmute immediately when TTS ends.
    """

    def __init__(self, text_in_port: int, text_out_port: int):
        self.ctx = zmq.Context()
        self.sock_in = self.ctx.socket(zmq.PULL)
        self.sock_in.setsockopt(zmq.CONFLATE, 1)
        self.sock_in.bind(f"tcp://*:{int(text_in_port)}")

        self.sock_out = self.ctx.socket(zmq.PUSH)
        self.sock_out.setsockopt(zmq.CONFLATE, 1)
        self.sock_out.bind(f"tcp://*:{int(text_out_port)}")

        self._tts_process: Optional[subprocess.Popen] = None
        self._tts_seq = 0
        self._lock = threading.Lock()
        self._engine = self._tts_cmd()

        # Startup banner (helps debug networking / missing TTS)
        eng = self._engine or "NONE"
        print(
            f"[tts] gateway up: recv tcp://*:{int(text_in_port)} send tcp://*:{int(text_out_port)} engine={eng}",
            file=sys.stderr,
            flush=True,
        )

    def _send_event(self, kind: str, seq: int) -> None:
        try:
            self.sock_out.send_string(f"__TTS_EVENT__:{kind}:{int(seq)}", flags=zmq.NOBLOCK)
        except Exception:
            pass

    def _send_info(self, text: str) -> None:
        msg = (text or "").strip()
        if not msg:
            return
        try:
            self.sock_out.send_string(msg, flags=zmq.NOBLOCK)
        except Exception:
            pass

    def _tts_cmd(self) -> Optional[str]:
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def speak_text(self, message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        # Don't speak internal event/control messages (but DO print them for debugging).
        if msg.startswith("__TTS_EVENT__:"):
            return
        if msg.startswith("__EVAL_LOG__:") or msg.startswith("__EVAL_STATUS__:"):
            # Eval logs/status are for operator visibility; never speak them.
            print(f"[eval] {msg}", file=sys.stderr)
            return

        # Always log + ACK so the client can confirm delivery even if TTS is missing.
        print(f"[tts] IN: {msg}", file=sys.stderr)
        self._send_info(f"__SAY_ACK__:{msg}")

        tts_cmd = self._tts_cmd()
        if not tts_cmd:
            self._send_info("__SAY_ERR__:no_tts_engine (install espeak-ng or espeak)")
            return

        with self._lock:
            self._tts_seq += 1
            seq = int(self._tts_seq)
            # Stop any previous speech to prevent overlaps
            if self._tts_process and self._tts_process.poll() is None:
                try:
                    self._tts_process.terminate()
                except Exception:
                    pass
            try:
                self._tts_process = subprocess.Popen(
                    [
                        tts_cmd,
                        "-v",
                        "en-us+f3",
                        "-p",
                        "75",
                        "-s",
                        "165",
                        "-a",
                        "190",
                        msg,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                self._tts_process = None
                return

        self._send_event("start", seq)

        def _wait_and_emit_end(p: subprocess.Popen, s: int) -> None:
            try:
                p.wait()
            except Exception:
                pass
            self._send_event("end", s)

        if self._tts_process is not None:
            threading.Thread(target=_wait_and_emit_end, args=(self._tts_process, seq), daemon=True).start()

    def poll_once(self) -> None:
        try:
            msg = self.sock_in.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return
        except Exception:
            return
        self.speak_text(msg)

    def close(self) -> None:
        try:
            if self._tts_process and self._tts_process.poll() is None:
                try:
                    self._tts_process.terminate()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.sock_out.close()
            self.sock_in.close()
            self.ctx.term()
        except Exception:
            pass

def _run_host_loop(stop_event: threading.Event) -> None:
    """
    Run the Sourccey host gateway loop (ports 5555-5558) until stop_event is set.

    This is a lightly-adapted version of `sourccey_host.main()` that can be stopped
    from another thread (KeyboardInterrupt only hits the main thread).
    """
    # Import lazily so this file can still be used as "audio only" without pulling in the full host stack.
    import logging

    from ..protobuf.generated import sourccey_pb2
    from .sourccey import Sourccey
    from .config_sourccey import SourcceyConfig
    from .sourccey_host import SourcceyHost

    logging.info("Configuring Sourccey")
    robot_config = SourcceyConfig(id="sourccey")
    robot = Sourccey(robot_config)

    logging.info("Connecting Sourccey")
    robot.connect()

    logging.info("Starting Host")
    host_config = SourcceyHostConfig()
    host = SourcceyHost(host_config)

    print("[host] Waiting for commands...")

    last_cmd_time = time.time()
    watchdog_active = False

    try:
        previous_observation = None
        while (not stop_event.is_set()) and (time.time() - last_cmd_time) < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)

                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)
                data = robot.protobuf_converter.protobuf_to_action(robot_action)

                _action_sent = robot.send_action(data)
                robot.update()

                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Host message handling failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                try:
                    robot.stop_base()
                except Exception:
                    pass

            # Send observation (best-effort)
            observation = None
            try:
                observation = robot.get_observation()
            except Exception:
                observation = None

            if observation is None or observation == {}:
                observation = previous_observation
            else:
                previous_observation = observation

            if observation is not None and observation != {}:
                try:
                    robot_state = robot.protobuf_converter.observation_to_protobuf(observation)
                    host.zmq_observation_socket.send(robot_state.SerializeToString(), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                except Exception as e:
                    logging.error(f"Failed to send observation: {e}")

            # Poll for incoming text (only supported in some host variants).
            # On `main`, text/TTS is handled by the `TextTtsGateway` in this file.
            if hasattr(host, "_poll_text_messages"):
                try:
                    host._poll_text_messages()
                except Exception:
                    pass

            # Rate limit
            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))

    finally:
        print("[host] Shutting down...")
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            host.disconnect()
        except Exception:
            pass


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    # Allow setting once via env var so you don't have to pass it every run.
    dev_env = os.environ.get("SOURCCEY_AUDIO_DEVICE")
    p.add_argument("--device", type=int, default=(int(dev_env) if dev_env is not None else None))
    p.add_argument("--sample-rate", type=int, default=int(os.environ.get("SOURCCEY_AUDIO_SR", "16000")))
    p.add_argument("--blocksize", type=int, default=int(os.environ.get("SOURCCEY_AUDIO_BLOCK", "3200")))
    p.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Number of input channels to capture (must match the selected audio device).",
    )
    p.add_argument(
        "--mono-channel",
        type=int,
        default=0,
        help="If channels>1, which channel to use for mono (0-based). Use -1 to average all channels.",
    )

    # DSP
    p.add_argument("--hpf-hz", type=float, default=120.0)
    p.add_argument("--agc-target-rms", type=float, default=220.0)
    p.add_argument("--agc-min-gain", type=float, default=0.2)
    p.add_argument("--agc-max-gain", type=float, default=3.0)

    # Gating
    p.add_argument("--audio-threshold", type=float, default=0.0)
    p.add_argument("--preroll-s", type=float, default=0.25)
    p.add_argument("--hangover-s", type=float, default=0.5)

    # Debug
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--debug-rms-interval-s",
        type=float,
        default=1.0,
        help="When --debug is set, print RMS/AGC info at most once per this many seconds.",
    )
    p.add_argument(
        "--run-host",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run Sourccey host gateway loop (ports 5555-5558) in this process.",
    )
    p.add_argument(
        "--run-tts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a minimal text gateway + TTS (ports 5557/5558) in this process.",
    )

    args = p.parse_args(argv)

    host_cfg = SourcceyHostConfig()
    pub = AudioStreamPublisher(host_cfg.port_zmq_audio)
    q: queue.Queue[bytes] = queue.Queue(maxsize=64)

    hpf = HighPass1(args.hpf_hz, args.sample_rate)
    agc = SmoothAGC(
        target_rms=args.agc_target_rms,
        min_gain=args.agc_min_gain,
        max_gain=args.agc_max_gain,
        sr=args.sample_rate,
        blocksize=args.blocksize,
    )

    def audio_cb(indata, frames, time_info, status):
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    block_s = args.blocksize / float(args.sample_rate)
    preroll_n = max(1, int(args.preroll_s / max(1e-6, block_s)))
    preroll = collections.deque(maxlen=preroll_n)

    streaming = False
    last_voice_ts = 0.0
    last_dbg_ts = 0.0

    try:
        stop_event = threading.Event()
        host_thread: Optional[threading.Thread] = None
        if bool(args.run_host):
            # Non-daemon so we can join cleanly on shutdown (avoids interpreter exiting while the thread is in ZMQ/C++).
            host_thread = threading.Thread(target=_run_host_loop, args=(stop_event,))
            host_thread.start()

        tts = None
        if bool(args.run_tts):
            tts = TextTtsGateway(host_cfg.port_zmq_text_in, host_cfg.port_zmq_text_out)

        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=args.blocksize,
            dtype="int16",
            channels=int(args.channels),
            device=args.device,
            callback=audio_cb,
        ):

            print("[voice] Listening (stateful HPF + smooth AGC, mono output)")

            while True:
                if tts is not None:
                    # Poll TTS/text channel even if audio is quiet.
                    tts.poll_once()

                try:
                    raw = q.get(timeout=0.25)
                except queue.Empty:
                    continue

                # Decode stereo PCM16
                buf = np.frombuffer(raw, dtype=np.int16)
                ch = int(args.channels)
                if ch <= 1:
                    mono = buf
                else:
                    if buf.size % ch != 0:
                        continue
                    frames = buf.reshape(-1, ch)
                    if int(args.mono_channel) == -1:
                        mono = np.mean(frames.astype(np.int32), axis=1).astype(np.int16)
                    else:
                        idx = max(0, min(ch - 1, int(args.mono_channel)))
                        mono = frames[:, idx]

                if mono.size == 0:
                    continue

                # ---- DSP PIPELINE ----
                mono = hpf.process_i16(mono)
                mono = agc.process_i16(mono)

                level = rms_i16(mono)
                now = time.time()
                above = args.audio_threshold <= 0 or level >= args.audio_threshold

                if args.debug:
                    interval = max(0.0, float(args.debug_rms_interval_s))
                    if interval == 0.0 or (now - last_dbg_ts) >= interval:
                        print(f"rms={level:6.1f} gain={agc.gain:4.2f}", file=sys.stderr)
                        last_dbg_ts = now

                mono_bytes = mono.tobytes()
                preroll.append(mono_bytes)

                if above:
                    last_voice_ts = now
                    if not streaming:
                        streaming = True
                        for b in preroll:
                            pub.send(b)
                    pub.send(mono_bytes)
                else:
                    if streaming and (now - last_voice_ts) <= args.hangover_s:
                        pub.send(mono_bytes)
                    else:
                        streaming = False

    except KeyboardInterrupt:
        print("[voice] Stopping...")
        try:
            stop_event.set()
        except Exception:
            pass
        return 0
    finally:
        try:
            stop_event.set()
        except Exception:
            pass
        try:
            if tts is not None:
                tts.close()
        except Exception:
            pass
        # Join host thread briefly to avoid C++/ZMQ teardown races.
        try:
            if host_thread is not None:
                host_thread.join(timeout=2.0)
        except Exception:
            pass
        pub.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
