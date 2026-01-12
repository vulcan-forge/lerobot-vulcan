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
import time
from typing import Optional

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

            # Poll for incoming text -> speak + emit TTS events
            host._poll_text_messages()

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
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--blocksize", type=int, default=3200)
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
        "--run-host",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run Sourccey host gateway loop (ports 5555-5558) in this process.",
    )

    args = p.parse_args(argv)

    pub = AudioStreamPublisher(SourcceyHostConfig().port_zmq_audio)
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

    try:
        stop_event = threading.Event()
        host_thread: Optional[threading.Thread] = None
        if bool(args.run_host):
            host_thread = threading.Thread(target=_run_host_loop, args=(stop_event,), daemon=True)
            host_thread.start()

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
                    print(f"rms={level:6.1f} gain={agc.gain:4.2f}", file=sys.stderr)

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
        pub.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
