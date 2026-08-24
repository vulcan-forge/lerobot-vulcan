#!/usr/bin/env python3
"""Standalone check for the underside (bottom) camera - run this ON THE PI.

The bottom camera is not in `sourccey_cameras_config()` yet, so the host never
opens it and it never reaches the observation stream. This script goes straight
to the V4L2 device instead, so you can confirm the camera is physically
connected and streaming before any software update wires it into the host.

    # find it and check it (auto-detects any camera that is not one of the four
    # configured /dev/camera* devices)
    uv run python scripts/sourccey_check_bottom_camera.py

    # you already know the device
    uv run python scripts/sourccey_check_bottom_camera.py --device /dev/video4

    # just list what V4L2 devices exist and which are already spoken for
    uv run python scripts/sourccey_check_bottom_camera.py --list

    # watch it live in a browser on your PC (the Pi has no display)
    uv run python scripts/sourccey_check_bottom_camera.py --view mjpeg

Every run saves a frame so you can eyeball what the camera actually sees.
Exit code is 0 only if a device opened and produced live, non-blank, non-frozen
frames.
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# The four cameras the robot already claims; whatever is left over on the USB
# bus is the interesting one.
KNOWN_CAMERA_LINKS = (
    "/dev/cameraFrontLeft",
    "/dev/cameraFrontRight",
    "/dev/cameraWristLeft",
    "/dev/cameraWristRight",
)
# A frame flatter than this is a blank sensor output, not a picture.
BLANK_FRAME_STD = 1.5


def known_device_nodes() -> dict[str, str]:
    """Map each configured camera symlink to the /dev/videoN it resolves to."""
    resolved = {}
    for link in KNOWN_CAMERA_LINKS:
        if os.path.exists(link):
            resolved[link] = os.path.realpath(link)
    return resolved


def list_devices() -> list[str]:
    """Print every V4L2 node and say which are already claimed by the robot."""
    known = known_device_nodes()
    claimed = set(known.values())
    nodes = sorted(glob.glob("/dev/video*"))

    print("[bottom] configured cameras:")
    for link in KNOWN_CAMERA_LINKS:
        target = known.get(link)
        print(f"[bottom]   {link:<24} -> {target or 'MISSING'}")

    print("[bottom] V4L2 nodes on this Pi:")
    unclaimed = []
    for node in nodes:
        tag = "claimed by the robot" if node in claimed else "unclaimed"
        print(f"[bottom]   {node:<16} {tag}")
        if node not in claimed:
            unclaimed.append(node)
    return unclaimed


def probe(device: str, args: argparse.Namespace) -> cv2.VideoCapture | None:
    """Open one device and return the capture if it yields a real frame."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Some UVC cameras hand back one empty frame while the stream spins up.
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.1)
    cap.release()
    return None


def find_bottom_camera(args: argparse.Namespace) -> tuple[str, cv2.VideoCapture] | None:
    """Try every unclaimed V4L2 node and return the first that streams."""
    unclaimed = list_devices()
    if not unclaimed:
        print("[bottom] no unclaimed V4L2 node - the bottom camera is not enumerating")
        return None

    print(f"[bottom] probing {len(unclaimed)} unclaimed node(s)...")
    for node in unclaimed:
        cap = probe(node, args)
        if cap is not None:
            print(f"[bottom] {node} streams - treating it as the bottom camera")
            return node, cap
        print(f"[bottom] {node} did not produce a frame")
    return None


class MjpegServer:
    """Minimal MJPEG view so a headless Pi can still be watched from your PC."""

    def __init__(self, port: int, title: str) -> None:
        self.port = int(port)
        self._title = title
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), self._handler())
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
                if self.path in ("/", "/index.html"):
                    page = (
                        f"<html><head><title>{server._title}</title></head>"
                        "<body style='margin:0;background:#111;color:#ddd;"
                        "font-family:sans-serif;text-align:center'>"
                        f"<h3>{server._title}</h3>"
                        "<img src='/stream.mjpg' style='max-width:100%'></body></html>"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while True:
                        with server._lock:
                            jpeg = server._jpeg
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

        return Handler

    def publish(self, frame: np.ndarray) -> None:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._lock:
                self._jpeg = buffer.tobytes()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def gui_available() -> bool:
    """True only if this OpenCV build can actually open a window.

    This repo pins `opencv-python-headless`, which ships without highgui, so
    `imshow` raises rather than drawing. Probe with a real window instead of
    assuming a desktop means a usable GUI.
    """
    try:
        cv2.namedWindow("__sourccey_gui_probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__sourccey_gui_probe__")
        return True
    except Exception:  # noqa: BLE001 - any failure means no usable GUI
        return False


def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "<pi-ip>"
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the underside camera directly on the Pi (bypasses sourccey_host.py)."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="V4L2 device to check (e.g. /dev/video4). Omit to auto-detect.",
    )
    parser.add_argument("--list", action="store_true", help="Only list V4L2 devices, then exit.")
    parser.add_argument("--width", type=int, default=320, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=240, help="Requested frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested frame rate.")
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="Pixel format; MJPG is what this camera needs for 30 FPS. Pass '' to let it pick.",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="How long to sample.")
    parser.add_argument(
        "--view",
        choices=("none", "mjpeg", "window"),
        default="none",
        help="none = probe and save a frame; mjpeg = watch from your PC in a browser.",
    )
    parser.add_argument("--http-port", type=int, default=8093, help="Port for the MJPEG view.")
    parser.add_argument(
        "--save-dir", default="camera_health", help="Where to write the captured frame."
    )
    args = parser.parse_args()

    if args.view == "window" and not gui_available():
        print("[bottom] FAIL  --view window was requested but this OpenCV build has no GUI "
              "support")
        print("[bottom]       (the repo pins opencv-python-headless, so cv2.imshow cannot draw)")
        print("[bottom]       use --view mjpeg to watch it in a browser instead")
        return 1

    if args.list:
        list_devices()
        return 0

    print("[bottom] ---- bottom camera check ----")
    if args.device:
        if not os.path.exists(args.device):
            print(f"[bottom] FAIL  no such device: {args.device}")
            print("[bottom]       run with --list to see what V4L2 nodes exist")
            return 1
        cap = probe(args.device, args)
        if cap is None:
            print(f"[bottom] FAIL  {args.device} opened but produced no frame "
                  "(is another process using it?)")
            return 1
        device = args.device
        print(f"[bottom] {device} streams")
    else:
        found = find_bottom_camera(args)
        if found is None:
            print("[bottom] FAIL  no working unclaimed camera found")
            print("[bottom]       check the USB cable/hub, then `lsusb` and "
                  "`dmesg -T | tail -30` for enumeration errors")
            return 1
        device, cap = found

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip()
    print(f"[bottom] negotiated {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} "
          f"fps {fourcc or '????'}")

    server: MjpegServer | None = None
    if args.view == "mjpeg":
        server = MjpegServer(args.http_port, f"Sourccey bottom camera ({device})")
        server.start()
        print(f"[bottom] open http://{lan_ip()}:{args.http_port}/ on your PC to watch it")
    elif args.view == "window":
        print("[bottom] press 'q' in the preview window to stop")

    frames = failures = blank = frozen = 0
    previous_checksum = None
    last_frame = None
    stamps: list[float] = []
    start = time.monotonic()
    try:
        while (time.monotonic() - start) < args.seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                time.sleep(0.02)
                continue
            frames += 1
            stamps.append(time.monotonic())
            last_frame = frame
            checksum = int(frame[::8, ::8].sum())
            if previous_checksum is not None and checksum == previous_checksum:
                frozen += 1
            previous_checksum = checksum
            if float(frame.std()) < BLANK_FRAME_STD:
                blank += 1
            if server is not None:
                server.publish(frame)
            elif args.view == "window":
                cv2.imshow("bottom camera", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        if server is not None:
            server.stop()
        if args.view == "window":
            # Never let teardown raise and mask the real error.
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    print(f"[bottom] {frames} frames at {fps:.1f} fps ({failures} dropped reads)")

    if last_frame is not None and args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "bottom.jpg"
        cv2.imwrite(str(path), last_frame)
        print(f"[bottom] saved {path} - open it to confirm the camera sees the floor")

    print()
    if frames == 0:
        print("[bottom] FAIL  the device opened but delivered no frames")
        return 1
    if blank >= frames:
        print("[bottom] FAIL  every frame is blank (black/flat) - no image data")
        return 1
    if frames > 10 and frozen > 0.9 * frames:
        print(f"[bottom] FAIL  feed is frozen ({frozen}/{frames} identical frames)")
        return 1
    if fps < 0.5 * args.fps:
        print(f"[bottom] WARN  only {fps:.1f} fps (asked for {args.fps}); USB bandwidth or power")
    print(f"[bottom] PASS  the bottom camera is connected and streaming on {device}")
    print(f"[bottom]       when you wire it into the host, use: index_or_path=\"{device}\"")
    print("[bottom]       (a /dev/v4l/by-path/... path or a udev name survives reboots better "
          "than /dev/videoN)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
