#!/usr/bin/env python3
"""Find and check the underside (bottom) camera - run this ON THE PI.

The bottom camera is not in `sourccey_cameras_config()` yet, so the host never
opens it and it never reaches the observation stream. This script goes straight
to the V4L2 device instead.

Identifying it is the hard part, because a Pi 5 exposes ~30 `/dev/videoN` nodes
and most are internal ISP/codec devices, not cameras. Two ways to cut through
that:

    # 1) unplug test - the reliable one. Lists USB cameras, asks you to unplug
    #    the bottom camera, sees which one vanished, asks you to plug it back in,
    #    then checks it and prints its STABLE device path.
    uv run python scripts/sourccey_check_bottom_camera.py --identify

    # 2) auto - assumes the only USB camera that is not one of the four
    #    configured /dev/camera* devices is the bottom one.
    uv run python scripts/sourccey_check_bottom_camera.py

    # you already know the device
    uv run python scripts/sourccey_check_bottom_camera.py --device /dev/video8

    # what USB cameras exist and which are spoken for
    uv run python scripts/sourccey_check_bottom_camera.py --list

    # watch it live in a browser on your PC (the Pi is headless)
    uv run python scripts/sourccey_check_bottom_camera.py --view mjpeg

Only real USB capture nodes are ever opened: candidates are filtered by their
sysfs device path (must sit under a USB device) and by `index == 0` (a UVC camera
also exposes an index-1 metadata node that blocks for 10s per read attempt).
That is what made blind probing take minutes.

Every run saves a frame so you can eyeball what the camera actually sees.
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

V4L_SYSFS = "/sys/class/video4linux"
# The four cameras the robot already claims; whatever USB camera is left over is
# the interesting one.
KNOWN_CAMERA_LINKS = (
    "/dev/cameraFrontLeft",
    "/dev/cameraFrontRight",
    "/dev/cameraWristLeft",
    "/dev/cameraWristRight",
)
# A frame flatter than this is a blank sensor output, not a picture.
BLANK_FRAME_STD = 1.5


@dataclass
class VideoDevice:
    """One /dev/videoN node, with the sysfs facts needed to judge it."""

    node: str
    name: str
    is_usb: bool
    index: int
    usb_port: str | None
    by_path: str | None
    by_id: str | None

    @property
    def is_capture(self) -> bool:
        """True for the node that actually delivers frames.

        A UVC camera registers two nodes: index 0 is video capture, index 1 is
        the metadata stream. Opening the metadata node succeeds and then blocks
        in select() until it times out, which is what made the old blind probe
        crawl.
        """
        return self.is_usb and self.index == 0

    @property
    def stable_path(self) -> str:
        """The path worth writing into a config: /dev/videoN moves on re-plug."""
        return self.by_path or self.by_id or self.node

    def describe(self) -> str:
        port = f" port={self.usb_port}" if self.usb_port else ""
        return f"{self.node:<14} {self.name[:34]:<34}{port}"


def _read_sysfs(node_name: str, attr: str) -> str:
    try:
        with open(os.path.join(V4L_SYSFS, node_name, attr)) as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _symlink_map(directory: str) -> dict[str, str]:
    """Map each /dev/videoN to a stable udev symlink pointing at it."""
    mapping: dict[str, str] = {}
    for link in glob.glob(os.path.join(directory, "*")):
        try:
            mapping.setdefault(os.path.realpath(link), link)
        except OSError:
            continue
    return mapping


def enumerate_devices() -> list[VideoDevice]:
    """Describe every V4L2 node on this machine."""
    by_path = _symlink_map("/dev/v4l/by-path")
    by_id = _symlink_map("/dev/v4l/by-id")

    devices: list[VideoDevice] = []
    for sysfs_entry in sorted(glob.glob(os.path.join(V4L_SYSFS, "video*"))):
        node_name = os.path.basename(sysfs_entry)
        node = f"/dev/{node_name}"
        if not os.path.exists(node):
            continue
        # The `device` symlink points into the bus the node hangs off; a USB
        # camera resolves to something containing /usb, while the Pi's ISP and
        # codec nodes resolve to platform devices.
        device_link = os.path.join(sysfs_entry, "device")
        real = os.path.realpath(device_link) if os.path.exists(device_link) else ""
        is_usb = "/usb" in real
        usb_port = os.path.basename(real).split(":")[0] if is_usb else None
        index_text = _read_sysfs(node_name, "index")
        devices.append(
            VideoDevice(
                node=node,
                name=_read_sysfs(node_name, "name") or "(unknown)",
                is_usb=is_usb,
                index=int(index_text) if index_text.isdigit() else -1,
                usb_port=usb_port,
                by_path=by_path.get(node),
                by_id=by_id.get(node),
            )
        )
    return devices


def claimed_nodes() -> dict[str, str]:
    """Map each /dev/videoN the robot already uses to its configured name."""
    claimed = {}
    for link in KNOWN_CAMERA_LINKS:
        if os.path.exists(link):
            claimed[os.path.realpath(link)] = link
    return claimed


def usb_cameras() -> list[VideoDevice]:
    return [d for d in enumerate_devices() if d.is_capture]


def print_inventory() -> list[VideoDevice]:
    """Show the USB cameras and who owns them; return the unclaimed ones."""
    claimed = claimed_nodes()
    cameras = usb_cameras()

    print("[bottom] configured cameras:")
    for link in KNOWN_CAMERA_LINKS:
        target = os.path.realpath(link) if os.path.exists(link) else None
        print(f"[bottom]   {link:<24} -> {target or 'MISSING'}")

    print(f"[bottom] USB capture devices found: {len(cameras)}")
    unclaimed = []
    for device in cameras:
        owner = claimed.get(device.node)
        tag = f"claimed by {os.path.basename(owner)}" if owner else "UNCLAIMED"
        print(f"[bottom]   {device.describe()}  {tag}")
        if owner is None:
            unclaimed.append(device)

    total_nodes = len(enumerate_devices())
    print(f"[bottom] (ignored {total_nodes - len(cameras)} non-USB / metadata V4L2 nodes - "
          "those are the Pi's internal ISP and codec devices)")
    return unclaimed


def identify_by_unplug(args: argparse.Namespace) -> VideoDevice | None:
    """Ask the operator to unplug the camera, and see which device disappears."""
    print("[bottom] ---- identify by unplug ----")
    before = {d.node: d for d in usb_cameras()}
    if not before:
        print("[bottom] FAIL  no USB cameras are enumerating at all - check the hub and cabling")
        return None
    print(f"[bottom] {len(before)} USB camera(s) currently connected:")
    for device in before.values():
        print(f"[bottom]   {device.describe()}")

    print()
    input("[bottom] Now UNPLUG the bottom camera, then press Enter... ")

    # udev can take a moment to tear the nodes down after the physical unplug.
    missing: list[VideoDevice] = []
    deadline = time.monotonic() + args.settle_seconds
    while time.monotonic() < deadline:
        current = {d.node for d in usb_cameras()}
        missing = [d for node, d in before.items() if node not in current]
        if missing:
            break
        time.sleep(0.3)

    if not missing:
        print(f"[bottom] FAIL  nothing disappeared after {args.settle_seconds:.0f}s")
        print("[bottom]       either the camera was not unplugged, or it is on a hub that keeps")
        print("[bottom]       the node alive - try unplugging at the camera end")
        return None
    if len(missing) > 1:
        print(f"[bottom] FAIL  {len(missing)} devices disappeared at once: "
              f"{', '.join(d.node for d in missing)}")
        print("[bottom]       unplug ONLY the bottom camera and re-run")
        return None

    found = missing[0]
    print(f"[bottom] {found.node} disappeared - that is the bottom camera")
    if found.usb_port:
        print(f"[bottom] it lives on USB port {found.usb_port}")

    print()
    input("[bottom] Plug it back in, then press Enter... ")

    # The node NUMBER can change across a re-plug, so match on the physical USB
    # port instead - that is the thing that stays put.
    deadline = time.monotonic() + args.settle_seconds
    while time.monotonic() < deadline:
        for device in usb_cameras():
            if found.usb_port and device.usb_port == found.usb_port:
                if device.node != found.node:
                    print(f"[bottom] it came back as {device.node} (the node number changed - "
                          "this is exactly why a by-path name is worth using)")
                else:
                    print(f"[bottom] {device.node} is back")
                return device
            if device.node == found.node:
                print(f"[bottom] {device.node} is back")
                return device
        time.sleep(0.3)

    print(f"[bottom] FAIL  it did not come back within {args.settle_seconds:.0f}s")
    print("[bottom]       re-seat the connector and re-run")
    return None


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


def gui_available() -> bool:
    """True only if this OpenCV build can actually open a window.

    This repo pins `opencv-python-headless`, which ships without highgui, so
    `imshow` raises rather than drawing.
    """
    try:
        cv2.namedWindow("__sourccey_gui_probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__sourccey_gui_probe__")
        return True
    except Exception:  # noqa: BLE001 - any failure means no usable GUI
        return False


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


def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "<pi-ip>"
    finally:
        sock.close()


def stream_check(device: str, args: argparse.Namespace) -> int:
    """Sample the chosen device and judge whether it is really working."""
    cap = probe(device, args)
    if cap is None:
        print(f"[bottom] FAIL  {device} opened but produced no frame "
              "(is another process using it?)")
        return 1

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
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    print(f"[bottom] {frames} frames at {fps:.1f} fps "
          f"({failures} dropped reads, {blank} blank, {frozen} frozen)")

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
    if blank:
        print(f"[bottom] WARN  {blank}/{frames} frames were black - intermittent read failures")
    if fps < 0.5 * args.fps:
        print(f"[bottom] WARN  only {fps:.1f} fps (asked for {args.fps}); USB bandwidth or power")
    print(f"[bottom] PASS  the bottom camera is connected and streaming on {device}")
    return 0


def print_wiring_hints(device: VideoDevice | None, node: str) -> None:
    """Tell the operator exactly what to put in the config when they wire it up."""
    print()
    print("[bottom] ---- to wire it into the host later ----")
    if device is None or device.stable_path == device.node:
        print(f"[bottom] index_or_path=\"{node}\"")
        print("[bottom] NOTE  /dev/videoN numbering moves when devices are re-plugged or")
        print("[bottom]       re-enumerated at boot. Prefer a by-path name if one appears")
        print("[bottom]       under /dev/v4l/by-path/ for this camera.")
        return

    print(f"[bottom] index_or_path=\"{device.stable_path}\"")
    print("[bottom]       (stable across re-plug, unlike the /dev/videoN number)")
    if device.usb_port:
        print()
        print("[bottom] or give it a name like the other four, with a udev rule in")
        print("[bottom] /etc/udev/rules.d/99-sourccey-cameras.rules:")
        print(f"[bottom]   SUBSYSTEM==\"video4linux\", KERNELS==\"{device.usb_port}\", "
              "ATTR{index}==\"0\", SYMLINK+=\"cameraBottom\"")
        print("[bottom] then: sudo udevadm control --reload-rules && sudo udevadm trigger")
        print("[bottom] and use index_or_path=\"/dev/cameraBottom\"")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and check the underside camera on the Pi (bypasses sourccey_host.py)."
    )
    parser.add_argument(
        "--identify",
        action="store_true",
        help="Unplug test: see which USB camera disappears, then check it.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="V4L2 device to check (e.g. /dev/video8). Omit to auto-detect.",
    )
    parser.add_argument("--list", action="store_true", help="Only list USB cameras, then exit.")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=15.0,
        help="How long to wait for udev after an unplug/re-plug during --identify.",
    )
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

    if not os.path.isdir(V4L_SYSFS):
        print(f"[bottom] FAIL  {V4L_SYSFS} does not exist - run this ON THE PI")
        return 1

    if args.view == "window" and not gui_available():
        print("[bottom] FAIL  --view window was requested but this OpenCV build has no GUI "
              "support")
        print("[bottom]       (the repo pins opencv-python-headless, so cv2.imshow cannot draw)")
        print("[bottom]       use --view mjpeg to watch it in a browser instead")
        return 1

    if args.list:
        print_inventory()
        return 0

    print("[bottom] ---- bottom camera check ----")
    device_info: VideoDevice | None = None

    if args.device:
        if not os.path.exists(args.device):
            print(f"[bottom] FAIL  no such device: {args.device}")
            print("[bottom]       run with --list to see the USB cameras on this Pi")
            return 1
        node = args.device
        real = os.path.realpath(node)
        device_info = next((d for d in enumerate_devices() if d.node == real), None)
    elif args.identify:
        device_info = identify_by_unplug(args)
        if device_info is None:
            return 1
        node = device_info.node
    else:
        unclaimed = print_inventory()
        if not unclaimed:
            print()
            print("[bottom] FAIL  every USB camera is already claimed by the robot config")
            print("[bottom]       the bottom camera is not enumerating: check its cable and hub,")
            print("[bottom]       then `lsusb` and `dmesg -T | tail -30`")
            return 1
        if len(unclaimed) > 1:
            print()
            print(f"[bottom] FAIL  {len(unclaimed)} unclaimed USB cameras - cannot tell which is")
            print("[bottom]       the bottom one. Re-run with --identify to unplug-test it.")
            return 1
        device_info = unclaimed[0]
        node = device_info.node
        print()
        print(f"[bottom] exactly one unclaimed USB camera: {node} - treating it as the bottom camera")

    print(f"[bottom] checking {node}")
    result = stream_check(node, args)
    if result == 0:
        print_wiring_hints(device_info, node)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
