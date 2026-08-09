from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Base-only Sourccey teleop helper that ignores arm teleoperation hardware.'
    )
    parser.add_argument('--remote-ip', type=str, default='192.168.1.211', help='IP address of the Sourccey host.')
    parser.add_argument('--robot-id', type=str, default='sourccey', help='Robot id for the Sourccey client.')
    parser.add_argument('--camera-key', type=str, default='front_left', help='Observation camera to display in the teleop preview window.')
    parser.add_argument('--window-name', type=str, default='Sourccey Base Teleop', help='OpenCV window title.')
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Disable OpenCV preview and read keys directly from the terminal.',
    )
    return parser


def _terminal_key() -> str | None:
    if msvcrt is None:
        return None
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in ('\x00', '\xe0'):
        if msvcrt.kbhit():
            msvcrt.getwch()
        return None
    if key == '\r':
        return None
    if key == '\x1b':
        return 'esc'
    return key.lower()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = SourcceyClientConfig(
        id=args.robot_id,
        remote_ip=args.remote_ip,
        host_session_mode="teleop_full",
    )

    robot = SourcceyClient(cfg)
    robot.connect()
    robot.untorque_left_active = True
    robot.untorque_right_active = True

    use_preview = not args.headless
    if use_preview:
        try:
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(args.window_name, 960, 720)
        except cv2.error:
            use_preview = False
            print('OpenCV preview is unavailable in this environment; falling back to headless terminal mode.')

    print('Base-only teleop started.')
    if use_preview:
        print('Click the preview window so it has focus before driving.')
    else:
        print('Terminal mode active; keep this terminal focused while driving.')
    print('Keys: w/s forward/back, a/d strafe, z/x rotate, q/e z up/down, r/f speed, space or esc quit')

    try:
        while True:
            obs = robot.get_observation()
            frame = obs.get(args.camera_key)
            if not isinstance(frame, np.ndarray):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)

            key_name: str | None = None
            if use_preview:
                view = frame.copy()
                lines = [
                    'W/S forward/back   A/D strafe   Z/X rotate',
                    'Q/E z up/down   R/F speed   SPACE or ESC quit',
                    f'Speed level: {robot.speed_index + 1}/3',
                ]
                for idx, text in enumerate(lines):
                    cv2.putText(
                        view,
                        text,
                        (8, 24 + idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow(args.window_name, view)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    key_name = 'esc'
                elif key == ord(' '):
                    key_name = 'space'
                elif key != 255:
                    key_name = chr(key).lower()
            else:
                key_name = _terminal_key()
                time.sleep(0.03)

            pressed: list[str] = []

            if key_name in ('esc', 'space'):
                break

            if key_name:
                if key_name in (cfg.teleop_keys['speed_up'], cfg.teleop_keys['speed_down']):
                    robot.on_key_down(key_name)
                if key_name in cfg.teleop_keys.values():
                    pressed = [key_name]

            z_obs = float(obs.get('z.pos', robot._z_pos_cmd))
            action = robot._from_keyboard_to_base_action(
                np.array(pressed, dtype=object),
                z_obs_pos=z_obs,
            )
            action['untorque_left'] = True
            action['untorque_right'] = True
            robot.send_action(action)
    finally:
        try:
            robot.send_action(
                {
                    'x.vel': 0.0,
                    'y.vel': 0.0,
                    'theta.vel': 0.0,
                    'z.pos': float(robot._z_pos_cmd),
                    'untorque_left': True,
                    'untorque_right': True,
                }
            )
        except Exception:
            pass
        robot.disconnect()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

