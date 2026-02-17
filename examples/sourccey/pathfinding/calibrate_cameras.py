import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient


def _load_default_arm_actions() -> dict[str, float]:
    """Load default arm actions from teleop JSONs (arms at side)."""
    try:
        from lerobot.teleoperators.sourccey.sourccey.sourccey_leader import sourccey_leader

        base_dir = Path(sourccey_leader.__file__).parent / "defaults"
        left_path = base_dir / "left_arm_default_action.json"
        right_path = base_dir / "right_arm_default_action.json"

        left = json.loads(left_path.read_text(encoding="utf-8")) if left_path.exists() else {}
        right = json.loads(right_path.read_text(encoding="utf-8")) if right_path.exists() else {}

        action = {f"left_{k}": float(v) for k, v in left.items()}
        action.update({f"right_{k}": float(v) for k, v in right.items()})
        return action
    except Exception:
        return {}


def _parse_args():
    p = argparse.ArgumentParser(description="Stereo camera calibration for Sourccey front cameras")
    p.add_argument("--remote_ip", required=True, help="Raspberry Pi IP running sourccey_host")
    p.add_argument("--board_cols", type=int, default=9, help="Inner corners per row")
    p.add_argument("--board_rows", type=int, default=6, help="Inner corners per column")
    p.add_argument("--square_size_m", type=float, default=0.024, help="Square size in meters")
    p.add_argument("--num_frames", type=int, default=25, help="Number of good stereo pairs to capture")
    p.add_argument("--left_key", default="front_left")
    p.add_argument("--right_key", default="front_right")
    p.add_argument("--output", default="examples/sourccey/pathfinding/camera_calibration.json")
    return p.parse_args()


def _find_corners(gray, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return True, corners


def main():
    args = _parse_args()
    pattern_size = (args.board_cols, args.board_rows)

    objp = np.zeros((args.board_rows * args.board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.board_cols, 0:args.board_rows].T.reshape(-1, 2)
    objp *= args.square_size_m

    objpoints: List[np.ndarray] = []
    imgpoints_l: List[np.ndarray] = []
    imgpoints_r: List[np.ndarray] = []

    robot = SourcceyClient(SourcceyClientConfig(remote_ip=args.remote_ip, id="sourccey"))
    robot.connect()
    arm_action = _load_default_arm_actions()

    try:
        print("Collecting chessboard pairs. Press Ctrl+C to stop early.")
        while len(objpoints) < args.num_frames:
            # Keep arms at side while capturing.
            if arm_action:
                robot.send_action({**arm_action, **{"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}})
            obs = robot.get_observation()
            left = obs.get(args.left_key)
            right = obs.get(args.right_key)
            if left is None or right is None:
                continue

            gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            found_l, corners_l = _find_corners(gray_l, pattern_size)
            found_r, corners_r = _find_corners(gray_r, pattern_size)

            if found_l and found_r:
                objpoints.append(objp.copy())
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)
                print(f"Captured {len(objpoints)}/{args.num_frames}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        robot.disconnect()

    if len(objpoints) < 5:
        raise SystemExit("Not enough valid pairs for calibration.")

    image_size = (gray_l.shape[1], gray_l.shape[0])

    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )

    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags,
    )

    out = {
        "image_size": image_size,
        "left": {
            "K": mtx_l.tolist(),
            "dist": dist_l.tolist(),
            "rms": float(ret_l),
        },
        "right": {
            "K": mtx_r.tolist(),
            "dist": dist_r.tolist(),
            "rms": float(ret_r),
        },
        "stereo": {
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "rms": float(ret_stereo),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote calibration to {out_path}")


if __name__ == "__main__":
    main()
