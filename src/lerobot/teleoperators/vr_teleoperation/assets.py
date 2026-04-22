from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _lerobot_root() -> Path | None:
    try:
        import lerobot
    except Exception:
        return None
    return Path(lerobot.__file__).resolve().parent.parent


def _existing_paths(
    candidates: Iterable[str | Path | None],
    *,
    expect_dir: bool,
) -> list[Path]:
    root = _lerobot_root()
    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        raw_path = Path(candidate)
        path_options = [raw_path]
        if root is not None and not raw_path.is_absolute():
            path_options.append(root / raw_path)
        for path in path_options:
            if path in seen:
                continue
            seen.add(path)
            if expect_dir and path.is_dir():
                resolved.append(path)
            elif not expect_dir and path.is_file():
                resolved.append(path)
    return resolved


@dataclass(frozen=True, slots=True)
class SourcceyTeleopAssetPaths:
    urdf_path: Path | None
    mesh_dir: Path | None
    calibration_path: Path | None


def resolve_sourccey_teleop_assets(
    *,
    urdf_path: str | None,
    mesh_path: str | None,
    calibration_path: str | None,
    arm_side: str,
) -> SourcceyTeleopAssetPaths:
    arm_side = arm_side.lower()
    urdf_candidates = [
        urdf_path,
        "lerobot/robots/sourccey/sourccey/sourccey/model/Arm.urdf",
        "lerobot/robots/sourccey/sourccey_v2beta/model/Arm.urdf",
        "src/lerobot/robots/sourccey/sourccey/sourccey/model/Arm.urdf",
        "src/lerobot/robots/sourccey/sourccey_v2beta/model/Arm.urdf",
    ]
    mesh_candidates = [
        mesh_path,
        "lerobot/robots/sourccey/sourccey/sourccey/model/meshes",
        "lerobot/robots/sourccey/sourccey_v2beta/model/meshes",
        "src/lerobot/robots/sourccey/sourccey/sourccey/model/meshes",
        "src/lerobot/robots/sourccey/sourccey_v2beta/model/meshes",
    ]
    calibration_candidates = [
        calibration_path,
        f"lerobot/robots/sourccey/sourccey/sourccey/defaults/{arm_side}_arm_default_calibration.json",
        f"lerobot/robots/sourccey/sourccey/sourccey/{arm_side}_arm_default_calibration.json",
        f"src/lerobot/robots/sourccey/sourccey/sourccey/defaults/{arm_side}_arm_default_calibration.json",
        f"src/lerobot/robots/sourccey/sourccey/sourccey/{arm_side}_arm_default_calibration.json",
    ]

    urdf_matches = _existing_paths(urdf_candidates, expect_dir=False)
    mesh_matches = _existing_paths(mesh_candidates, expect_dir=True)
    calibration_matches = _existing_paths(calibration_candidates, expect_dir=False)

    return SourcceyTeleopAssetPaths(
        urdf_path=urdf_matches[0] if urdf_matches else None,
        mesh_dir=mesh_matches[0] if mesh_matches else None,
        calibration_path=calibration_matches[0] if calibration_matches else None,
    )
