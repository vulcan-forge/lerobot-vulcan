"""Desktop extras setup helpers for lerobot-vulcan."""

from pathlib import Path
import os
import shutil
import subprocess
from typing import Callable

StatusFn = Callable[[str], None]


def install_sourccey_desktop_extras(
    project_root: Path,
    print_status: StatusFn,
    print_success: StatusFn,
    print_error: StatusFn,
) -> bool:
    """Install the Sourccey desktop extras with uv."""
    if shutil.which("uv") is None:
        print_error("uv is not installed; cannot install Sourccey desktop extras.")
        return False

    print_status("Installing Sourccey desktop extras with uv...")
    try:
        subprocess.run(
            ["uv", "pip", "install", "-e", ".[sourccey-desktop, xvla]"],
            check=True,
            cwd=project_root,
        )
        print_success("Sourccey desktop extras installed.")
        if not install_rerun_viewer(project_root, print_status, print_success, print_error):
            return False
        return True
    except subprocess.CalledProcessError as exc:
        print_error(f"Failed to install Sourccey desktop extras: {exc}")
        return False


def install_rerun_viewer(
    project_root: Path,
    print_status: StatusFn,
    print_success: StatusFn,
    print_error: StatusFn,
) -> bool:
    """Ensure the Rerun viewer is available in the uv venv."""
    if shutil.which("uv") is None:
        print_error("uv is not installed; cannot install Rerun viewer.")
        return False

    venv_bin = project_root / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    rerun_bin = venv_bin / ("rerun.exe" if os.name == "nt" else "rerun")

    if rerun_bin.exists():
        print_success(f"Rerun viewer already available at {rerun_bin}")
        return True

    print_status("Installing Rerun viewer with uv (rerun-sdk==0.26.2)...")
    try:
        subprocess.run(
            ["uv", "pip", "install", "rerun-sdk==0.26.2"],
            check=True,
            cwd=project_root,
        )
    except subprocess.CalledProcessError as exc:
        print_error(f"Failed to install Rerun viewer: {exc}")
        return False

    if rerun_bin.exists():
        print_success(f"Rerun viewer installed at {rerun_bin}")
        return True

    print_error(
        "Rerun viewer install completed but binary was not found in the venv. "
        "Check the uv install logs for errors."
    )
    return False
