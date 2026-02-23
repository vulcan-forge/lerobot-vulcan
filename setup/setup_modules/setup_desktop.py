"""Desktop extras setup helpers for lerobot-vulcan."""

from pathlib import Path
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
            ["uv", "pip", "install", "-e", ".[sourccey-desktop]"],
            check=True,
            cwd=project_root,
        )
        print_success("Sourccey desktop extras installed.")
        return True
    except subprocess.CalledProcessError as exc:
        print_error(f"Failed to install Sourccey desktop extras: {exc}")
        return False
