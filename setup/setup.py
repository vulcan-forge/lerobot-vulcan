#!/usr/bin/env python3
"""
LeRobot Vulcan Setup Script

This script sets up the development environment for the LeRobot Vulcan project.
It checks for required dependencies and installs all necessary packages.

Requirements:
- Python 3.10+
- uv (Python package manager) - optional but recommended
- Git
"""

import argparse
import os
import sys
import subprocess
import platform
import shutil
import stat
from pathlib import Path
from typing import Tuple, Optional

from setup_modules.setup_desktop import install_sourccey_desktop_extras

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color


class SetupScript:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []

    #################################################################
    # Print functions
    #################################################################
    def print_status(self, message: str, color: str = Colors.BLUE):
        """Print a status message with color"""
        print(f"{color}[INFO]{Colors.NC} {message}")

    def print_success(self, message: str):
        """Print a success message"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

    def print_warning(self, message: str):
        """Print a warning message"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
        self.warnings.append(message)

    def print_error(self, message: str):
        """Print an error message"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
        self.errors.append(message)

    #################################################################
    # Check functions
    #################################################################

    def check_command_exists(self, command: str) -> bool:
        """Check if a command exists in the system PATH"""
        return shutil.which(command) is not None

    def get_command_version(self, command: str) -> Optional[str]:
        """Get the version of a command if it exists"""
        try:
            result = subprocess.run([command, "--version"],
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return None

    def get_venv_python_path(self) -> Path:
        """Get the Python interpreter path from the virtual environment"""
        if (platform.system() == "Windows"):
            return self.project_root / ".venv" / "Scripts" / "python.exe"
        else:
            return self.project_root / ".venv" / "bin" / "python"

    def is_macos_or_windows(self) -> bool:
        """Return True when running on macOS or Windows."""
        return platform.system() in {"Darwin", "Windows"}

    def ensure_executable(self, path: Path) -> None:
        """Ensure a file is executable on Unix-like systems."""
        if platform.system() == "Windows":
            return
        try:
            if not path.exists():
                self.print_warning(f"Expected executable not found at {path}")
                return
            mode = path.stat().st_mode
            if mode & 0o111:
                return
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            self.print_success(f"Marked executable: {path}")
        except Exception as e:
            self.print_warning(f"Failed to set executable permissions for {path}: {e}")

    def check_python_version(self) -> bool:
        """Check if Python 3.10+ is installed"""
        self.print_status("Checking Python version...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            self.print_error(f"Python 3.10+ is required, but found {version.major}.{version.minor}")
            self.print_error("Please install Python 3.10 or higher from https://python.org")
            return False

        self.print_success(f"Python {version.major}.{version.minor}.{version.micro} is installed")
        return True

    def check_git(self) -> bool:
        """Check if Git is installed"""
        self.print_status("Checking Git installation...")

        if not self.check_command_exists("git"):
            self.print_error("Git is not installed")
            self.print_error("Please install Git from https://git-scm.com/")
            return False

        version = self.get_command_version("git")
        if version:
            self.print_success(f"Git is installed: {version}")
        else:
            self.print_success("Git is installed")
        return True

    def check_uv(self) -> bool:
        """Check if uv is installed (optional but recommended)"""
        self.print_status("Checking uv installation (optional)...")

        if not self.check_command_exists("uv"):
            self.print_warning("uv is not installed (optional but recommended)")
            self.print_warning("Install uv from https://docs.astral.sh/uv/getting-started/installation/")
            return False

        version = self.get_command_version("uv")
        if version:
            self.print_success(f"uv is installed: {version}")
        else:
            self.print_success("uv is installed")
        return True

    def check_project_structure(self) -> bool:
        """Check if we're in the correct project directory"""
        self.print_status("Checking project structure...")

        required_files = ["pyproject.toml", "src/lerobot/__init__.py"]
        required_dirs = ["src", "src/lerobot"]

        missing_files = []
        missing_dirs = []

        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)

        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)

        if missing_files or missing_dirs:
            self.print_error("Project structure is incomplete:")
            for file_path in missing_files:
                self.print_error(f"  Missing file: {file_path}")
            for dir_path in missing_dirs:
                self.print_error(f"  Missing directory: {dir_path}")
            self.print_error("Please run this script from the root of the lerobot-vulcan project")
            return False

        self.print_success("Project structure is correct")
        return True

    def check_venv_exists(self) -> bool:
        """Check if .venv directory exists"""
        return (self.project_root / '.venv').exists()

    def handle_existing_venv(self) -> bool:
        """Handle existing .venv directory - always delete and recreate"""
        self.print_status("Removing existing venv...")
        try:
            shutil.rmtree(self.project_root / '.venv')
            self.print_success("Existing venv removed")
            return False  # venv no longer exists
        except Exception as e:
            self.print_error(f"Error removing .venv: {e}")
            return True  # keep existing if can't remove

    #################################################################
    # Setup functions
    #################################################################

    def install_uv(self) -> bool:
        """Install UV if not present"""
        self.print_status("Installing uv...")

        try:
            # Check if we're on a Debian-based system
            is_debian = os.path.exists('/etc/debian_version')

            if is_debian:
                self.print_warning("Detected Debian-based system (like Raspberry Pi OS)")
                self.print_warning("Please install the following packages first:")
                self.print_warning("1. sudo apt install python3-full python3-pip pipx")
                self.print_warning("2. pipx install uv")
                self.print_warning("Then run this script again.")
                return False
            else:
                # Install uv using pip
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
                self.print_success("uv installed successfully!")
                self.print_warning("Please close and reopen your terminal, then run this script again")
                self.print_warning("This is needed for uv to be available in your path.")
                return True

        except subprocess.CalledProcessError as e:
            self.print_error(f"Error installing uv: {e}")
            return False

    def setup_python_environment(self) -> bool:
        """Setup Python environment"""
        self.print_status("Setting up Python environment...")

        # ALWAYS remove existing .venv to ensure clean installation
        if self.check_venv_exists():
            self.print_status("Removing existing virtual environment for clean installation...")
            try:
                shutil.rmtree(self.project_root / '.venv')
                self.print_success("Existing .venv removed")
            except Exception as e:
                self.print_error(f"Failed to remove existing .venv: {e}")
                return False

        # Create fresh virtual environment
        try:
            # Check if uv is available
            if self.check_command_exists("uv"):
                self.print_status("Using uv for Python environment setup...")

                # Create virtual environment with Python 3.10
                subprocess.run(["uv", "venv"],
                             check=True, cwd=self.project_root)
                self.print_success("Virtual environment created with uv")

                python_path = self.get_venv_python_path()
                self.ensure_executable(python_path)

                # Install dependencies with sourccey extras. On macOS/Windows we gracefully
                # fall back if a stale resolver path still tries to pull vosk.
                install_result = subprocess.run(
                    ["uv", "pip", "install", "--python", str(python_path), "-e", ".[sourccey]"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )

                if install_result.returncode == 0:
                    self.print_success("LeRobot sourccey dependencies installed with uv")
                else:
                    resolver_output = f"{install_result.stdout or ''}\n{install_result.stderr or ''}".lower()
                    if self.is_macos_or_windows() and "vosk" in resolver_output:
                        self.print_warning(
                            "vosk is not supported on this platform. "
                            "Falling back to install without sourccey extra audio dependency."
                        )
                        subprocess.run(
                            ["uv", "pip", "install", "--python", str(python_path), "-e", "."],
                            check=True,
                            cwd=self.project_root,
                        )
                        self.print_success("Installed core LeRobot dependencies with uv")
                    else:
                        raise subprocess.CalledProcessError(
                            install_result.returncode,
                            ["uv", "pip", "install", "--python", str(python_path), "-e", ".[sourccey]"],
                            output=install_result.stdout,
                            stderr=install_result.stderr,
                        )

            else:
                self.print_error("uv not available")
                self.print_error("Please install uv from https://docs.astral.sh/uv/getting-started/installation/")
                self.print_error("Then run this script again.")
                return False

            return True

        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to setup Python environment: {e}")
            if getattr(e, "stderr", None):
                self.print_error(f"Installer stderr: {e.stderr.strip()}")
            return False

    def setup_desktop_extras(self) -> bool:
        """Install Sourccey desktop extras."""
        return install_sourccey_desktop_extras(
            project_root=self.project_root,
            print_status=self.print_status,
            print_success=self.print_success,
            print_error=self.print_error,
        )

    def fix_final_ownership(self) -> bool:
        """Restore project directory ownership to the normal user after setup."""
        if platform.system() == "Windows":
            return True  # No-op on Windows

        # If setup is not running with elevated privileges, ownership should already be correct.
        sudo_user = os.environ.get("SUDO_USER")
        is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        if not sudo_user and not is_root:
            self.print_status("Ownership restore skipped (setup not running with sudo/root).")
            return True

        self.print_status("Restoring project directory ownership to normal user...")

        user = sudo_user or os.environ.get("USER", "sourccey")
        group = user
        project_root_str = str(self.project_root)

        try:
            # Resolve user's primary group (macOS users are typically in 'staff', not '<user>').
            group_result = subprocess.run(
                ["id", "-gn", user],
                capture_output=True,
                text=True,
                check=False
            )
            if group_result.returncode == 0 and group_result.stdout.strip():
                group = group_result.stdout.strip()
            elif platform.system() == "Darwin":
                group = "staff"

            cmd_prefix = [] if is_root else ["sudo"]

            # Linux-only immutable flag cleanup
            if platform.system() == "Linux" and shutil.which("chattr"):
                subprocess.run(cmd_prefix + ["chattr", "-i", "-R", project_root_str], check=False)

            # Restore ownership (macOS can error if group name is invalid)
            chown_cmd = cmd_prefix + ["chown", "-R", f"{user}:{group}", project_root_str]
            chown_result = subprocess.run(chown_cmd, capture_output=True, text=True)
            if chown_result.returncode != 0:
                stderr = (chown_result.stderr or "").strip()
                if platform.system() == "Darwin" and "illegal group name" in stderr:
                    fallback_group = "staff"
                    if group != fallback_group:
                        subprocess.run(
                            cmd_prefix + ["chown", "-R", f"{user}:{fallback_group}", project_root_str],
                            check=True,
                        )
                    else:
                        subprocess.run(cmd_prefix + ["chown", "-R", user, project_root_str], check=True)
                else:
                    raise subprocess.CalledProcessError(
                        chown_result.returncode, chown_cmd, output=chown_result.stdout, stderr=stderr
                    )

            # Restore permissions
            subprocess.run(cmd_prefix + ["chmod", "-R", "u+rwX", project_root_str], check=True)

            self.print_success(f"Ownership successfully restored to {user}.")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to restore ownership: {e}")
            return False
        except Exception as e:
            self.print_warning(f"Could not restore ownership: {e}")
            return True  # Continue anyway

    def cleanup_evdev_conflict(self) -> bool:
        """Clean up evdev package conflicts"""
        self.print_status("Cleaning up evdev package conflicts...")

        try:
            # Try to uninstall evdev via pip first
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "evdev", "-y"],
                        capture_output=True, text=True)

            # Also try with uv
            if self.check_command_exists("uv"):
                subprocess.run(["uv", "pip", "uninstall", "evdev"],
                            capture_output=True, text=True)

            self.print_success("evdev cleanup completed")
            return True

        except Exception as e:
            self.print_warning(f"evdev cleanup had issues: {e}")
            return True  # Continue anyway

    def _venv_has_module(self, module_name: str) -> bool:
        """Check whether a module is importable in the project virtual environment."""
        python_path = self.get_venv_python_path()
        if not python_path.exists():
            return False

        result = subprocess.run(
            [str(python_path), "-c", f"import {module_name}"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )
        return result.returncode == 0

    def ensure_protobuf_dependencies(self) -> bool:
        """Ensure protobuf compiler dependencies are available in the project venv."""
        python_path = self.get_venv_python_path()
        if not python_path.exists():
            self.print_error(f"Virtual environment Python not found at {python_path}")
            return False

        if self._venv_has_module("grpc_tools"):
            return True

        self.print_status("grpc_tools not found in venv. Installing grpcio-tools...")
        try:
            if self.check_command_exists("uv"):
                subprocess.run(
                    ["uv", "pip", "install", "--python", str(python_path), "grpcio-tools"],
                    check=True,
                    cwd=self.project_root,
                )
                self.print_success("Installed grpcio-tools with uv")
            else:
                self.print_warning("uv not available, falling back to pip for grpcio-tools installation")
                subprocess.run(
                    [str(python_path), "-m", "pip", "install", "grpcio-tools"],
                    check=True,
                    cwd=self.project_root,
                )
                self.print_success("Installed grpcio-tools with pip")

            if not self._venv_has_module("grpc_tools"):
                self.print_error("grpcio-tools installed, but grpc_tools is still not importable.")
                return False

            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install grpcio-tools: {e}")
            return False

    def compile_profobufs(self):
        """Compile Sourccey protobuf"""
        # Clean up evdev conflicts first
        self.cleanup_evdev_conflict()

        try:
            # Avoid `uv run` here: it can trigger a full dependency resolution/split-environment solve.
            # We already created the venv earlier, so we can run the compiler script directly.
            python_path = self.get_venv_python_path()

            if not python_path.exists():
                self.print_error(f"Virtual environment Python not found at {python_path}")
                self.print_error("Please run the setup script first to create the virtual environment")
                return False

            compile_script = (
                self.project_root
                / "src"
                / "lerobot"
                / "robots"
                / "sourccey"
                / "sourccey"
                / "protobuf"
                / "compile.py"
            )

            if not compile_script.exists():
                self.print_error(f"Sourccey protobuf compiler not found at {compile_script}")
                return False

            if not self.ensure_protobuf_dependencies():
                self.print_error("Missing protobuf dependencies for compilation")
                return False

            self.print_status("Compiling Sourccey protobuf...")
            subprocess.run([str(python_path), str(compile_script)], check=True, cwd=self.project_root)

            self.print_success("Sourccey protobuf compiled successfully!")
            return True

        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to compile Sourccey protobuf: {e}")
            return False

    #################################################################
    # Summary and main functions
    #################################################################

    def print_summary(self):
        """Print setup summary"""
        self.print_header("SETUP SUMMARY")

        if self.errors:
            self.print_error(f"Setup completed with {len(self.errors)} error(s):")
            for error in self.errors:
                print(f"  • {error}")
            print()

        if self.warnings:
            self.print_warning(f"Setup completed with {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()

        if not self.errors:
            self.print_success("Setup completed successfully! 🎉")
            print()
            self.print_status("To finish setup, please:")
            self.print_status("1. Close and reopen your terminal")
            self.print_status("2. Navigate to this directory")
            self.print_status("3. Activate the virtual environment:")
            if platform.system() == "Windows":
                self.print_status("source .venv\\Scripts\\activate")
            else:
                self.print_status("source .venv/bin/activate")
            print()

    def print_header(self, message: str):
        """Print a header message"""
        print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
        print(f"{Colors.CYAN}{message.center(60)}{Colors.NC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}\n")

    def run(self, desktop: bool = False) -> bool:
        """Run the complete setup process"""
        self.print_header("LEROBOT VULCAN SETUP")

        # Check system requirements
        self.print_header("CHECKING SYSTEM REQUIREMENTS")

        checks = [
            self.check_python_version(),
            self.check_git(),
            self.check_project_structure(),
        ]

        # Check uv (optional)
        uv_available = self.check_uv()
        if not uv_available:
            self.print_status("uv not found, attempting to install automatically...")
            if not self.install_uv():
                self.print_warning("Failed to install uv, will use pip instead")
            else:
                # Exit so user can restart with uv in PATH
                return True

        if not all(checks):
            self.print_error("System requirements check failed. Please install missing dependencies.")
            return False

        # Setup project
        self.print_header("SETTING UP PROJECT")

        setup_steps = [self.setup_python_environment()]
        if desktop:
            setup_steps.append(self.setup_desktop_extras())
        setup_steps.append(self.compile_profobufs())

        if not all(setup_steps):
            self.print_error("Project setup failed.")
            return False

        # Final ownership fix (Unix systems only)
        if not self.fix_final_ownership():
            self.print_error("Failed to restore project ownership.")
            return False

        self.print_summary()
        return len(self.errors) == 0

################################################################
# Main function
################################################################
def setup(desktop: bool = False):
    """Setup the project"""
    setup = SetupScript()
    success = setup.run(desktop=desktop)
    return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LeRobot Vulcan Setup")
    parser.add_argument(
        "--desktop",
        action="store_true",
        default=False,
        help="Install Sourccey desktop extras (sourccey-desktop).",
    )
    args = parser.parse_args()

    success = setup(desktop=args.desktop)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
