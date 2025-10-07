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

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Tuple, Optional

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
        """Handle existing .venv directory"""
        while True:
            response = input("\n.venv already exists. What would you like to do?\n"
                            "1: Skip venv creation (keep existing)\n"
                            "2: Delete and recreate venv\n"
                            "3: Exit\n"
                            "Choose (1-3): ").strip()

            if response == '1':
                self.print_status("Keeping existing venv...")
                return True
            elif response == '2':
                self.print_status("Removing existing venv...")
                try:
                    shutil.rmtree(self.project_root / '.venv')
                    self.print_success("Existing venv removed")
                    return False  # venv no longer exists
                except Exception as e:
                    self.print_error(f"Error removing .venv: {e}")
                    return True  # keep existing if can't remove
            elif response == '3':
                self.print_status("Exiting setup...")
                sys.exit(0)
            else:
                self.print_error("Invalid choice. Please choose 1, 2, or 3.")

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

        # Check if venv exists and handle accordingly
        if self.check_venv_exists():
            self.print_success("Using existing virtual environment")
            return True

        try:
            # Check if uv is available
            if self.check_command_exists("uv"):
                self.print_status("Using uv for Python environment setup...")

                # Create virtual environment with Python 3.10
                subprocess.run(["uv", "venv"],
                             check=True, cwd=self.project_root)
                self.print_success("Virtual environment created with uv")

                # Install dependencies with feetech and smolvla extras
                subprocess.run(["uv", "pip", "install", "-e", ".[feetech,smolvla,sourccey]"],
                             check=True, cwd=self.project_root)
                self.print_success("LeRobot dependencies installed with uv")

            else:
                self.print_warning("uv not available, using pip...")

                # Create virtual environment with venv
                venv_path = self.project_root / ".venv"
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
                self.print_success("Virtual environment created with venv")

                # Install dependencies
                pip_path = venv_path / ("Scripts" if platform.system() == "Windows" else "bin") / "pip"
                subprocess.run([str(pip_path), "install", "-e", ".[feetech,smolvla,sourccey]"], check=True)
                self.print_success("LeRobot dependencies installed with pip")

            return True

        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to setup Python environment: {e}")
            return False

    def compile_profobufs(self):
        """Compile Sourccey protobuf"""
        try:
            # Use the Python interpreter from the virtual environment
            python_path = self.get_venv_python_path()

            if not python_path.exists():
                self.print_error(f"Virtual environment Python not found at {python_path}")
                self.print_error("Please run the setup script first to create the virtual environment")
                return False

            # Run the protobuf compilation using subprocess
            self.print_status("Compiling Sourccey protobuf...")
            result = subprocess.run([
                str(python_path), "-c",
                "from lerobot.robots.sourccey.sourccey.protobuf.compile import compile_sourccey_protobuf; compile_sourccey_protobuf()"
            ], check=True, cwd=self.project_root)

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

    def run(self) -> bool:
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
            install_uv = input("\nWould you like to install uv now? (y/n): ").strip().lower()
            if install_uv in ['y', 'yes']:
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

        setup_steps = [
            self.setup_python_environment(),
            self.compile_profobufs()
        ]

        if not all(setup_steps):
            self.print_error("Project setup failed.")
            return False

        self.print_summary()
        return len(self.errors) == 0

################################################################
# Main function
################################################################
def setup():
    """Setup the project"""
    setup = SetupScript()
    success = setup.run()
    return success

def main():
    """Main entry point"""
    success = setup()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
