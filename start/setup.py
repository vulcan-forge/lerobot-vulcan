import os
import subprocess
import sys
import shutil
from pathlib import Path

def install_uv():
    """Install UV if not present"""
    try:
        subprocess.run(['uv', '--version'], capture_output=True, check=True)
        print("UV already installed!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("UV not found. Installing UV...")
        try:
            # First, check if we're on a Debian-based system
            is_debian = os.path.exists('/etc/debian_version')

            if is_debian:
                print("\nDetected Debian-based system (like Raspberry Pi OS)")
                print("Please install the following packages first:")
                print("1. sudo apt install python3-full python3-pip pipx")
                print("2. pipx install uv")
                print("\nThen run this script again.")
                sys.exit(1)
            else:
                # Original behavior for non-Debian systems
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
                print("\nUV installed successfully!")
                print("\nPlease:")
                print("1. Close and reopen your terminal")
                print("2. Run this script again to complete setup")
                print("This is needed for UV to be available in your path.")
                sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"Error installing UV: {e}")
            return False

def check_venv_exists():
    """Check if .venv directory exists"""
    return Path('.venv').exists()

def handle_existing_venv():
    """Handle existing .venv directory"""
    while True:
        response = input("\n.venv already exists. What would you like to do?\n"
                        "1: Skip venv creation (keep existing)\n"
                        "2: Delete and recreate venv\n"
                        "3: Exit\n"
                        "Choose (1-3): ").strip()

        if response == '1':
            print("Keeping existing venv...")
            return True
        elif response == '2':
            print("Removing existing venv...")
            try:
                shutil.rmtree('.venv')
                return False  # venv no longer exists
            except Exception as e:
                print(f"Error removing .venv: {e}")
                return True  # keep existing if can't remove
        elif response == '3':
            print("Exiting setup...")
            sys.exit(0)
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")

def setup_venv():
    """Setup virtual environment using UV"""
    print("Setting up virtual environment...")

    # Check if venv exists and handle accordingly
    if check_venv_exists():
        if handle_existing_venv():
            return True  # User chose to keep existing venv

    try:
        # Get parent directory of where setup.py is located
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Create virtual environment with Python 3.10
        subprocess.run(["uv", "venv", ".venv", "-p", "3.10"], check=True, cwd=parent_dir)

        # Install dependencies with feetech and smolvla extras
        # The base dependencies are included automatically
        subprocess.run(["uv", "pip", "install", "-e", ".[feetech,smolvla]"], check=True, cwd=parent_dir)

        print("Virtual environment created successfully!")
        print("Installed dependencies: base + feetech + smolvla")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up virtual environment: {e}")
        return False

def main():
    """Main setup function"""
    print("Starting repository setup...")

    # Install UV if needed
    if not install_uv():
        print("Failed to install UV. Please install manually with: pip install uv")
        sys.exit(1)

    # Setup virtual environment
    if not setup_venv():
        sys.exit(1)

    print("\nSetup complete!")
    print("\nTo finish setup, please:")
    print("1. Close and reopen your terminal")
    print("2. Navigate to this directory")
    print("3. Activate the virtual environment manually:")
    if os.name == 'nt':  # Windows
        print("   .venv\\Scripts\\activate")
    else:  # Unix-like systems
        print("   source .venv/bin/activate")

if __name__ == "__main__":
    main()
