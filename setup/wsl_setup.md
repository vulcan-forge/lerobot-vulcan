# Run this before running setup.py

sudo apt update
sudo apt install -y build-essential python3-dev linux-libc-dev pkg-config

# Install FFMPEG

sudo apt-get update
sudo apt-get install -y ffmpeg

# Setup

python3 setup/setup.py
