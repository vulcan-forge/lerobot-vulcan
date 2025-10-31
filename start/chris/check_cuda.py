#!/usr/bin/env python3
"""Minimal CUDA test script - checks if CUDA is installed and GPU is being used."""

import subprocess
import torch

# Check if nvidia-smi is available (CUDA installed)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    print(f"CUDA installed: {'✓' if result.returncode == 0 else '✗'}")
except:
    print("CUDA installed: ✗")

# Check if PyTorch can use CUDA
try:
    print(f"GPU available: {'✓' if torch.cuda.is_available() else '✗'}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
except:
    print("GPU available: ✗") 