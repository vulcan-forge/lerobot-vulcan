#!/usr/bin/env python3

import cv2
import glob
import subprocess
import os

def run_shell_command(cmd):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running command: {e}"

def main():
    print("=== Camera Detection Test ===\n")
    
    # Check what video devices exist
    print("1. Available video devices:")
    video_devices = glob.glob('/dev/video*')
    if video_devices:
        for device in video_devices:
            print(f"  {device}")
    else:
        print("  No video devices found")
    
    print("\n2. USB devices (looking for cameras):")
    usb_output = run_shell_command("lsusb")
    camera_lines = [line for line in usb_output.split('\n') if 'camera' in line.lower() or 'webcam' in line.lower()]
    if camera_lines:
        for line in camera_lines:
            print(f"  {line}")
    else:
        print("  No camera-like USB devices found")
    
    print("\n3. Testing video devices:")
    working_cameras = []
    
    # Test cameras 0-20
    for i in range(21):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera {i}: ✓ Working - Frame shape: {frame.shape}")
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"    Properties: {width}x{height} @ {fps:.1f}fps")
                working_cameras.append(i)
            else:
                print(f"  Camera {i}: ✗ No frame")
            cap.release()
        else:
            print(f"  Camera {i}: ✗ Cannot open")
    
    print(f"\n4. Summary:")
    print(f"  Total video devices found: {len(video_devices)}")
    print(f"  Working cameras: {len(working_cameras)}")
    if working_cameras:
        print(f"  Working camera indices: {working_cameras}")
    
    # Test specific video devices mentioned in sourccey config
    print(f"\n5. Testing sourccey-specific video devices:")
    sourccey_devices = ['/dev/video0', '/dev/video4', '/dev/video8', '/dev/video12']
    for device in sourccey_devices:
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  {device}: ✓ Working - Frame shape: {frame.shape}")
            else:
                print(f"  {device}: ✗ No frame")
            cap.release()
        else:
            print(f"  {device}: ✗ Cannot open")

if __name__ == "__main__":
    main()