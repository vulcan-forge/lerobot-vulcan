#!/usr/bin/env python3

import cv2
import glob
import subprocess
import time

def get_device_info(device_path):
    """Get detailed information about a video device"""
    try:
        # Try to get device info using v4l2-ctl
        result = subprocess.run(['v4l2-ctl', '--device', device_path, '--all'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"v4l2-ctl failed: {result.stderr}"
    except FileNotFoundError:
        return "v4l2-ctl not available"
    except Exception as e:
        return f"Error: {e}"

def test_camera_device_detailed(device_path):
    """Test if a camera device can be opened and read from with detailed info"""
    print(f"\n{'='*60}")
    print(f"Testing {device_path}")
    print(f"{'='*60}")
    
    # Get device info
    device_info = get_device_info(device_path)
    print(f"Device info:\n{device_info}")
    
    try:
        cap = cv2.VideoCapture(device_path)
        if not cap.isOpened():
            print(f"✗ Could not open {device_path}")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"✗ Could not read frame from {device_path}")
            cap.release()
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ {device_path} works!")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Try to get more properties
        for prop in [cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_CONTRAST, cv2.CAP_PROP_SATURATION]:
            value = cap.get(prop)
            print(f"  Property {prop}: {value}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Error with {device_path}: {e}")
        return False

def check_usb_connections():
    """Check USB connections for cameras"""
    print("\n" + "="*60)
    print("USB Camera Connections")
    print("="*60)
    
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            camera_lines = [line for line in lines if 'Microdia' in line or 'Webcam' in line]
            for line in camera_lines:
                print(f"  {line}")
        else:
            print("Failed to get USB info")
    except Exception as e:
        print(f"Error getting USB info: {e}")

def main():
    print("=== Detailed Camera Detection Test ===\n")
    
    # Check USB connections first
    check_usb_connections()
    
    # Get all video devices
    video_devices = sorted(glob.glob('/dev/video*'), key=lambda x: int(x.split('video')[1]))
    
    print(f"\nFound {len(video_devices)} video devices:")
    for device in video_devices:
        print(f"  {device}")
    
    working_cameras = []
    failed_cameras = []
    
    for device in video_devices:
        if test_camera_device_detailed(device):
            working_cameras.append(device)
        else:
            failed_cameras.append(device)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total video devices: {len(video_devices)}")
    print(f"Working cameras: {len(working_cameras)}")
    print(f"Failed cameras: {len(failed_cameras)}")
    
    if working_cameras:
        print("\nWorking camera devices:")
        for camera in working_cameras:
            print(f"  ✓ {camera}")
    
    if failed_cameras:
        print("\nFailed camera devices:")
        for camera in failed_cameras:
            print(f"  ✗ {camera}")
    
    # Check if we have 4 working cameras
    if len(working_cameras) >= 4:
        print(f"\n✓ You have {len(working_cameras)} working cameras - enough for sourcsey!")
        print("\nRecommended camera configuration:")
        for i, camera in enumerate(working_cameras[:4]):
            camera_names = ["front_left", "front_right", "wrist_left", "wrist_right"]
            print(f"  {camera_names[i]}: {camera}")
    else:
        print(f"\n⚠ You only have {len(working_cameras)} working cameras out of 4 needed")
        print("This might be due to:")
        print("  1. One camera not being properly connected")
        print("  2. Driver issues with one of the cameras")
        print("  3. USB bandwidth limitations")
        print("  4. Permission issues")

if __name__ == "__main__":
    main() 