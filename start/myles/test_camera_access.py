#!/usr/bin/env python3

import cv2
import glob
import time

def test_camera_device(device_path):
    """Test if a camera device can be opened and read from"""
    print(f"Testing {device_path}...")
    
    try:
        cap = cv2.VideoCapture(device_path)
        if not cap.isOpened():
            print(f"  ✗ Could not open {device_path}")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"  ✗ Could not read frame from {device_path}")
            cap.release()
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  ✓ {device_path} works! Resolution: {width}x{height}, FPS: {fps}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"  ✗ Error with {device_path}: {e}")
        return False

def main():
    print("=== Testing Camera Device Access ===\n")
    
    # Get all video devices
    video_devices = sorted(glob.glob('/dev/video*'), key=lambda x: int(x.split('video')[1]))
    
    working_cameras = []
    
    for device in video_devices:
        if test_camera_device(device):
            working_cameras.append(device)
        print()
    
    print("=== Summary ===")
    print(f"Total video devices found: {len(video_devices)}")
    print(f"Working cameras: {len(working_cameras)}")
    
    if working_cameras:
        print("\nWorking camera devices:")
        for camera in working_cameras:
            print(f"  {camera}")
        
        print("\nYou can use these devices in your LeRobot configuration.")
        print("For example, if /dev/video20 works, you can set:")
        print("  camera_port = '/dev/video20'")
    else:
        print("\nNo working cameras found. This might be a permissions issue.")
        print("Try running: sudo usermod -a -G video $USER")
        print("Then log out and log back in.")

if __name__ == "__main__":
    main() 