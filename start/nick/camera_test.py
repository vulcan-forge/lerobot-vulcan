#!/usr/bin/env python3
"""
Simple Camera Port Detection for Linux

Shows which USB ports your cameras are connected to.
"""

import glob
import os
import re
import subprocess
from typing import Dict, List


def run_command(cmd: str) -> str:
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except:
        return ""


def get_real_cameras() -> List[Dict]:
    """Get real cameras using v4l2-ctl"""
    cameras = []
    # Try with sudo first, then without
    v4l2_output = run_command("sudo v4l2-ctl --list-devices 2>/dev/null")

    if not v4l2_output:
        v4l2_output = run_command("v4l2-ctl --list-devices 2>/dev/null")

    if v4l2_output:
        current_camera = None
        for line in v4l2_output.split('\n'):
            line = line.strip()
            if line and not line.startswith('/dev/'):
                # This is a camera name line
                if 'usb' in line.lower():
                    current_camera = {
                        'name': line,
                        'devices': []
                    }
                    cameras.append(current_camera)
            elif line.startswith('/dev/video') and current_camera:
                # This is a video device for the current camera
                current_camera['devices'].append(line)

    return cameras


def get_usb_port_info() -> Dict[str, str]:
    """Get USB port information for all devices"""
    port_info = {}
    usb_tree = run_command("lsusb -t 2>/dev/null")

    if usb_tree:
        for line in usb_tree.split('\n'):
            if line.strip() and 'Dev' in line and 'Class=Video' in line:
                # Parse lines like: |__ Port 001: Dev 011, If 0, Class=Video, Driver=uvcvideo, 480M
                port_match = re.search(r'Port (\d+)(?:\.(\d+))?', line)
                dev_match = re.search(r'Dev (\d+)', line)

                if port_match and dev_match:
                    port_parts = port_match.groups()
                    if port_parts[1]:  # Has sub-port
                        port = f"{port_parts[0]}.{port_parts[1]}"
                    else:
                        port = port_parts[0]

                    dev_num = dev_match.group(1)

                    # Find the bus number from the line above
                    bus_match = re.search(r'Bus (\d+)', line)
                    if bus_match:
                        bus_num = bus_match.group(1)
                        device_key = f"{bus_num}:{dev_num}"
                        port_info[device_key] = f"Port {port}"
                    else:
                        # Look for bus in previous lines
                        lines_before = usb_tree.split('\n')[:usb_tree.split('\n').index(line)]
                        for prev_line in reversed(lines_before):
                            bus_match = re.search(r'Bus (\d+)', prev_line)
                            if bus_match:
                                bus_num = bus_match.group(1)
                                device_key = f"{bus_num}:{dev_num}"
                                port_info[device_key] = f"Port {port}"
                                break

    return port_info


def get_video_device_usb_info(device_path: str) -> Dict:
    """Get USB info for a video device using udevadm"""
    usb_info = {}

    # Use udevadm to get device properties
    udev_output = run_command(f"udevadm info --query=property --name={device_path}")

    if udev_output:
        for line in udev_output.split('\n'):
            if line.startswith('ID_VENDOR_ID='):
                usb_info['vendor_id'] = line.split('=')[1]
            elif line.startswith('ID_MODEL_ID='):
                usb_info['model_id'] = line.split('=')[1]
            elif line.startswith('ID_USB_INTERFACE_NUM='):
                usb_info['interface'] = line.split('=')[1]

    # Try to get bus/device numbers from sysfs
    try:
        # Get the device path in sysfs
        device_name = device_path.split('/')[-1]
        sysfs_path = run_command(f"readlink -f /sys/class/video4linux/{device_name}/device")

        if sysfs_path:
            # Navigate up to find the USB device directory
            current_path = sysfs_path
            for _ in range(10):  # Limit depth to avoid infinite loops
                if os.path.exists(f"{current_path}/busnum") and os.path.exists(f"{current_path}/devnum"):
                    busnum = run_command(f"cat {current_path}/busnum 2>/dev/null")
                    devnum = run_command(f"cat {current_path}/devnum 2>/dev/null")
                    if busnum and devnum:
                        usb_info['bus_num'] = busnum.strip()
                        usb_info['dev_num'] = devnum.strip()
                        break
                current_path = os.path.dirname(current_path)
                if current_path == '/':
                    break
    except:
        pass

    return usb_info


def find_camera_ports():
    """Find all cameras and their USB ports"""
    print("Camera USB Port Detection")
    print("=" * 30)

    # Get real cameras using v4l2-ctl
    real_cameras = get_real_cameras()

    if not real_cameras:
        print("No real cameras found.")
        return

    print(f"Found {len(real_cameras)} camera(s):\n")

    for camera in real_cameras:
        print(f"Camera: {camera['name']}")
        print(f"  Video devices: {', '.join(camera['devices'])}")

        # Get USB info for the first device of this camera
        if camera['devices']:
            usb_info = get_video_device_usb_info(camera['devices'][0])
            if usb_info and 'bus_num' in usb_info and 'dev_num' in usb_info:
                bus = usb_info['bus_num']
                dev = usb_info['dev_num']
                vendor_info = ""
                if 'vendor_id' in usb_info and 'model_id' in usb_info:
                    vendor_info = f" ({usb_info['vendor_id']}:{usb_info['model_id']})"
                print(f"  USB Bus: {bus}, Device: {dev}{vendor_info}")

        print()


if __name__ == "__main__":
    find_camera_ports()
