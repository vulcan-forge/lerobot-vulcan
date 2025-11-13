"""
Diagnostic test script for camera connection issues.

This script:
1. Tests each camera individually to identify which ones work
2. Checks if device paths exist
3. Tests connection with/without warmup
4. Provides detailed diagnostics about camera state
5. Tests actual frame reading capability
"""

import logging
import sys
import traceback
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.robots import make_robot_from_config
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)


def check_device_exists(device_path: str) -> bool:
    """Check if a device file exists."""
    return os.path.exists(device_path)


def test_single_camera(cam_name: str, cam_config: OpenCVCameraConfig) -> dict:
    """Test a single camera and return diagnostic information."""
    results = {
        "name": cam_name,
        "path": cam_config.index_or_path,
        "device_exists": False,
        "can_open": False,
        "can_configure": False,
        "can_read_without_warmup": False,
        "can_read_with_warmup": False,
        "errors": [],
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing camera: {cam_name}")
    logger.info(f"Device path: {cam_config.index_or_path}")
    logger.info(f"{'='*60}")

    # Check if device exists
    if isinstance(cam_config.index_or_path, str):
        results["device_exists"] = check_device_exists(cam_config.index_or_path)
        logger.info(f"Device exists: {results['device_exists']}")
        if not results["device_exists"]:
            results["errors"].append(f"Device path {cam_config.index_or_path} does not exist")
            return results

    # Create camera instance
    try:
        camera = OpenCVCamera(cam_config)
        logger.info(f"Created camera instance: {camera}")
    except Exception as e:
        results["errors"].append(f"Failed to create camera instance: {e}")
        logger.error(f"Failed to create camera: {e}")
        return results

    # Test 1: Try to open without warmup
    try:
        logger.info("Test 1: Attempting to connect WITHOUT warmup...")
        camera.connect(warmup=False)
        results["can_open"] = True
        results["can_configure"] = True
        logger.info("✓ Camera opened and configured successfully")

        # Test reading without warmup
        try:
            logger.info("Test 2: Attempting to read frame (no warmup)...")
            frame = camera.read()
            results["can_read_without_warmup"] = True
            logger.info(f"✓ Successfully read frame: shape={frame.shape}, dtype={frame.dtype}")
        except Exception as e:
            results["errors"].append(f"Read failed without warmup: {e}")
            logger.warning(f"✗ Read failed without warmup: {e}")

        # Disconnect before next test
        try:
            camera.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    except Exception as e:
        results["errors"].append(f"Connection failed: {e}")
        logger.error(f"✗ Connection failed: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Try to clean up
        try:
            if camera.is_connected:
                camera.disconnect()
        except:
            pass
        return results

    # Test 3: Try to open WITH warmup
    try:
        logger.info("Test 3: Attempting to connect WITH warmup...")
        camera.connect(warmup=True)
        results["can_read_with_warmup"] = True
        logger.info("✓ Camera connected with warmup successfully")

        # Test reading after warmup
        try:
            logger.info("Test 4: Attempting to read frame (after warmup)...")
            frame = camera.read()
            logger.info(f"✓ Successfully read frame after warmup: shape={frame.shape}, dtype={frame.dtype}")
        except Exception as e:
            results["errors"].append(f"Read failed after warmup: {e}")
            logger.warning(f"✗ Read failed after warmup: {e}")

        # Disconnect
        try:
            camera.disconnect()
            logger.info("✓ Camera disconnected successfully")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    except Exception as e:
        results["errors"].append(f"Connection with warmup failed: {e}")
        logger.error(f"✗ Connection with warmup failed: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Try to clean up
        try:
            if camera.is_connected:
                camera.disconnect()
        except:
            pass

    return results


def test_all_cameras():
    """Test all cameras individually."""
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE CAMERA DIAGNOSTICS")
    logger.info("="*60)

    camera_configs = {
        "front_left": OpenCVCameraConfig(
            index_or_path="/dev/cameraFrontLeft",
            width=320,
            height=240,
            fps=30,
        ),
        "front_right": OpenCVCameraConfig(
            index_or_path="/dev/cameraFrontRight",
            width=320,
            height=240,
            fps=30,
        ),
        "wrist_left": OpenCVCameraConfig(
            index_or_path="/dev/cameraWristLeft",
            width=320,
            height=240,
            fps=30,
        ),
        "wrist_right": OpenCVCameraConfig(
            index_or_path="/dev/cameraWristRight",
            width=320,
            height=240,
            fps=30,
        ),
    }

    all_results = {}

    for cam_name, cam_config in camera_configs.items():
        all_results[cam_name] = test_single_camera(cam_name, cam_config)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    for cam_name, results in all_results.items():
        logger.info(f"\n{cam_name}:")
        logger.info(f"  Device exists: {results['device_exists']}")
        logger.info(f"  Can open: {results['can_open']}")
        logger.info(f"  Can read (no warmup): {results['can_read_without_warmup']}")
        logger.info(f"  Can read (with warmup): {results['can_read_with_warmup']}")
        if results['errors']:
            logger.info(f"  Errors: {len(results['errors'])}")
            for error in results['errors']:
                logger.info(f"    - {error}")

    return all_results


def test_robot_connection():
    """Test connecting the full robot with all cameras."""
    logger.info("\n" + "="*60)
    logger.info("TESTING FULL ROBOT CONNECTION")
    logger.info("="*60)

    config = SourcceyConfig(
        id="sourccey",
        cameras={
            "front_left": OpenCVCameraConfig(
                index_or_path="/dev/cameraFrontLeft",
                width=320,
                height=240,
                fps=30,
            ),
            "front_right": OpenCVCameraConfig(
                index_or_path="/dev/cameraFrontRight",
                width=320,
                height=240,
                fps=30,
            ),
            "wrist_left": OpenCVCameraConfig(
                index_or_path="/dev/cameraWristLeft",
                width=320,
                height=240,
                fps=30,
            ),
            "wrist_right": OpenCVCameraConfig(
                index_or_path="/dev/cameraWristRight",
                width=320,
                height=240,
                fps=30,
            ),
        },
    )

    device = make_robot_from_config(config)

    try:
        logger.info("Attempting to connect full robot...")
        device.connect(calibrate=False)
        logger.info("✓ Full robot connection successful!")

        # Try to read from cameras
        logger.info("Attempting to read observations...")
        obs = device.get_observation()
        logger.info(f"✓ Successfully read observations. Keys: {list(obs.keys())}")

    except Exception as e:
        logger.error(f"✗ Full robot connection failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
    finally:
        try:
            logger.info("Disconnecting robot...")
            device.disconnect()
            logger.info("✓ Robot disconnected successfully")
        except Exception as e:
            logger.error(f"✗ Disconnection failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")


def list_available_devices():
    """List all available video devices."""
    logger.info("\n" + "="*60)
    logger.info("AVAILABLE VIDEO DEVICES")
    logger.info("="*60)

    dev_dir = Path("/dev")
    video_devices = sorted(dev_dir.glob("video*"))

    if not video_devices:
        logger.warning("No /dev/video* devices found")
    else:
        logger.info(f"Found {len(video_devices)} video device(s):")
        for dev in video_devices:
            logger.info(f"  - {dev}")

    # Also check for camera symlinks
    camera_devices = sorted(dev_dir.glob("camera*"))
    if camera_devices:
        logger.info(f"\nFound {len(camera_devices)} camera symlink(s):")
        for dev in camera_devices:
            target = dev.resolve() if dev.is_symlink() else None
            logger.info(f"  - {dev} -> {target if target else 'not a symlink'}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Camera Connection Diagnostic Test Suite")
    print("="*60 + "\n")

    try:
        # Step 1: List available devices
        list_available_devices()

        # Step 2: Test each camera individually
        camera_results = test_all_cameras()

        # Step 3: Test full robot connection
        test_robot_connection()

        print("\n" + "="*60)
        print("All diagnostics completed")
        print("="*60)

        # Final summary
        working_cameras = [name for name, res in camera_results.items()
                          if res['can_read_without_warmup'] or res['can_read_with_warmup']]
        broken_cameras = [name for name, res in camera_results.items()
                         if not (res['can_read_without_warmup'] or res['can_read_with_warmup'])]

        if working_cameras:
            print(f"\n✓ Working cameras: {', '.join(working_cameras)}")
        if broken_cameras:
            print(f"\n✗ Broken cameras: {', '.join(broken_cameras)}")

    except Exception as e:
        logger.exception("Test suite failed with exception:")
        print(f"\nFULL TRACEBACK:\n{traceback.format_exc()}")
        sys.exit(1)
