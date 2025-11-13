"""
Test script to reproduce and verify fixes for camera connection/disconnection issues.

This script tests:
1. Camera connection timeout handling
2. Graceful disconnection when cameras fail to connect
3. Partial connection state handling
"""

import logging
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.robots import make_robot_from_config
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)


def test_camera_connection_failure():
    """Test that camera connection failures are handled gracefully."""
    logger.info("=" * 60)
    logger.info("Test 1: Camera connection failure handling")
    logger.info("=" * 60)

    # Create config similar to auto_calibrate - need to use OpenCVCameraConfig objects
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
        logger.info("Attempting to connect device (this may fail on cameras)...")
        device.connect(calibrate=False)
        logger.info("✓ Connection successful")
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.info("Testing graceful disconnection after connection failure...")
    finally:
        try:
            logger.info("Attempting to disconnect...")
            device.disconnect()
            logger.info("✓ Disconnection successful (no errors raised)")
        except Exception as e:
            logger.error(f"✗ Disconnection failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error("This indicates the disconnect() method needs to be more robust")
            raise


def test_partial_connection():
    """Test handling of partial connection states."""
    logger.info("=" * 60)
    logger.info("Test 2: Partial connection state handling")
    logger.info("=" * 60)

    config = SourcceyConfig(id="sourccey")
    device = make_robot_from_config(config)

    # Try to disconnect without connecting first
    try:
        logger.info("Attempting to disconnect without connecting first...")
        device.disconnect()
        logger.info("✓ Disconnection handled gracefully (no error raised)")
    except Exception as e:
        logger.warning(f"Disconnect raised error (may be expected): {e}")
        logger.warning(f"Exception type: {type(e).__name__}")
        logger.warning(f"Traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Camera Connection/Disconnection Test Suite")
    print("=" * 60 + "\n")

    try:
        test_camera_connection_failure()
        print()
        test_partial_connection()
        print("\n" + "=" * 60)
        print("All tests completed")
        print("=" * 60)
    except Exception as e:
        logger.exception("Test suite failed with exception:")
        print(f"\nFULL TRACEBACK:\n{traceback.format_exc()}")
        sys.exit(1)
