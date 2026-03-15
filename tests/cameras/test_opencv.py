#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_opencv.py::test_connect
# ```

import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

RealVideoCapture = cv2.VideoCapture


class MockLoopingVideoCapture:
    """
    Wraps the real OpenCV VideoCapture.
    Motivation: cv2.VideoCapture(file.png) is only valid for one read.
    Strategy: Read the file once & return the cached frame for subsequent reads.
    Consequence: No recurrent I/O operations, but we keep the test artifacts simple.
    """

    def __init__(self, *args, **kwargs):
        args_clean = [str(a) if isinstance(a, Path) else a for a in args]
        self._real_vc = RealVideoCapture(*args_clean, **kwargs)
        self._cached_frame = None

    def read(self):
        ret, frame = self._real_vc.read()

        if ret:
            self._cached_frame = frame
            return ret, frame

        if not ret and self._cached_frame is not None:
            return True, self._cached_frame.copy()

        return ret, frame

    def __getattr__(self, name):
        return getattr(self._real_vc, name)


class MockReconnectVideoCapture:
    instance_count = 0
    frame = np.full((120, 160, 3), 127, dtype=np.uint8)

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instance_count += 1
        self.instance_id = type(self).instance_count
        self._is_open = True
        self._width = self.frame.shape[1]
        self._height = self.frame.shape[0]
        self._fps = 30.0

    def isOpened(self):
        return self._is_open

    def read(self):
        if not self._is_open:
            return False, None

        if self.instance_id == 1:
            return False, None

        return True, self.frame.copy()

    def release(self):
        self._is_open = False

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
        elif prop == cv2.CAP_PROP_FPS:
            self._fps = float(value)
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FOURCC:
            return 0.0
        return 0.0

    def getBackendName(self):
        return "MOCK"


@pytest.fixture(autouse=True)
def patch_opencv_videocapture():
    """
    Automatically patches cv2.VideoCapture for all tests.
    """
    module_path = OpenCVCamera.__module__
    target = f"{module_path}.cv2.VideoCapture"

    with patch(target, new=MockLoopingVideoCapture):
        yield


# NOTE(Steven): more tests + assertions?
TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
DEFAULT_PNG_FILE_PATH = TEST_ARTIFACTS_DIR / "image_160x120.png"
TEST_IMAGE_SIZES = ["128x128", "160x120", "320x180", "480x270"]
TEST_IMAGE_PATHS = [TEST_ARTIFACTS_DIR / f"image_{size}.png" for size in TEST_IMAGE_SIZES]

def _check_opencv_backends_available():
    """Check if OpenCV has working backends for image files."""
    try:
        if not DEFAULT_PNG_FILE_PATH.exists():
            return False

        # Check if FFmpeg backend works
        cap = cv2.VideoCapture(str(DEFAULT_PNG_FILE_PATH), cv2.CAP_FFMPEG)
        ffmpeg_works = cap.isOpened()
        cap.release()

        if ffmpeg_works:
            return True

        # Try DirectShow backend (Windows)
        cap = cv2.VideoCapture(str(DEFAULT_PNG_FILE_PATH), cv2.CAP_DSHOW)
        dshow_works = cap.isOpened()
        cap.release()

        return dshow_works
    except Exception:
        return False


def _check_opencv_image_support():
    """Check if OpenCV can handle image files on this platform."""
    if sys.platform == "win32":
        # On Windows, VideoCapture with DirectShow backend doesn't support image files
        # This is a known limitation - DirectShow can't open static image files as video sources
        return False
    else:
        # On Linux/macOS, assume it works (usually does)
        return True


# Reusable skip conditions
SKIP_NO_OPENCV_IMAGE_SUPPORT = pytest.mark.skipif(
    not _check_opencv_image_support(),
    reason="OpenCV cannot open image files as video sources. "
           "This is common on Windows without proper codecs. "
           "To fix: Install FFmpeg or use a different OpenCV build with image support. "
           "Run: conda install ffmpeg or pip install opencv-python-headless"
)

SKIP_NO_OPENCV_BACKENDS = pytest.mark.skipif(
    not _check_opencv_backends_available(),
    reason="OpenCV backends (FFmpeg/DirectShow) not available for image files. "
           "Install FFmpeg or use OpenCV with proper codec support."
)


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = OpenCVCameraConfig(index_or_path=0)

    _ = OpenCVCamera(config)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        assert camera.is_connected


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_connect_already_connected():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect()


def test_connect_invalid_camera_path():
    config = OpenCVCameraConfig(index_or_path="nonexistent/camera.png")

    camera = OpenCVCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_invalid_width_connect():
    config = OpenCVCameraConfig(
        index_or_path=DEFAULT_PNG_FILE_PATH,
        width=99999,  # Invalid width to trigger error
        height=480,
    )

    camera = OpenCVCamera(config)
    with pytest.raises(RuntimeError):
        camera.connect(warmup=False)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path, warmup_s=0)

    with OpenCVCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)

    camera = OpenCVCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_disconnect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_disconnect_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.disconnect()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
def test_async_read(index_or_path):
    config = OpenCVCameraConfig(index_or_path=index_or_path, warmup_s=0)

    with OpenCVCamera(config) as camera:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_async_read_timeout():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera, pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)  # consumes any available frame by then
        camera.async_read(timeout_ms=0)  # request immediately another one


def test_async_read_reconnects_after_consecutive_failures():
    module_path = OpenCVCamera.__module__
    target = f"{module_path}.cv2.VideoCapture"
    MockReconnectVideoCapture.instance_count = 0

    config = OpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=160,
        height=120,
        warmup_s=0,
        max_consecutive_read_failures=2,
        reconnect_interval_s=0.01,
    )

    with patch(target, new=MockReconnectVideoCapture):
        with OpenCVCamera(config) as camera:
            img = camera.async_read(timeout_ms=500)

            assert isinstance(img, np.ndarray)
            assert img.shape == (120, 160, 3)
            assert MockReconnectVideoCapture.instance_count >= 2
            assert camera.thread is not None
            assert camera.thread.is_alive()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_async_read_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)
    camera = OpenCVCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_read_latest():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # ensure at least one fresh frame is captured
        frame = camera.read()
        latest = camera.read_latest()

        assert isinstance(latest, np.ndarray)
        assert latest.shape == frame.shape


def test_read_latest_before_connect():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH)

    camera = OpenCVCamera(config)
    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read_latest()


def test_read_latest_high_frequency():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # prime to ensure frames are available
        ref = camera.read()

        for _ in range(20):
            latest = camera.read_latest()
            assert isinstance(latest, np.ndarray)
            assert latest.shape == ref.shape


def test_read_latest_too_old():
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, warmup_s=0)

    with OpenCVCamera(config) as camera:
        # prime to ensure frames are available
        _ = camera.read()

        with pytest.raises(TimeoutError):
            _ = camera.read_latest(max_age_ms=0)  # immediately too old


def test_fourcc_configuration():
    """Test FourCC configuration validation and application."""

    # Test MJPG specifically (main use case)
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc="MJPG")
    camera = OpenCVCamera(config)
    assert camera.config.fourcc == "MJPG"

    # Test a few other common formats
    valid_fourcc_codes = ["YUYV", "YUY2", "RGB3"]

    for fourcc in valid_fourcc_codes:
        config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc=fourcc)
        camera = OpenCVCamera(config)
        assert camera.config.fourcc == fourcc

    # Test invalid FOURCC codes
    invalid_fourcc_codes = ["ABC", "ABCDE", ""]

    for fourcc in invalid_fourcc_codes:
        with pytest.raises(ValueError):
            OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc=fourcc)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
def test_fourcc_with_camera():
    """Test FourCC functionality with actual camera connection."""
    config = OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, fourcc="MJPG", warmup_s=0)

    # Connect should work with MJPG specified
    with OpenCVCamera(config) as camera:
        assert camera.is_connected

        # Read should work normally
        img = camera.read()
        assert isinstance(img, np.ndarray)


@SKIP_NO_OPENCV_IMAGE_SUPPORT
@pytest.mark.parametrize("index_or_path", TEST_IMAGE_PATHS, ids=TEST_IMAGE_SIZES)
@pytest.mark.parametrize(
    "rotation",
    [
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ],
    ids=["no_rot", "rot90", "rot180", "rot270"],
)
def test_rotation(rotation, index_or_path):
    filename = Path(index_or_path).name
    dimensions = filename.split("_")[-1].split(".")[0]  # Assumes filenames format (_wxh.png)
    original_width, original_height = map(int, dimensions.split("x"))

    config = OpenCVCameraConfig(index_or_path=index_or_path, rotation=rotation, warmup_s=0)
    with OpenCVCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)

        if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            assert camera.width == original_height
            assert camera.height == original_width
            assert img.shape[:2] == (original_width, original_height)
        else:
            assert camera.width == original_width
            assert camera.height == original_height
            assert img.shape[:2] == (original_height, original_width)
