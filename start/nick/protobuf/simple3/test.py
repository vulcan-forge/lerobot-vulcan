"""
Full-scale robot data test with all joints, sensors, and 4 camera images.
This tests the real-world scenario with large data payloads.
"""

import json
import time
import zmq
import threading
import numpy as np
import cv2
import base64

# We'll generate this from full_robot.proto
try:
    import full_robot_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    print("Note: Run 'python -m grpc_tools.protoc -I. --python_out=. full_robot.proto' first")


def create_camera_image(width=640, height=480, quality=90):
    """Create a realistic camera image."""
    # Generate a realistic-looking image (simulate camera feed)
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Add some structure to make it more realistic
    cv2.rectangle(image, (50, 50), (200, 150), (255, 255, 255), 2)
    cv2.circle(image, (320, 240), 50, (0, 255, 0), 2)

    # Encode as JPEG
    ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()


def create_full_robot_data():
    """Create a complete robot observation dict with all data."""
    # Generate camera images
    camera_front = create_camera_image()
    camera_left = create_camera_image()
    camera_right = create_camera_image()
    camera_rear = create_camera_image()

    observation = {
        # Left arm joints
        "left_shoulder_pan": 0.123456789,
        "left_shoulder_lift": -0.456789123,
        "left_elbow_flex": 0.789123456,
        "left_wrist_flex": -0.234567890,
        "left_wrist_roll": 0.567890123,
        "left_gripper": 0.890123456,

        # Right arm joints
        "right_shoulder_pan": -0.123456789,
        "right_shoulder_lift": 0.456789123,
        "right_elbow_flex": -0.789123456,
        "right_wrist_flex": 0.234567890,
        "right_wrist_roll": -0.567890123,
        "right_gripper": -0.890123456,

        # Base velocities
        "x_velocity": 0.500000000,
        "y_velocity": -0.300000000,
        "z_velocity": 0.000000000,
        "theta_velocity": 0.200000000,

        # Additional sensor data
        "battery_level": 85.5,
        "temperature": 23.7,
        "emergency_stop": False,
        "status_code": 200,
        "cpu_usage": 45.2,
        "memory_usage": 67.8,

        # Camera images (base64 encoded for JSON)
        "camera_front": base64.b64encode(camera_front).decode('utf-8'),
        "camera_left": base64.b64encode(camera_left).decode('utf-8'),
        "camera_right": base64.b64encode(camera_right).decode('utf-8'),
        "camera_rear": base64.b64encode(camera_rear).decode('utf-8'),

        "timestamp": time.time()
    }

    return observation, {
        'camera_front': camera_front,
        'camera_left': camera_left,
        'camera_right': camera_right,
        'camera_rear': camera_rear
    }


def json_server():
    """JSON server for full robot data."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5559")

    print("JSON Server (full robot data) running on port 5559...")

    while True:
        try:
            message = socket.recv_string()
            data = json.loads(message)

            # Echo back with some processing
            response = {
                "echo_left_shoulder": data["left_shoulder_pan"],
                "echo_battery": data["battery_level"],
                "echo_camera_count": len([k for k in data.keys() if k.startswith("camera_")])
            }
            socket.send_string(json.dumps(response))

        except KeyboardInterrupt:
            break

    socket.close()
    context.term()


def json_client():
    """JSON client for full robot data."""
    time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5559")

    data, _ = create_full_robot_data()

    # Benchmark JSON with full robot data
    start_time = time.perf_counter()

    for i in range(100):  # Reduced iterations due to large data
        message = json.dumps(data)
        socket.send_string(message)

        response = socket.recv_string()
        result = json.loads(response)

    end_time = time.perf_counter()

    print(f"JSON (full robot data): {100} round trips in {(end_time - start_time)*1000:.2f}ms")
    print(f"JSON (full robot data): {(end_time - start_time)/100*1000:.2f}μs per message")

    socket.close()
    context.term()


def protobuf_server():
    """Protobuf server for full robot data."""
    if not PROTOBUF_AVAILABLE:
        print("Protobuf not available - skipping protobuf server")
        return

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5560")

    print("Protobuf Server (full robot data) running on port 5560...")

    while True:
        try:
            message = socket.recv()
            msg = full_robot_pb2.RobotObservation()
            msg.ParseFromString(message)

            # Echo back
            response = full_robot_pb2.RobotObservation()
            response.left_shoulder_pan = msg.left_shoulder_pan
            response.battery_level = msg.battery_level
            response.timestamp = msg.timestamp

            socket.send(response.SerializeToString())

        except KeyboardInterrupt:
            break

    socket.close()
    context.term()


def protobuf_client():
    """Protobuf client for full robot data."""
    if not PROTOBUF_AVAILABLE:
        print("Protobuf not available - skipping protobuf client")
        return

    time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5560")

    _, camera_data = create_full_robot_data()

    # Benchmark Protobuf with full robot data
    start_time = time.perf_counter()

    for i in range(100):  # Reduced iterations due to large data
        msg = full_robot_pb2.RobotObservation()

        # Set all fields
        msg.left_shoulder_pan = 0.123456789
        msg.left_shoulder_lift = -0.456789123
        msg.left_elbow_flex = 0.789123456
        msg.left_wrist_flex = -0.234567890
        msg.left_wrist_roll = 0.567890123
        msg.left_gripper = 0.890123456
        msg.right_shoulder_pan = -0.123456789
        msg.right_shoulder_lift = 0.456789123
        msg.right_elbow_flex = -0.789123456
        msg.right_wrist_flex = 0.234567890
        msg.right_wrist_roll = -0.567890123
        msg.right_gripper = -0.890123456
        msg.x_velocity = 0.500000000
        msg.y_velocity = -0.300000000
        msg.z_velocity = 0.000000000
        msg.theta_velocity = 0.200000000
        msg.battery_level = 85.5
        msg.temperature = 23.7
        msg.emergency_stop = False
        msg.status_code = 200
        msg.cpu_usage = 45.2
        msg.memory_usage = 67.8
        msg.timestamp = time.time()

        # Add camera images
        for cam_name, cam_data in camera_data.items():
            camera = msg.cameras.add()
            camera.name = cam_name
            camera.jpeg_data = cam_data
            camera.width = 640
            camera.height = 480
            camera.quality = 90

        socket.send(msg.SerializeToString())

        response = socket.recv()
        result = full_robot_pb2.RobotObservation()
        result.ParseFromString(response)

    end_time = time.perf_counter()

    print(f"Protobuf (full robot data): {100} round trips in {(end_time - start_time)*1000:.2f}ms")
    print(f"Protobuf (full robot data): {(end_time - start_time)/100*1000:.2f}μs per message")

    socket.close()
    context.term()


def compare_sizes():
    """Compare message sizes for full robot data."""
    data, camera_data = create_full_robot_data()

    # JSON size
    json_msg = json.dumps(data)
    json_size = len(json_msg.encode('utf-8'))

    print(f"\nFull Robot Data Size Comparison:")
    print(f"JSON: {json_size:,} bytes ({json_size / 1024:.2f} KB)")

    if PROTOBUF_AVAILABLE:
        # Protobuf size
        msg = full_robot_pb2.RobotObservation()
        msg.left_shoulder_pan = 0.123456789
        msg.left_shoulder_lift = -0.456789123
        msg.left_elbow_flex = 0.789123456
        msg.left_wrist_flex = -0.234567890
        msg.left_wrist_roll = 0.567890123
        msg.left_gripper = 0.890123456
        msg.right_shoulder_pan = -0.123456789
        msg.right_shoulder_lift = 0.456789123
        msg.right_elbow_flex = -0.789123456
        msg.right_wrist_flex = 0.234567890
        msg.right_wrist_roll = -0.567890123
        msg.right_gripper = -0.890123456
        msg.x_velocity = 0.500000000
        msg.y_velocity = -0.300000000
        msg.z_velocity = 0.000000000
        msg.theta_velocity = 0.200000000
        msg.battery_level = 85.5
        msg.temperature = 23.7
        msg.emergency_stop = False
        msg.status_code = 200
        msg.cpu_usage = 45.2
        msg.memory_usage = 67.8
        msg.timestamp = time.time()

        # Add camera images
        for cam_name, cam_data in camera_data.items():
            camera = msg.cameras.add()
            camera.name = cam_name
            camera.jpeg_data = cam_data
            camera.width = 640
            camera.height = 480
            camera.quality = 90

        protobuf_size = len(msg.SerializeToString())

        print(f"Protobuf: {protobuf_size:,} bytes ({protobuf_size / 1024:.2f} KB)")
        print(f"Protobuf is {json_size/protobuf_size:.1f}x smaller!")
        print(f"Size reduction: {((json_size - protobuf_size) / json_size * 100):.1f}%")

        # Calculate bandwidth savings for 30Hz
        json_per_second = json_size * 30
        protobuf_per_second = protobuf_size * 30
        savings_per_second = json_per_second - protobuf_per_second

        print(f"\nBandwidth at 30Hz:")
        print(f"JSON: {json_per_second / 1024:.1f} KB/sec")
        print(f"Protobuf: {protobuf_per_second / 1024:.1f} KB/sec")
        print(f"Savings: {savings_per_second / 1024:.1f} KB/sec")
    else:
        print("Protobuf: Not available (run protoc first)")


if __name__ == "__main__":
    print("Full Robot Data Protobuf vs JSON Test")
    print("=" * 50)
    print("Includes: 12 joints + 4 sensors + 4 cameras (640x480)")

    # Compare message sizes
    compare_sizes()

    print("\nTo run the full test:")
    print("1. Generate protobuf code:")
    print("   python -m grpc_tools.protoc -I. --python_out=. full_robot.proto")
    print("2. Run servers in separate terminals:")
    print("   python test.py --json-server")
    print("   python test.py --protobuf-server")
    print("3. Run clients:")
    print("   python test.py --json-client")
    print("   python test.py --protobuf-client")

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--json-server":
            json_server()
        elif sys.argv[1] == "--json-client":
            json_client()
        elif sys.argv[1] == "--protobuf-server":
            protobuf_server()
        elif sys.argv[1] == "--protobuf-client":
            protobuf_client()
