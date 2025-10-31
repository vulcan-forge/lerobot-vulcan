"""
Larger protobuf test with 20 fields - more realistic robot data.
Compares JSON vs Protobuf for a larger message.
"""

import json
import time
import zmq
import threading

# We'll generate this from robot_data.proto
try:
    import robot_data_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    print("Note: Run 'python -m grpc_tools.protoc -I. --python_out=. robot_data.proto' first")


def create_large_robot_data():
    """Create a larger robot observation dict (like real robot data)."""
    return {
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
        "status_code": 200
    }


def json_server():
    """JSON server for larger data."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5557")

    print("JSON Server (large data) running on port 5557...")

    while True:
        try:
            message = socket.recv_string()
            data = json.loads(message)

            # Echo back with some processing
            response = {
                "echo_left_shoulder": data["left_shoulder_pan"],
                "echo_right_shoulder": data["right_shoulder_pan"],
                "echo_battery": data["battery_level"],
                "echo_status": data["status_code"]
            }
            socket.send_string(json.dumps(response))

        except KeyboardInterrupt:
            break

    socket.close()
    context.term()


def json_client():
    """JSON client for larger data."""
    time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5557")

    data = create_large_robot_data()

    # Benchmark JSON with larger data
    start_time = time.perf_counter()

    for i in range(1000):
        message = json.dumps(data)
        socket.send_string(message)

        response = socket.recv_string()
        result = json.loads(response)

    end_time = time.perf_counter()

    print(f"JSON (20 fields): {1000} round trips in {(end_time - start_time)*1000:.2f}ms")
    print(f"JSON (20 fields): {(end_time - start_time)/1000*1000:.2f}μs per message")

    socket.close()
    context.term()


def protobuf_server():
    """Protobuf server for larger data."""
    if not PROTOBUF_AVAILABLE:
        print("Protobuf not available - skipping protobuf server")
        return

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5558")

    print("Protobuf Server (large data) running on port 5558...")

    while True:
        try:
            message = socket.recv()
            msg = robot_data_pb2.RobotObservation()
            msg.ParseFromString(message)

            # Echo back
            response = robot_data_pb2.RobotObservation()
            response.left_shoulder_pan = msg.left_shoulder_pan
            response.right_shoulder_pan = msg.right_shoulder_pan
            response.battery_level = msg.battery_level
            response.status_code = msg.status_code

            socket.send(response.SerializeToString())

        except KeyboardInterrupt:
            break

    socket.close()
    context.term()


def protobuf_client():
    """Protobuf client for larger data."""
    if not PROTOBUF_AVAILABLE:
        print("Protobuf not available - skipping protobuf client")
        return

    time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5558")

    # Benchmark Protobuf with larger data
    start_time = time.perf_counter()

    for i in range(1000):
        msg = robot_data_pb2.RobotObservation()
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

        socket.send(msg.SerializeToString())

        response = socket.recv()
        result = robot_data_pb2.RobotObservation()
        result.ParseFromString(response)

    end_time = time.perf_counter()

    print(f"Protobuf (20 fields): {1000} round trips in {(end_time - start_time)*1000:.2f}ms")
    print(f"Protobuf (20 fields): {(end_time - start_time)/1000*1000:.2f}μs per message")

    socket.close()
    context.term()


def compare_sizes():
    """Compare message sizes for larger data."""
    data = create_large_robot_data()

    # JSON size
    json_msg = json.dumps(data)
    json_size = len(json_msg.encode('utf-8'))

    print(f"\nLarge Message Size Comparison (20 fields):")
    print(f"JSON: {json_size} bytes ({json_size / 1024:.2f} KB)")

    if PROTOBUF_AVAILABLE:
        # Protobuf size
        msg = robot_data_pb2.RobotObservation()
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

        protobuf_size = len(msg.SerializeToString())

        print(f"Protobuf: {protobuf_size} bytes ({protobuf_size / 1024:.2f} KB)")
        print(f"Protobuf is {json_size/protobuf_size:.1f}x smaller!")
        print(f"Size reduction: {((json_size - protobuf_size) / json_size * 100):.1f}%")
    else:
        print("Protobuf: Not available (run protoc first)")


if __name__ == "__main__":
    print("Large Protobuf vs JSON Test (20 fields)")
    print("=" * 50)

    # Compare message sizes
    compare_sizes()

    print("\nTo run the full test:")
    print("1. Generate protobuf code:")
    print("   python -m grpc_tools.protoc -I. --python_out=. robot_data.proto")
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
