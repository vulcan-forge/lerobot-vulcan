"""
The simplest possible protobuf test.
Just sends a simple key-value pair and compares JSON vs Protobuf.
"""

import json
import time
import zmq
import threading

# We'll generate this from simple.proto
try:
    import simple_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    print("Note: Run 'python -m grpc_tools.protoc -I. --python_out=. simple.proto' first")


def create_simple_data():
    """Create a simple dictionary like what robots send."""
    return {
        "motor_position": 0.12345
    }


def json_server():
    """Simple JSON server."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("JSON Server running on port 5555...")

    while True:
        try:
            # Receive JSON message
            message = socket.recv_string()
            data = json.loads(message)

            # Echo back
            response = {"echo": data["motor_position"]}
            socket.send_string(json.dumps(response))

        except KeyboardInterrupt:
            break

    socket.close()
    context.term()


def json_client():
    """Simple JSON client."""
    time.sleep(1)  # Wait for server

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    data = create_simple_data()

    # Benchmark JSON
    start_time = time.perf_counter()

    for i in range(1000):
        # Send JSON
        message = json.dumps(data)
        socket.send_string(message)

        # Receive response
        response = socket.recv_string()
        result = json.loads(response)

    end_time = time.perf_counter()

    print(f"JSON: {1000} round trips in {(end_time - start_time)*1000:.2f}ms")
    print(f"JSON: {(end_time - start_time)/1000*1000:.2f}μs per message")

    socket.close()
    context.term()


def protobuf_server():
    """Simple Protobuf server."""
    if not PROTOBUF_AVAILABLE:
        print("Protobuf not available - skipping protobuf server")
        return

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5556")

    print("Protobuf Server running on port 5556...")

    while True:
        try:
            # Receive protobuf message
            message = socket.recv()
            msg = simple_pb2.SimpleMessage()
            msg.ParseFromString(message)

            # Echo back
            response = simple_pb2.SimpleMessage()
            response.key = "echo"
            response.value = msg.value
            socket.send(response.SerializeToString())

        except KeyboardInterrupt:
            break

    socket.close()
    context.term()


def protobuf_client():
    """Simple Protobuf client."""
    if not PROTOBUF_AVAILABLE:
        print("Protobuf not available - skipping protobuf client")
        return

    time.sleep(1)  # Wait for server

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    # Benchmark Protobuf
    start_time = time.perf_counter()

    for i in range(1000):
        # Send protobuf
        msg = simple_pb2.SimpleMessage()
        msg.key = "motor_position"
        msg.value = 0.12345
        socket.send(msg.SerializeToString())

        # Receive response
        response = socket.recv()
        result = simple_pb2.SimpleMessage()
        result.ParseFromString(response)

    end_time = time.perf_counter()

    print(f"Protobuf: {1000} round trips in {(end_time - start_time)*1000:.2f}ms")
    print(f"Protobuf: {(end_time - start_time)/1000*1000:.2f}μs per message")

    socket.close()
    context.term()


def compare_sizes():
    """Compare message sizes."""
    data = create_simple_data()

    # JSON size
    json_msg = json.dumps(data)
    json_size = len(json_msg.encode('utf-8'))

    print(f"\nMessage Size Comparison:")
    print(f"JSON: '{json_msg}' = {json_size} bytes")

    if PROTOBUF_AVAILABLE:
        # Protobuf size
        msg = simple_pb2.SimpleMessage()
        msg.key = "motor_position"
        msg.value = 0.12345
        protobuf_size = len(msg.SerializeToString())

        print(f"Protobuf: {protobuf_size} bytes")
        print(f"Protobuf is {json_size/protobuf_size:.1f}x smaller!")
    else:
        print("Protobuf: Not available (run protoc first)")


if __name__ == "__main__":
    print("Simple Protobuf vs JSON Test")
    print("=" * 40)

    # Compare message sizes
    compare_sizes()

    print("\nTo run the full test:")
    print("1. Generate protobuf code:")
    print("   python -m grpc_tools.protoc -I. --python_out=. simple.proto")
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
