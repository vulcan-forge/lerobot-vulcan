#!/bin/bash
echo "Setting up full robot protobuf test..."

# Install dependencies
pip install protobuf grpcio-tools opencv-python numpy

# Generate Python code from .proto file
python -m grpc_tools.protoc -I. --python_out=. full_robot.proto

echo "Done! Now you can run:"
echo "python test.py"
