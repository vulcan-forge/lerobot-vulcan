#!/bin/bash
echo "Setting up large protobuf test..."

# Install protobuf if not already installed
pip install protobuf grpcio-tools

# Generate Python code from .proto file
python -m grpc_tools.protoc -I. --python_out=. robot_data.proto

echo "Done! Now you can run:"
echo "python test.py"
