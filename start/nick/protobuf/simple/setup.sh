#!/bin/bash
echo "Setting up simple protobuf test..."

# Install protobuf if not already installed
uv pip install protobuf grpcio-tools

# Generate Python code from .proto file
uv run python -m grpc_tools.protoc -I. --python_out=. simple.proto

echo "Done! Now you can run:"
echo "python test.py"
