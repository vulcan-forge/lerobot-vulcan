#!/usr/bin/env python3
"""
Protobuf compilation script for Sourccey robot.
This script compiles all .proto files into Python modules.
"""

import importlib
import os
import sys
import subprocess
import pathlib
from typing import List, Tuple

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import grpc_tools
        return True
    except ImportError:
        print("Error: grpc_tools not found. Please install it with:")
        print("pip install grpcio-tools")
        return False

def get_proto_files() -> List[str]:
    """Get list of .proto files in the current directory."""
    proto_files = []
    current_dir = pathlib.Path(__file__).parent

    for file in current_dir.glob("*.proto"):
        proto_files.append(file.name)

    return sorted(proto_files)

def compile_proto_files(proto_files: List[str], output_dir: str = "generated") -> Tuple[bool, List[str]]:
    """
    Compile protobuf files to Python modules.

    Args:
        proto_files: List of .proto file names
        output_dir: Output directory for generated Python files

    Returns:
        Tuple of (success, list of generated files)
    """
    current_dir = pathlib.Path(__file__).parent
    output_path = current_dir / output_dir

    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)

    generated_files = []
    success = True

    for proto_file in proto_files:
        print(f"Compiling {proto_file}...")

        try:
            # Run protoc command
            cmd = [
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I{current_dir}",
                f"--python_out={output_path}",
                str(current_dir / proto_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Get the generated Python file name
            python_file = proto_file.replace(".proto", "_pb2.py")
            generated_file = output_path / python_file

            if generated_file.exists():
                generated_files.append(str(generated_file))
                print(f"  ✓ Generated: {generated_file}")
            else:
                print(f"  ✗ Failed to generate: {generated_file}")
                success = False

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error compiling {proto_file}:")
            print(f"    {e.stderr}")
            success = False
        except Exception as e:
            print(f"  ✗ Unexpected error compiling {proto_file}: {e}")
            success = False

    return success, generated_files

def create_init_file(output_dir: str = "generated"):
    """Create __init__.py file in the generated directory."""
    current_dir = pathlib.Path(__file__).parent
    output_path = current_dir / output_dir
    init_file = output_path / "__init__.py"

    init_content = '''"""
Generated protobuf modules for Sourccey robot.
This module contains the compiled protobuf classes.
"""

# Import all generated modules
try:
    from . import sourccey_pb2

    __all__ = [
        'sourccey_pb2'
    ]
except ImportError as e:
    print(f"Warning: Could not import protobuf modules: {e}")
    print("Make sure to run compile.py first to generate the protobuf files.")
'''

    with open(init_file, 'w') as f:
        f.write(init_content)

    print(f"✓ Created: {init_file}")

def verify_compilation(generated_files: List[str]) -> bool:
    """Verify that the generated files can be imported."""
    print("\nVerifying compilation...")

    success = True
    generated_dir = pathlib.Path(generated_files[0]).parent

    # Add the generated directory to Python path temporarily
    original_path = sys.path.copy()
    sys.path.insert(0, str(generated_dir))

    try:
        for file_path in generated_files:
            try:
                # Use a simpler import approach
                module_name = pathlib.Path(file_path).stem

                # Try to import the module directly
                module = importlib.import_module(module_name)
                print(f"  ✓ {module_name} imports successfully")
            except Exception as e:
                print(f"  ✗ {module_name} failed to import: {e}")
                success = False
    finally:
        # Restore original Python path
        sys.path = original_path

    return success

def test_protobuf_functionality():
    """Test that the protobuf classes can be instantiated and used."""
    print("\nTesting protobuf functionality...")

    generated_dir = pathlib.Path(__file__).parent / "generated"
    original_path = sys.path.copy()
    sys.path.insert(0, str(generated_dir))

    try:
        # Test sourccey_pb2
        import sourccey_pb2

        # Test MotorJoint
        motor_joint = sourccey_pb2.MotorJoint()
        motor_joint.shoulder_pan = 1.0
        motor_joint.shoulder_lift = 2.0
        motor_joint.elbow_flex = 3.0
        motor_joint.wrist_flex = 4.0
        motor_joint.wrist_roll = 5.0
        motor_joint.gripper = 6.0
        print("  ✓ MotorJoint can be created and populated")

        # Test BaseVelocity
        base_velocity = sourccey_pb2.BaseVelocity()
        base_velocity.x_vel = 1.0
        base_velocity.y_vel = 2.0
        base_velocity.z_vel = 3.0
        base_velocity.theta_vel = 4.0
        print("  ✓ BaseVelocity can be created and populated")

        # Test CameraImage
        camera_image = sourccey_pb2.CameraImage()
        camera_image.name = "test_camera"
        camera_image.jpeg_data = b"fake_jpeg_data"
        camera_image.width = 640
        camera_image.height = 480
        camera_image.quality = 90
        camera_image.timestamp = 1234567890.0
        print("  ✓ CameraImage can be created and populated")

        # Test SourcceyRobotState
        robot_state = sourccey_pb2.SourcceyRobotState()
        robot_state.left_arm_joints.CopyFrom(motor_joint)
        robot_state.right_arm_joints.CopyFrom(motor_joint)
        robot_state.base_velocity.CopyFrom(base_velocity)
        robot_state.left_wrist_camera.append(camera_image)
        robot_state.right_wrist_camera.append(camera_image)
        robot_state.left_front_camera.append(camera_image)
        robot_state.right_front_camera.append(camera_image)
        print("  ✓ SourcceyRobotState can be created and populated")

        # Test SourcceyRobotAction
        robot_action = sourccey_pb2.SourcceyRobotAction()
        robot_action.left_arm_target_joints.CopyFrom(motor_joint)
        robot_action.right_arm_target_joints.CopyFrom(motor_joint)
        robot_action.base_target_velocity.CopyFrom(base_velocity)
        print("  ✓ SourcceyRobotAction can be created and populated")

        # Test serialization
        serialized_state = robot_state.SerializeToString()
        deserialized_state = sourccey_pb2.SourcceyRobotState()
        deserialized_state.ParseFromString(serialized_state)
        print("  ✓ Serialization and deserialization works")

        return True

    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False
    finally:
        sys.path = original_path

def main():
    """Main compilation function."""
    print("Sourccey Protobuf Compiler")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Get proto files
    proto_files = get_proto_files()
    if not proto_files:
        print("No .proto files found in current directory.")
        sys.exit(1)

    print(f"Found {len(proto_files)} proto files:")
    for proto_file in proto_files:
        print(f"  - {proto_file}")

    # Compile proto files
    print(f"\nCompiling proto files...")
    success, generated_files = compile_proto_files(proto_files)

    if not success:
        print("\n❌ Compilation failed!")
        sys.exit(1)

    # Create __init__.py file
    create_init_file()

    # Verify compilation
    verify_success = verify_compilation(generated_files)

    # Test functionality
    functionality_success = test_protobuf_functionality()

    if verify_success and functionality_success:
        print("\n✅ All protobuf files compiled successfully!")
        print(f"Generated {len(generated_files)} Python modules in 'generated/' directory.")
        print("\nYou can now use the protobuf methods in Sourccey classes.")
    else:
        print("\n⚠️  Compilation completed but some tests failed.")
        if not verify_success:
            print("  - Module import verification failed")
        if not functionality_success:
            print("  - Functionality testing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
