"""
Generated protobuf modules for Sourccey robot.
This module contains the compiled protobuf classes.
"""

# Import all generated modules
try:
    from . import sourccey_common_pb2
    from . import sourccey_follower_pb2
    from . import sourccey_robot_pb2

    __all__ = [
        'sourccey_common_pb2',
        'sourccey_follower_pb2',
        'sourccey_robot_pb2'
    ]
except ImportError as e:
    print(f"Warning: Could not import protobuf modules: {e}")
    print("Make sure to run compile.py first to generate the protobuf files.")
