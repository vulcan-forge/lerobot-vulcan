import cv2
import sys

cameras = [
    ('FrontLeft', '/dev/cameraFrontLeft'),
    ('FrontRight', '/dev/cameraFrontRight'), 
    ('WristLeft', '/dev/cameraWristLeft'),
    ('WristRight', '/dev/cameraWristRight')
]

for name, path in cameras:
    print(f'Testing {name}: {path}')
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'  ✅ {name} working - frame shape: {frame.shape}')
        else:
            print(f'  ❌ {name} opened but read failed')
        cap.release()
    else:
        print(f'  ❌ {name} failed to open')
    print()