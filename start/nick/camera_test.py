import cv2

# Test different indices
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i} can read frames")
        else:
            print(f"Camera {i} detected but can't read")
        cap.release()
    else:
        print(f"Camera {i} not available")
