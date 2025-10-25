import cv2

cap = cv2.VideoCapture("tests/artifacts/cameras/image_160x120.png", cv2.CAP_FFMPEG)
print("FFmpeg open:", cap.isOpened())
cap.release()

cap2 = cv2.VideoCapture("tests/artifacts/cameras/image_160x120.png", cv2.CAP_DSHOW)
print("DirectShow open:", cap2.isOpened())
cap2.release()