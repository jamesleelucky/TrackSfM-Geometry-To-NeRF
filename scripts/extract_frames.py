import cv2
import os

video_path = "data/video/capture.mov"
out_dir = "data/frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
idx = 0
frame_skip = 3  # KEEP THIS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if idx % frame_skip == 0:
        cv2.imwrite(f"{out_dir}/{idx:03d}.png", frame)

    idx += 1

cap.release()
print("Frame extraction complete.")
