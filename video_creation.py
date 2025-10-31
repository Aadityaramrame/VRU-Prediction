import cv2
import os
from natsort import natsorted

# --- CONFIG ---
IMAGE_FOLDER = r"C:\Users\cl502_23\Downloads\DL_project\detection_output\ring_side_right"
OUTPUT_VIDEO = r"C:/Users/cl502_23/Downloads/DL_project/detection_output/video_right.avi"
FPS = 10  # 🔹 10 FPS = smoother, slower playback
# You can try 5 for extra slow or 15 for realistic car movement

# --- Get all image file names ---
images = [img for img in os.listdir(IMAGE_FOLDER) if img.lower().endswith((".jpg", ".png", ".jpeg"))]
images = natsorted(images)  # sort in order like frame1, frame2, frame10, etc.

if not images:
    raise ValueError("No image files found in the folder!")

# --- Read first image to get video size ---
first_frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
height, width, layers = first_frame.shape
size = (width, height)

# --- Create video writer ---
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"XVID"), FPS, size)

# --- Write frames ---
for i, image_name in enumerate(images):
    img_path = os.path.join(IMAGE_FOLDER, image_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"⚠️ Skipping unreadable image: {image_name}")
        continue
    out.write(frame)
    if i % 100 == 0:
        print(f"Processed {i}/{len(images)} frames")

# --- Finish ---
out.release()
print(f"✅ Video saved to {OUTPUT_VIDEO}")
