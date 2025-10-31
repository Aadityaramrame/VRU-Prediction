import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

# ==== CONFIG ====
video_path = r"C:\Users\cl502_23\Documents\outputsfinal\final_tracking_output2.avi"
output_excel = r"C:\Users\cl502_23\Documents\outputsfinal\final_tracking_output2_log.xlsx"
model_name = "yolov8n.pt"  # or yolov8m.pt for better accuracy

# ==== LOAD MODEL ====
model = YOLO(model_name)

# ==== DETECTION CLASSES (only these) ====
target_classes = ["car", "person", "bicycle", "motorcycle"]

# ==== VIDEO INPUT ====
cap = cv2.VideoCapture(video_path)
frame_id = 0

# For logging
records = []

# Tracker dictionary: keeps last known positions to avoid ID duplication
object_tracker = defaultdict(lambda: {"centroid": None, "track_id": None})
track_counter = 0

# ==== HELPER ====
def get_depth_status(bbox, frame_height):
    """Estimate relative closeness based on bounding box height."""
    _, y, _, h = bbox
    rel_height = h / frame_height
    if rel_height > 0.4:
        return "Close"
    elif rel_height > 0.2:
        return "Medium"
    else:
        return "Far"

# ==== PROCESS FRAMES ====
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    frame_height, frame_width = frame.shape[:2]

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = model.names[int(box.cls)]
        if cls not in target_classes:
            continue

        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # ---- Simple manual tracking (prevent duplicate IDs) ----
        found_match = False
        for tid, data in object_tracker.items():
            prev_cx, prev_cy = data["centroid"] if data["centroid"] else (None, None)
            if prev_cx is not None and abs(cx - prev_cx) < 40 and abs(cy - prev_cy) < 40:
                object_tracker[tid]["centroid"] = (cx, cy)
                track_id = tid
                found_match = True
                break

        if not found_match:
            track_counter += 1
            track_id = track_counter
            object_tracker[track_id]["centroid"] = (cx, cy)

        depth_status = get_depth_status((x1, y1, w, h), frame_height)

        records.append({
            "Frame_ID": frame_id,
            "Track_ID": track_id,
            "Class_Name": cls,
            "X": x1,
            "Y": y1,
            "Width": w,
            "Height": h,
            "Depth_Status": depth_status,
            "Confidence": round(conf, 3)
        })

cap.release()

# ==== SAVE TO EXCEL ====
df = pd.DataFrame(records)
df.to_excel(output_excel, index=False)
print(f"✅ Tracking log saved to: {output_excel}")
