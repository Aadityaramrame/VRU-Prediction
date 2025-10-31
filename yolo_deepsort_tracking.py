import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\cl502_23\Downloads\DL_project\detection_output\Yolo_output\video_right.avi"   # Input video
OUTPUT_PATH = r"C:/Users/cl502_23/Downloads/DL_project/detection_output/output_tracked_right.avi"  # Output video
YOLO_MODEL_PATH = "yolov8n.pt" 

# --- Load YOLOv8 model ---
print("🚀 Loading YOLOv8 model...")
model = YOLO(YOLO_MODEL_PATH)

# --- Initialize DeepSORT tracker ---
tracker = DeepSort(
    max_age=7,                # how long to keep "lost" tracks before deleting
    n_init=3,                  # number of frames before a track is confirmed
    nms_max_overlap=1.0,       # non-max suppression threshold
    max_cosine_distance=0.3,   # appearance similarity threshold
    embedder="mobilenet",      # feature extractor
    half=True,                 # use FP16 for speed
)

# --- Load Video ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"❌ Cannot open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Output Video Writer ---
out = cv2.VideoWriter(
    OUTPUT_PATH, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
)

print("🎥 Starting Detection + Tracking...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO Detection ---
    results = model(frame, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Filter detections (keep people, bicycles, etc.)
            if cls_id in [0, 1, 2]:  # 0-person, 1-bicycle, 2-car (change per need)
                detections.append(([x1, y1, w, h], conf, "VRU"))

    # --- DeepSORT Tracking ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- Draw Tracks ---
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Show + Write Frame ---
    cv2.imshow("YOLOv8 + DeepSORT", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Tracking complete. Output saved to: {OUTPUT_PATH}")
