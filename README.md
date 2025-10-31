# 🧠 VRU Detection, Tracking & Forecasting using Transformers

## 📌 Overview  
This project presents an **end-to-end pipeline for Vulnerable Road User (VRU) understanding** — from detection and tracking to future trajectory prediction.  
It integrates **Deep Learning, Computer Vision, and Sequential Forecasting** to accurately **predict the next (X, Y) positions** of VRUs such as pedestrians, cyclists, and vehicles using **Transformer-based models**.

---

## 🚦 Project Pipeline  

### 1️⃣ Detection — *YOLO-based VRU Identification*  
- Multi-camera video input processed using **YOLO (You Only Look Once)** for object detection.  
- Each VRU (pedestrian, cyclist, vehicle) is localized with bounding boxes and class labels.  
- Depth estimation integrated to improve spatial awareness.

### 2️⃣ Tracking — *Multi-object Tracking with DeepSORT*  
- Detected VRUs are tracked across frames using **DeepSORT**, which combines motion and appearance features.  
- Each entity is assigned a **unique TRACK_ID** for continuous observation.  
- Tracking results are exported as CSV/Excel logs containing position and motion data.

### 3️⃣ Feature Engineering  
- Using the tracking logs, motion-based features are computed:
  - `delta_X`, `delta_Y` → displacement  
  - `speed` → velocity magnitude  
  - `direction` → movement angle  
  - `delta_time` → frame time difference  
  - `confidence` → detection reliability  
- These features form the input for trajectory forecasting.  
- Output stored as: `final_features.xlsx`.

### 4️⃣ Forecasting — *Transformer-based VRU Trajectory Prediction*  
- A **PyTorch Transformer model** predicts the next `(X, Y)` positions based on the past sequence of movements.  
- The model learns temporal patterns in VRU trajectories, offering robust short-term forecasts even under occlusions or noise.  
- Outputs include:
  - Predicted vs True next positions  
  - Visualization plots comparing trajectories  

---

## ⚙️ Technologies Used  

| Category | Libraries / Tools |
|-----------|------------------|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | DeepSORT |
| Feature Engineering | Pandas, NumPy |
| Forecasting | PyTorch (Transformer Encoder) |
| Visualization | Matplotlib, Seaborn |
| Misc | tqdm, OpenCV |

---

## 🚀 How to Run  

### 1️⃣ Install Requirements  
```bash
pip install -r requirements.txt

2️⃣ Run Detection
python detection.py --input video_path --output detections/

3️⃣ Run Tracking
python tracking.py --detections detections/ --output outputs/

4️⃣ Generate Features
python feature_engineering.py

5️⃣ Train Transformer Forecasting Model
python vru_forecasting_transformer.py

📊 Results

Prediction Task: Next (X, Y) position forecasting

True_X	Pred_X	True_Y	Pred_Y
-0.1430	-0.0200	-1.1667	-0.7944
-1.1723	-1.2759	-1.1667	-1.0285
0.7800	0.6092	-2.5331	-1.0167

Loss: ~0.07 (Normalized MSE)

Visualization:

Trajectories of actual vs predicted motion plotted using Matplotlib.

Predicted trajectories closely follow true VRU motion patterns, validating the model’s learning capability.

🧩 Key Highlights

Fully modular pipeline (Detection → Tracking → Forecasting)

Combines YOLO + DeepSORT + Transformers

Scalable to multi-camera urban scenes

Integrates depth and motion cues for better context

Feature-engineered dataset reusable for other forecasting models (LSTM, CNN-LSTM, GRU)

🔮 Future Improvements

Multi-step forecasting for longer prediction horizons

Integrate map or lane context (scene semantics)

Deploy real-time system using ONNX or TorchScript

Introduce uncertainty estimation for safer prediction

👩‍💻 Contributors

Anusri Kadam & Team
B.Tech – Project-Based Learning (FinTech & AI Vision Domain)
Supervised by: Faculty, Department of Computer Engineering
