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

## 📋 How to Run
### 1️⃣ Install Requirements

First, install the necessary Python dependencies:

pip install -r requirements.txt

### 2️⃣ Run Detection

Run the detection script on a given video:

python detection.py --input video_path --output detections/

### 3️⃣ Run Tracking

Track the detected objects from the previous step:

python tracking.py --detections detections/ --output outputs/

### 4️⃣ Generate Features

Generate the required features for forecasting:

python feature_engineering.py

### 5️⃣ Train Transformer Forecasting Model

Train the forecasting model using the feature-engineered dataset:

python vru_forecasting_transformer.py

### 📊 Results
Prediction Task

Task: Predict the next (X, Y) position of VRUs (Vulnerable Road Users).

True_X	Pred_X	True_Y	Pred_Y
-0.1430	-0.0200	-1.1667	-0.7944
-1.1723	-1.2759	-1.1667	-1.0285
0.7800	0.6092	-2.5331	-1.0167

Loss: ~0.07 (Normalized MSE)

### Visualization:

The plot compares actual vs predicted trajectories using Matplotlib.

Predicted trajectories closely follow the actual VRU motion patterns, validating the model’s performance.

## 🧩 Key Highlights

Fully Modular Pipeline: Detection → Tracking → Forecasting

YOLO + DeepSORT + Transformers: Combines state-of-the-art techniques for object detection, tracking, and forecasting.

Scalable: Can be extended to multi-camera urban environments.

Context Integration: Combines depth and motion cues for better prediction context.

Feature Engineering: Dataset can be reused for other forecasting models like LSTM, CNN-LSTM, and GRU.

## 🔮 Future Improvements

Multi-Step Forecasting: Expand to forecast longer time horizons.

Scene Semantics: Integrate map or lane context for more precise predictions.

Real-Time System: Deploy the model using ONNX or TorchScript for real-time predictions.

Uncertainty Estimation: Implement techniques to estimate uncertainty for safer predictions.

## 👩‍💻 Contributors

Aaditya Ramrame, Aarushi Mathur, Abhav Bhanot, Anusri Kadam

Deep Learning Project

Supervised by: Faculty, Department of Artificial Intelligence and Machine Learning
