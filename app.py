import streamlit as st
import time
import subprocess
from pathlib import Path

# =============== PAGE CONFIG ===============
st.set_page_config(
    page_title="VRU Detection & Forecasting Dashboard",
    page_icon="🚗",
    layout="wide"
)

# =============== HEADER ===============
st.title("🚦 Vulnerable Road User (VRU) Detection, Tracking & Forecasting System")

st.markdown("""
<style>
.big-font { font-size:20px !important; font-weight:500; }
.success-box {
    padding: 12px;
    border-radius: 10px;
    background-color: #E8F5E9;
    border-left: 5px solid #4CAF50;
    color: #1B5E20;
    margin-bottom: 20px;
}
.note-box {
    background-color: #F0F4FF;
    border-left: 5px solid #1E88E5;
    padding: 15px;
    border-radius: 10px;
    color: #0D47A1;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='success-box'>
<b>Objective:</b> Build an end-to-end pipeline to <b>Detect, Track, and Forecast Vulnerable Road Users (VRUs)</b> 
like pedestrians and cyclists using <b>Computer Vision and Deep Learning</b>.  
The project aims to simulate real-world road user behavior for <b>autonomous safety systems</b>.
</div>
""", unsafe_allow_html=True)

# =============== SIDEBAR NOTE ===============
st.sidebar.markdown("""
<div class='note-box'>
<h4>🧩 Project Summary</h4>
<p>This dashboard executes the <b>complete VRU pipeline</b>:</p>
<ul>
<li>Convert images → video</li>
<li>Detect & track VRUs (YOLO + DeepSORT)</li>
<li>Generate structured Excel output</li>
<li>Extract features for forecasting</li>
<li>Predict future trajectories (Transformer)</li>
<li>Visualize predicted vs actual motion</li>
</ul>
</div>
""", unsafe_allow_html=True)

# =============== USER INPUT BOX ===============
st.subheader("📂 Input Configuration")
st.write("Specify your **input dataset or file path** before running the pipeline.")

input_path = st.text_input(
    "Enter the full path to your dataset or input folder:",
    placeholder="e.g. C:/Users/YourName/Documents/VRU_Dataset"
)

st.write("Alternatively, upload a small sample file for testing:")
uploaded_file = st.file_uploader("Upload file (optional)", type=["csv", "xlsx", "mp4", "zip"])

if uploaded_file:
    temp_path = Path("uploaded_" + uploaded_file.name)
    temp_path.write_bytes(uploaded_file.getvalue())
    st.success(f"✅ Uploaded file saved as: {temp_path}")
    input_path = str(temp_path)

# =============== FILE PATH CONFIG ===============
scripts = [
    ("Video Creation", "video_creation.py"),
    ("YOLO + DeepSORT Tracking", "yolo_deepsort_tracker.py"),
    ("Excel Generation", "excel_generation.py"),
    ("Feature Engineering", "feature_engineering_forecasting.py"),
    ("Trajectory Forecasting (Transformer)", "vru_forecasting_transformer.py"),
    ("Trajectory Visualization", "animated_visualization.py")
]

# =============== RUN BUTTON ===============
st.markdown("---")
if st.button("🚀 Run Full VRU Pipeline"):
    if not input_path:
        st.warning("⚠️ Please specify an input path or upload a file before starting.")
    else:
        st.markdown(f"### 🛠 Running Full Pipeline on: `{input_path}`")
        progress = st.progress(0)
        status_placeholder = st.empty()
        total_scripts = len(scripts)

        for i, (label, script) in enumerate(scripts):
            progress.progress((i + 1) / total_scripts)
            status_placeholder.markdown(f"**▶️ Stage {i+1}/{total_scripts}: {label}...**")
            st.write(f"Running `{script}`...")
            try:
                # Optionally pass input_path as argument to scripts
                result = subprocess.run(["python", script, input_path], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success(f"✅ {label} completed successfully!")
                else:
                    st.error(f"❌ {label} failed!\n\n{result.stderr}")
                    break
            except Exception as e:
                st.error(f"⚠️ Error running {script}: {e}")
                break
            time.sleep(0.5)

        progress.progress(1.0)
        status_placeholder.markdown("### ✅ Pipeline Complete!")

        gif_path = Path("trajectory_comparison.gif")
        if gif_path.exists():
            st.image(str(gif_path), caption="Predicted vs. True Trajectories", use_container_width=True)
        else:
            st.info("ℹ️ Visualization not found — check `animated_visualization.py` output.")
else:
    st.info("Click **'🚀 Run Full VRU Pipeline'** to start processing all stages automatically.")

# Footer
st.markdown("---")
st.markdown("👩‍💻 **Developed for the VRU Detection Project — Forecasting Human Motion for Safer Roads.**")
