import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# --- CONFIG ---
input_file = r"C:\Users\cl502_23\Documents\outputsfinal\final_tracking_output2_log.xlsx"
output_file = r"C:\Users\cl502_23\Documents\outputsfinal\final_features.xlsx"
chunk_size = 100000  # number of rows per chunk (Excel can't stream like CSV)

# --- Remove output file if exists ---
if os.path.exists(output_file):
    os.remove(output_file)

# --- Columns for new features ---
feature_cols = ["delta_X", "delta_Y", "speed", "direction", "delta_time"]

# --- Function to process a single chunk ---
def process_chunk(df):
    # If TIMESTAMP column doesn't exist, create a pseudo one
    if "TIMESTAMP" not in df.columns:
        df["TIMESTAMP"] = pd.to_datetime(df["Frame_ID"], unit="s", errors="coerce")

    # Ensure numeric columns
    for col in ["X", "Y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["TIMESTAMP", "X", "Y"], inplace=True)
    df.sort_values(by=["Track_ID", "TIMESTAMP"], inplace=True)

    # Initialize new columns
    for col in feature_cols:
        df[col] = 0.0

    # Compute motion features per track
    for track_id, group in df.groupby("Track_ID"):
        group = group.sort_values("TIMESTAMP")

        delta_x = group["X"].diff().fillna(0)
        delta_y = group["Y"].diff().fillna(0)
        delta_t = group["TIMESTAMP"].diff().dt.total_seconds().fillna(0)
        delta_t = delta_t.replace(0, np.nan)

        speed = np.sqrt(delta_x**2 + delta_y**2) / delta_t
        speed = speed.fillna(0)
        direction = np.arctan2(delta_y, delta_x).fillna(0)

        df.loc[group.index, "delta_X"] = delta_x
        df.loc[group.index, "delta_Y"] = delta_y
        df.loc[group.index, "speed"] = speed
        df.loc[group.index, "direction"] = direction
        df.loc[group.index, "delta_time"] = delta_t.fillna(0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df["speed"] = np.clip(df["speed"], 0, df["speed"].quantile(0.99))

    return df

# --- Read and process Excel in chunks ---
print("🔹 Processing Excel dataset in chunks...")

# Load file in pieces (Excel can't stream like CSV)
df_full = pd.read_excel(input_file)
num_rows = len(df_full)
chunks = range(0, num_rows, chunk_size)

all_chunks = []
for start in tqdm(chunks, desc="Processing chunks"):
    end = start + chunk_size
    chunk = df_full.iloc[start:end].copy()
    processed_chunk = process_chunk(chunk)
    all_chunks.append(processed_chunk)

# --- Combine and write to Excel ---
final_df = pd.concat(all_chunks, ignore_index=True)
final_df.to_excel(output_file, index=False)

print(f"✅ Feature-engineered data saved to {output_file}")
