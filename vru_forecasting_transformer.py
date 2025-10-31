import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ==========================
# LOAD DATA
# ==========================
# ✅ Update this to your actual file path
df = pd.read_excel(r"C:\Users\cl502_23\Documents\outputsfinal\finalfinaloutputs\final_features.xlsx")

# Normalize column names
df.columns = df.columns.str.lower()

# Sort for temporal order
df = df.sort_values(by=["track_id", "frame_id"])

# Drop unnecessary columns
keep_cols = ["x", "y", "speed", "direction", "delta_x", "delta_y", "confidence", "track_id", "frame_id"]
df = df[[c for c in keep_cols if c in df.columns]]

# Normalize numeric columns
num_cols = ["x", "y", "speed", "direction", "delta_x", "delta_y", "confidence"]
df[num_cols] = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-8)

# ==========================
# DATASET CLASS
# ==========================
class VRUDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        self.tracks = []
        grouped = df.groupby("track_id")
        for tid, group in grouped:
            arr = group[num_cols].values
            if len(arr) > seq_len:
                for i in range(len(arr) - seq_len):
                    x_seq = arr[i:i+seq_len]
                    y_seq = arr[i+seq_len, :2]  # predict next (x,y)
                    self.tracks.append((x_seq, y_seq))

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        x, y = self.tracks[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Create dataset
dataset = VRUDataset(df, SEQ_LEN)

# Train-test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==========================
# TRANSFORMER MODEL
# ==========================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=256),
            num_layers=num_layers
        )
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.input_fc(x) + self.pos_emb
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.output_fc(x)

# Instantiate model
model = TransformerModel(input_dim=len(num_cols), seq_len=SEQ_LEN).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ==========================
# TRAINING
# ==========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_dl):.4f}")

# ==========================
# EVALUATION
# ==========================
model.eval()
with torch.no_grad():
    total_loss = 0
    for xb, yb in test_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = criterion(preds, yb)
        total_loss += loss.item()
    print(f"✅ Test Loss: {total_loss/len(test_dl):.4f}")

# ==========================
# SAMPLE PREDICTION
# ==========================
xb, yb = next(iter(test_dl))
with torch.no_grad():
    preds = model(xb.to(DEVICE)).cpu().numpy()

print("\nSample Predictions (Next X,Y):")
print(pd.DataFrame({
    "True_X": yb[:,0].numpy(),
    "Pred_X": preds[:,0],
    "True_Y": yb[:,1].numpy(),
    "Pred_Y": preds[:,1],
}).head(10))
