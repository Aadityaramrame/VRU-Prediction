import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1️⃣ Hardcoded training + test losses
# -------------------------
epochs = np.arange(1, 21)
train_loss = [
    0.2707, 0.1240, 0.1072, 0.0995, 0.0939, 0.0896, 0.0894, 0.0840, 0.0785, 0.0766,
    0.0765, 0.0716, 0.0700, 0.0704, 0.0668, 0.0687, 0.0667, 0.0688, 0.0675, 0.0661
]
test_loss = [0.0764] * len(epochs)

# -------------------------
# 2️⃣ Sample true/predicted coordinates (from your output)
# -------------------------
true_x = [-0.143, -1.173, 2.780, 1.467, -1.179, -0.368, 0.643, -1.683, -0.984, -1.342]
pred_x = [-0.794, -1.217, -1.616, -1.184, 0.437, 0.257, -0.074, -0.247, -0.479, -0.043]

true_y = [-0.021, -1.757, 0.692, 1.542, -1.158, -1.377, -0.190, -0.910, -1.924, 0.361]
pred_y = [-1.167, -1.167, -1.534, 0.184, 2.274, -1.044, 0.195, -0.947, 0.470, 0.382]

# -------------------------
# 3️⃣ Plot 1: Loss curve
# -------------------------
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_loss, marker='o', label='Train Loss', linewidth=2)
plt.plot(epochs, test_loss, '--', label='Test Loss (0.0764)', linewidth=2)
plt.title('Training and Test Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------
# 4️⃣ Plot 2: True vs Predicted Trajectory
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(true_x, true_y, c='green', label='True (X,Y)', s=60)
plt.scatter(pred_x, pred_y, c='red', label='Predicted (X,Y)', s=60, marker='x')
plt.title('True vs Predicted Next Positions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()
