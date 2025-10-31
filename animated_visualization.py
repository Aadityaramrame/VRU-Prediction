import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Hardcoded example trajectory data ---
frames = 50

# True trajectory (green)
true_x = np.linspace(0, 10, frames)
true_y = np.sin(true_x) + 0.5 * np.random.randn(frames) * 0.1  # noisy

# Predicted trajectory (red)
pred_x = np.linspace(0, 10, frames)
pred_y = np.sin(pred_x + 0.2) + 0.5 * np.random.randn(frames) * 0.1

# --- Setup plot ---
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
ax.set_title("True vs Predicted Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, linestyle="--", alpha=0.6)

# Create line and point objects for animation
true_line, = ax.plot([], [], color='green', label='True Path')
pred_line, = ax.plot([], [], color='red', label='Predicted Path')
true_point, = ax.plot([], [], 'go', markersize=8)
pred_point, = ax.plot([], [], 'ro', markersize=8)
ax.legend(loc="upper right")

# --- Update function ---
def update(frame):
    true_point.set_data([true_x[frame]], [true_y[frame]])
    pred_point.set_data([pred_x[frame]], [pred_y[frame]])
    true_line.set_data(true_x[:frame+1], true_y[:frame+1])
    pred_line.set_data(pred_x[:frame+1], pred_y[:frame+1])
    return true_point, pred_point, true_line, pred_line

# --- Animate ---
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

# --- Save as GIF (works everywhere) ---
ani.save("trajectory_comparison.gif", writer='pillow', fps=10)

print("✅ Animation saved as 'trajectory_comparison.gif' in your current folder!")

plt.show()
