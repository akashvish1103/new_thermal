# Using the same data and gradient descent process as before, we will create a 3D visualization of the loss surface and
#  the trajectory of the parameters (m and b) during training.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# CREATE SAMPLE DATA
# ==========================================================

np.random.seed(42)

x = np.linspace(0, 10, 50)
y = 42.55 * x + 0.35 + np.random.randn(50) * 15

# ==========================================================
# GRADIENT DESCENT
# ==========================================================

random_m = 250
random_b = 2

epochs = 100
learning_rate = 0.0001

m_history = []
b_history = []
loss_history = []

for epoch in range(epochs):

    predictions = random_m * x + random_b

    loss = np.sum((y - predictions) ** 2)

    m_history.append(random_m)
    b_history.append(random_b)
    loss_history.append(loss)

    slope_b = -2 * np.sum(y - predictions)

    slope_m = -2 * np.sum(
        (y - predictions) * x
    )

    random_b = random_b - learning_rate * slope_b
    random_m = random_m - learning_rate * slope_m

print(f"Final m = {random_m}")
print(f"Final b = {random_b}")

# ==========================================================
# CREATE LOSS SURFACE
# ==========================================================

m_min = min(m_history) - 20
m_max = max(m_history) + 20

b_min = min(b_history) - 20
b_max = max(b_history) + 20

m_range = np.linspace(m_min, m_max, 80)
b_range = np.linspace(b_min, b_max, 80)

M, B = np.meshgrid(m_range, b_range)

Loss = np.zeros(M.shape)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):

        m = M[i, j]
        b = B[i, j]

        pred = m * x + b

        Loss[i, j] = np.sum((y - pred) ** 2)

# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(figsize=(16, 7))

# -----------------------
# LEFT: Regression Line
# -----------------------

ax1 = fig.add_subplot(1, 2, 1)

ax1.scatter(x, y, color='blue', label='Data')

line, = ax1.plot(
    [],
    [],
    color='red',
    linewidth=3,
    label='Gradient Descent Line'
)

ax1.set_title("Linear Regression")
ax1.legend()

# -----------------------
# RIGHT: 3D Loss Surface
# -----------------------

ax2 = fig.add_subplot(1, 2, 2, projection='3d')

surface = ax2.plot_surface(
    M,
    B,
    Loss,
    cmap='viridis',
    alpha=0.7
)

trajectory, = ax2.plot(
    [],
    [],
    [],
    color='red',
    linewidth=3
)

point, = ax2.plot(
    [],
    [],
    [],
    'ro',
    markersize=8
)

ax2.set_xlabel("m")
ax2.set_ylabel("b")
ax2.set_zlabel("Loss")

ax2.set_title("3D Loss Surface")

# ==========================================================
# ANIMATION FUNCTION
# ==========================================================

def update(frame):

    # -------------------------
    # LEFT PANEL
    # -------------------------

    current_m = m_history[frame]
    current_b = b_history[frame]

    y_pred = current_m * x + current_b

    line.set_data(x, y_pred)

    ax1.set_title(
        f"Epoch = {frame}\n"
        f"m = {current_m:.2f}, b = {current_b:.2f}"
    )

    # -------------------------
    # RIGHT PANEL
    # -------------------------

    trajectory.set_data(
        m_history[:frame + 1],
        b_history[:frame + 1]
    )

    trajectory.set_3d_properties(
        loss_history[:frame + 1]
    )

    point.set_data(
        [current_m],
        [current_b]
    )

    point.set_3d_properties(
        [loss_history[frame]]
    )

    return line, trajectory, point


# ==========================================================
# RUN ANIMATION
# ==========================================================

ani = FuncAnimation(
    fig,
    update,
    frames=len(m_history),
    interval=150,
    repeat=False
)

plt.tight_layout()
plt.show()