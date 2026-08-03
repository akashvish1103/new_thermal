import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Setup Dummy Data ---
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

st.title("Linear Regression: Gradient Descent Visualizer")

# --- Sidebar Parameters ---
st.sidebar.header("Hyperparameters")
epochs = st.sidebar.slider("Epochs", 1, 2000, 50)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.01, format="%.4f")
# We divide by length to normalize the gradient
n = len(x)

# --- Initialize Parameters ---
m = 250.0  # Initial guess
b = 2.0    # Initial guess

# --- Visualization Setup ---
fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', alpha=0.5, label='Data')
line, = ax.plot([], [], color='red', linewidth=2, label='Current Model')
ax.set_xlim(0, 2)
ax.set_ylim(0, 15)
ax.legend()

plot_placeholder = st.empty()
status_text = st.empty()

# --- Gradient Descent Logic ---
if st.button("Run Training"):
    for i in range(epochs):
        # Calculate gradients (using Mean Squared Error derivative)
        y_pred = m * x.ravel() + b
        
        # Derivatives of Mean Squared Error
        db = (-2/n) * np.sum(y.ravel() - y_pred)
        dm = (-2/n) * np.sum(x.ravel() * (y.ravel() - y_pred))

        # Update parameters
        b = b - (learning_rate * db)
        m = m - (learning_rate * dm)

        # Update plot
        line.set_data(x, m * x + b)
        plot_placeholder.pyplot(fig)
        
        status_text.text(f"Epoch: {i+1}/{epochs} | Slope (m): {m:.2f} | Intercept (b): {b:.2f}")
        
        # Slow down slightly to watch the animation
        time.sleep(0.01)

    st.success("Training Complete!")