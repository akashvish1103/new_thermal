# Gradient Descent Visualization Tool for Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# ── Generate data (same as your notebook) ────────────────────────────────────
X, y = make_regression(n_samples=100, n_features=1, noise=3, random_state=42)      # make changes here for different data
x = X.ravel()

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# ── Default GD parameters (same as your notebook) ────────────────────────────
INIT_M    = 250
INIT_B    = 2
INIT_LR   = 0.001
INIT_EP   = 100

# ── Pre-compute full GD history ───────────────────────────────────────────────
def compute_history(m0, b0, lr, epochs):
    m, b = m0, b0
    hist = [(m, b)]
    for _ in range(epochs):
        residual  = y - m * x - b
        slope_m   = -2 * np.sum(residual * x)
        slope_b   = -2 * np.sum(residual)
        m = m - lr * slope_m
        b = b - lr * slope_b
        hist.append((m, b))
    return hist

history = compute_history(INIT_M, INIT_B, INIT_LR, INIT_EP)

# ── Plot setup ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.38, right=0.97, top=0.93)

x_line = np.linspace(x.min() - 0.3, x.max() + 0.3, 200)

# Static elements
scatter = ax.scatter(x, y, color='blue', s=20, zorder=5, label='Data points')
line_sklearn, = ax.plot(
    x_line, model.coef_[0] * x_line + model.intercept_,
    color='grey', linewidth=2, linestyle='--', label='Sklearn best fit (grey)'
)

# Dynamic elements
trail_lines = []          # black semi-transparent lines, added each epoch
line_start, = ax.plot([], [], color='red',   linewidth=2,   label='GD start (red)')
line_cur,   = ax.plot([], [], color='black', linewidth=1.5, alpha=0.8, label='GD current (black)')
line_final, = ax.plot([], [], color='green', linewidth=2.5, label='GD final (green)', zorder=4)

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Gradient Descent — Linear Regression', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)

epoch_text = ax.text(
    0.98, 0.05, '', transform=ax.transAxes,
    ha='right', va='bottom', fontsize=10, color='#333'
)

def clear_trail():
    for l in trail_lines:
        l.remove()
    trail_lines.clear()

def draw_epoch(step):
    m, b = history[step]
    m0_val, b0_val = history[0]
    m_final, b_final = history[-1]

    line_start.set_data(x_line, m0_val * x_line + b0_val)

    clear_trail()
    for i in range(1, step):
        mi, bi = history[i]
        alpha = max(0.05, min(0.45, 0.08 + 0.35 * i / len(history)))
        l, = ax.plot(x_line, mi * x_line + bi, color='black', linewidth=0.8, alpha=alpha)
        trail_lines.append(l)

    if step == len(history) - 1:
        line_cur.set_data([], [])
        line_final.set_data(x_line, m_final * x_line + b_final)
    else:
        line_cur.set_data(x_line, m * x_line + b)
        line_final.set_data([], [])

    loss = float(np.mean((y - m * x - b) ** 2))
    epoch_text.set_text(
        f'Epoch {step}/{len(history)-1}   m={m:.3f}   b={b:.3f}   MSE={loss:.2f}'
    )
    fig.canvas.draw_idle()

draw_epoch(0)

# ── Widgets ───────────────────────────────────────────────────────────────────
ax_ep  = plt.axes([0.10, 0.28, 0.78, 0.03])
ax_lr  = plt.axes([0.10, 0.23, 0.78, 0.03])
ax_m0  = plt.axes([0.10, 0.18, 0.78, 0.03])
ax_b0  = plt.axes([0.10, 0.13, 0.78, 0.03])

sl_ep  = Slider(ax_ep,  'Epochs',         10,  300, valinit=INIT_EP,  valstep=5)
sl_lr  = Slider(ax_lr,  'Learning rate', 1e-4, 0.1, valinit=INIT_LR)
sl_m0  = Slider(ax_m0,  'Initial m₀',   -300,  300, valinit=INIT_M,   valstep=10)
sl_b0  = Slider(ax_b0,  'Initial b₀',    -50,   50, valinit=INIT_B,   valstep=1)

ax_step_sl = plt.axes([0.10, 0.07, 0.60, 0.03])
sl_step    = Slider(ax_step_sl, 'Epoch →', 0, INIT_EP, valinit=0, valstep=1)

ax_btn_reset = plt.axes([0.10, 0.01, 0.12, 0.05])
ax_btn_prev  = plt.axes([0.24, 0.01, 0.12, 0.05])
ax_btn_step  = plt.axes([0.38, 0.01, 0.12, 0.05])
ax_btn_play  = plt.axes([0.52, 0.01, 0.12, 0.05])
ax_btn_end   = plt.axes([0.66, 0.01, 0.12, 0.05])

btn_reset = Button(ax_btn_reset, 'Reset')
btn_prev  = Button(ax_btn_prev,  '◀ Prev')
btn_step  = Button(ax_btn_step,  'Step ▶')
btn_play  = Button(ax_btn_play,  'Play')
btn_end   = Button(ax_btn_end,   'End ⏭')

state = {'step': 0, 'playing': False, 'anim': None}

def rebuild(_=None):
    global history
    state['playing'] = False
    btn_play.label.set_text('Play')
    if state['anim']:
        state['anim'].event_source.stop()
    history = compute_history(
        int(sl_m0.val), int(sl_b0.val),
        sl_lr.val, int(sl_ep.val)
    )
    sl_step.valmax = len(history) - 1
    sl_step.ax.set_xlim(0, len(history) - 1)
    state['step'] = 0
    sl_step.set_val(0)
    clear_trail()
    draw_epoch(0)

sl_ep.on_changed(rebuild)
sl_lr.on_changed(rebuild)
sl_m0.on_changed(rebuild)
sl_b0.on_changed(rebuild)

def on_step_slider(val):
    state['step'] = int(val)
    draw_epoch(state['step'])
sl_step.on_changed(on_step_slider)

def on_reset(event):
    rebuild()
    sl_step.set_val(0)

def on_prev(event):
    state['step'] = max(0, state['step'] - 1)
    sl_step.set_val(state['step'])

def on_step_btn(event):
    if state['step'] < len(history) - 1:
        state['step'] += 1
        sl_step.set_val(state['step'])

def animate(frame):
    if not state['playing']:
        return
    if state['step'] < len(history) - 1:
        state['step'] += 1
        sl_step.set_val(state['step'])
    else:
        state['playing'] = False
        btn_play.label.set_text('Play')

def on_play(event):
    if state['playing']:
        state['playing'] = False
        btn_play.label.set_text('Play')
        if state['anim']:
            state['anim'].event_source.stop()
    else:
        if state['step'] >= len(history) - 1:
            state['step'] = 0
        state['playing'] = True
        btn_play.label.set_text('Pause')
        state['anim'] = animation.FuncAnimation(
            fig, animate, interval=120, cache_frame_data=False
        )
        plt.draw()

def on_end(event):
    state['step'] = len(history) - 1
    sl_step.set_val(state['step'])

btn_reset.on_clicked(on_reset)
btn_prev.on_clicked(on_prev)
btn_step.on_clicked(on_step_btn)
btn_play.on_clicked(on_play)
btn_end.on_clicked(on_end)

plt.suptitle('', fontsize=13, fontweight='bold', y=0.97)
print(f"Sklearn  →  m = {model.coef_[0]:.5f},  b = {model.intercept_:.5f}")
print(f"GD final →  m = {history[-1][0]:.5f},  b = {history[-1][1]:.5f}")
plt.show()


