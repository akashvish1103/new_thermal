import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, welch


# ============================================================
# 1. CREATE A SYNTHETIC BREATHING SIGNAL
# ============================================================

fps = 10                  # samples per second
duration = 120            # seconds
t = np.arange(0, duration, 1 / fps)

# True breathing frequency
# 0.25 Hz = 0.25 breaths/sec = 15 breaths/min
true_breath_hz = 0.25

rng = np.random.default_rng(42)

# Breathing component
breathing = 2.0 * np.sin(
    2 * np.pi * true_breath_hz * t
)

# Slow baseline drift
slow_drift = 0.015 * t

# Random noise
noise = 0.45 * rng.normal(size=len(t))

# Faster interference/noise
interference = 0.25 * np.sin(
    2 * np.pi * 1.2 * t
)

# Final raw signal
raw_signal = (
    100
    + breathing
    + slow_drift
    + interference
    + noise
)


# ============================================================
# 2. DETRENDING
# ============================================================

detrended = detrend(raw_signal)


# ============================================================
# 3. BAND-PASS FILTER
#    Keep only 0.1–0.8 Hz
# ============================================================

nyq = fps / 2

b, a = butter(
    4,
    [0.1 / nyq, 0.8 / nyq],
    btype="band"
)

filtered = filtfilt(
    b,
    a,
    detrended
)


# ============================================================
# 4. WELCH POWER SPECTRAL DENSITY
# ============================================================

freqs, psd = welch(
    filtered,
    fs=fps,
    nperseg=min(256, len(filtered) // 2)
)

# Only look at breathing frequency range
mask = (
    (freqs >= 0.1) &
    (freqs <= 0.8)
)

# Find strongest frequency
peak_freq = freqs[mask][
    np.argmax(psd[mask])
]

# Convert Hz → BPM
bpm = peak_freq * 60


# ============================================================
# 5. MOVING AVERAGE
#    Used for later peak/bottom analysis
# ============================================================

window = 10

smoothed = np.convolve(
    filtered,
    np.ones(window) / window,
    mode="valid"
)

t_smoothed = t[:len(smoothed)]


# ============================================================
# 6. PEAK / BOTTOM DETECTION
# ============================================================

slope = np.diff(smoothed)

peak_x = []
peak_y = []

bottom_x = []
bottom_y = []

for i in range(1, len(slope)):

    # Peak
    if slope[i - 1] > 0 and slope[i] <= 0:

        peak_x.append(i)
        peak_y.append(smoothed[i])

    # Bottom
    elif slope[i - 1] < 0 and slope[i] >= 0:

        bottom_x.append(i)
        bottom_y.append(smoothed[i])


# ============================================================
# GRAPH 1 — RAW SIGNAL
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(15, 14))

# Flatten axes so we can use axes[0], axes[1], ...
axes = axes.flatten()


# ============================================================
# GRAPH 1 — RAW SIGNAL
# ============================================================

axes[0].plot(t, raw_signal)

axes[0].set_title("1. Raw Signal")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Signal")
axes[0].set_xlim(0, 40)
axes[0].grid(alpha=0.3)


# ============================================================
# GRAPH 2 — DETRENDED SIGNAL
# ============================================================

axes[1].plot(t, detrended)

axes[1].set_title("2. After Detrending")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Signal")
axes[1].set_xlim(0, 40)
axes[1].grid(alpha=0.3)


# ============================================================
# GRAPH 3 — BAND-PASS FILTERED SIGNAL
# ============================================================

axes[2].plot(t, filtered)

axes[2].set_title("3. Band-pass Filtered (0.1–0.8 Hz)")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Signal")
axes[2].set_xlim(0, 40)
axes[2].grid(alpha=0.3)


# ============================================================
# GRAPH 4 — WELCH PSD
# ============================================================

axes[3].plot(freqs[mask], psd[mask])

axes[3].axvline(
    peak_freq,
    linestyle="--",
    label=f"Peak = {peak_freq:.3f} Hz = {bpm:.1f} BPM"
)

axes[3].set_title("4. Welch Power Spectral Density (PSD)")
axes[3].set_xlabel("Frequency (Hz)")
axes[3].set_ylabel("Power")
axes[3].legend()
axes[3].grid(alpha=0.3)


# ============================================================
# GRAPH 5 — MOVING AVERAGE
# ============================================================

axes[4].plot(
    t,
    filtered,
    alpha=0.35,
    label="Filtered signal"
)

axes[4].plot(
    t_smoothed,
    smoothed,
    linewidth=2,
    label="Moving average (10 samples)"
)

axes[4].set_title("5. Moving Average Smoothing")
axes[4].set_xlabel("Time (seconds)")
axes[4].set_ylabel("Signal")
axes[4].set_xlim(0, 40)
axes[4].legend()
axes[4].grid(alpha=0.3)


# ============================================================
# GRAPH 6 — PEAK / BOTTOM DETECTION
# ============================================================

axes[5].plot(
    t_smoothed,
    smoothed,
    label="Smoothed signal"
)

axes[5].scatter(
    np.array(peak_x) / fps,
    peak_y,
    label="Peaks"
)

axes[5].scatter(
    np.array(bottom_x) / fps,
    bottom_y,
    label="Bottoms"
)

axes[5].set_title("6. Peak / Bottom Detection")
axes[5].set_xlabel("Time (seconds)")
axes[5].set_ylabel("Signal")
axes[5].set_xlim(0, 40)
axes[5].legend()
axes[5].grid(alpha=0.3)


# ============================================================
# FINAL LAYOUT
# ============================================================

fig.suptitle(
    f"Breathing Signal Processing Pipeline | Estimated BPM = {bpm:.1f}",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.show()

# ============================================================
# FINAL RESULT
# ============================================================

print(
    f"True breathing rate:      "
    f"{true_breath_hz * 60:.1f} BPM"
)

print(
    f"Estimated breathing rate: "
    f"{bpm:.1f} BPM"
)