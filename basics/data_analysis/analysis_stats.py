import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
import os
import seaborn as sns

# ============================================================
# CONFIG  — CHANGE THESE
# ============================================================

CSV_PATH     = r"C:\Users\Akash Vishwakarma\Desktop\new_thermal\data\sneha_roi_temperatures.csv"# CHANGE
SUBJECT_NAME = "sneha"                                                                   # CHANGE
OUTPUT_DIR   = r"D:\Lie Detection Data HTI\ROI_CSV_Output\EDA_Plots"                      # CHANGE
FPS          = 8                                                                          # CHANGE: your camera fps
SMOOTH_SEC   = 7                                                                           # rolling average window in seconds

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

ROIS = [
    "left_eye_temp",
    "right_eye_temp",
    "forehead_temp",
    "nose_temp",
    "left_cheek_temp",
    "right_cheek_temp",
]

ROI_COLORS = {
    "left_eye_temp":    "#1f77b4",   # blue
    "right_eye_temp":   "#ff7f0e",   # orange
    "forehead_temp":    "#2ca02c",   # green
    "nose_temp":        "#d62728",   # red
    "left_cheek_temp":  "#9467bd",   # purple
    "right_cheek_temp": "#8c564b",   # brown
}

SMOOTH_WIN = FPS * SMOOTH_SEC

# ============================================================
# SMOOTHED COLUMNS
# ============================================================

for roi in ROIS:
    df[roi + "_smooth"] = df[roi].rolling(
        window=SMOOTH_WIN,
        center=True,
        min_periods=1
    ).mean()

# ============================================================
# HELPER: time axis
# ============================================================

x_frames = df["frame"].values if "frame" in df.columns else np.arange(len(df))
x_sec    = x_frames / FPS       # seconds axis

# ============================================================
# PLOT 1 — Individual ROI subplots (raw + smooth)
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(f"Facial ROI Temperature Signals — {SUBJECT_NAME}", fontsize=15, fontweight="bold")

for ax, roi in zip(axes.flat, ROIS):
    col   = ROI_COLORS[roi]
    label = roi.replace("_temp", "").replace("_", " ").title()

    ax.plot(x_sec, df[roi], color=col, alpha=0.25, linewidth=0.6, label="Raw")
    ax.plot(x_sec, df[roi + "_smooth"], color=col, linewidth=1.8, label=f"{SMOOTH_SEC}s smooth")

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate mean
    mean_val = df[roi].mean()
    ax.axhline(mean_val, color=col, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(x_sec[-1] * 0.02, mean_val + 0.02, f"μ={mean_val:.2f}°C",
            color=col, fontsize=7.5)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_ROI_timeseries.png")
plt.savefig(out1, dpi=150)
plt.show()
print(f"Saved → {out1}")


# ============================================================
# PLOT 2 — All ROIs overlaid on one plot
# ============================================================

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle(f"All ROI Temperatures Overlaid — {SUBJECT_NAME}", fontsize=13, fontweight="bold")

for roi in ROIS:
    label = roi.replace("_temp", "").replace("_", " ").title()
    ax.plot(x_sec, df[roi + "_smooth"], color=ROI_COLORS[roi], linewidth=1.5, label=label)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_ROI_overlaid.png")
plt.savefig(out2, dpi=150)
plt.show()
print(f"Saved → {out2}")


# ============================================================
# PLOT 3 — Distribution boxplots per ROI
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(f"ROI Temperature Distribution — {SUBJECT_NAME}", fontsize=13, fontweight="bold")

data_for_box = [df[roi].dropna().values for roi in ROIS]
labels_box   = [roi.replace("_temp","").replace("_"," ").title() for roi in ROIS]
colors_box   = [ROI_COLORS[roi] for roi in ROIS]

bp = ax.boxplot(data_for_box, patch_artist=True, labels=labels_box, widths=0.5)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel("Temperature (°C)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_ROI_boxplot.png")
plt.savefig(out3, dpi=150)
plt.show()
print(f"Saved → {out3}")


# ============================================================
# PLOT 4 — Cheek Asymmetry over time
# ============================================================

df["cheek_asymmetry"] = df["left_cheek_temp"] - df["right_cheek_temp"]
df["eye_asymmetry"]   = df["left_eye_temp"]   - df["right_eye_temp"]

df["cheek_asym_smooth"] = df["cheek_asymmetry"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
df["eye_asym_smooth"]   = df["eye_asymmetry"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
fig.suptitle(f"Left–Right Asymmetry Over Time — {SUBJECT_NAME}", fontsize=13, fontweight="bold")

for ax, col_smooth, col_raw, title, color in [
    (axes[0], "cheek_asym_smooth", "cheek_asymmetry", "Cheek Asymmetry (L − R)", "#2ca02c"),
    (axes[1], "eye_asym_smooth",   "eye_asymmetry",   "Eye Asymmetry (L − R)",   "#1f77b4"),
]:
    ax.plot(x_sec, df[col_raw],    color=color, alpha=0.2, linewidth=0.6)
    ax.plot(x_sec, df[col_smooth], color=color, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Δ Temperature (°C)")
    ax.grid(True, alpha=0.3)

axes[1].set_xlabel("Time (s)")
plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_asymmetry.png")
plt.savefig(out4, dpi=150)
plt.show()
print(f"Saved → {out4}")


# ============================================================
# PLOT 5 — Correlation heatmap between ROIs
# ============================================================

import seaborn as sns

corr = df[ROIS].corr()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f"ROI Temperature Correlation — {SUBJECT_NAME}", fontsize=13, fontweight="bold")

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=ax,
    xticklabels=[r.replace("_temp","").replace("_"," ").title() for r in ROIS],
    yticklabels=[r.replace("_temp","").replace("_"," ").title() for r in ROIS],
)
plt.tight_layout()
out5 = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_correlation.png")
plt.savefig(out5, dpi=150)
plt.show()
print(f"Saved → {out5}")


# ============================================================
# SUMMARY STATS — printed to console + saved to CSV
# ============================================================

stats = df[ROIS].describe().T
stats["range"] = stats["max"] - stats["min"]
stats.index    = [r.replace("_temp","").replace("_"," ").title() for r in ROIS]

print("\n" + "="*60)
print(f"  SUMMARY STATISTICS — {SUBJECT_NAME}")
print("="*60)
print(stats[["mean","std","min","max","range"]].round(3).to_string())

stats_path = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_summary_stats.csv")
stats.to_csv(stats_path)
print(f"\nStats saved → {stats_path}")