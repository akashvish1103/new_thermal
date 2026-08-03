# Using EWMA to smooth the loss curve and parameter trajectories.

"""
Thermal Face ROI Extractor with Live Graph
==========================================
Reads a thermal grey video, detects 6 ROIs via MediaPipe Face Mesh,
extracts mean pixel intensity per ROI per frame, applies Exponential
Moving Average (EMA) smoothing, and plots all signals in a live
matplotlib window alongside the video feed.

Dependencies:
    pip install opencv-python mediapipe numpy matplotlib
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import utilities as ut


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# -----------------------------
# Input Video Path
# -----------------------------
# VIDEO_PATH = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# VIDEO_PATH = VIDEO_PATH = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
# VIDEO_PATH = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# VIDEO_PATH = r"C:\Users\Akash Vishwakarma\Pictures\Camera Roll\WIN_20260525_15_35_42_Pro.mp4"

# Exponential Moving Average alpha (0 < α ≤ 1)
# Lower  = smoother but more lag
# Higher = faster response but noisier
EMA_ALPHA = 0.5                                  # weightage of current data , 1-EMA_ALPHA will be the weightage of previous data.

# How many frames to display in the rolling graph window
GRAPH_WINDOW = 300          # ~10 s at 30 fps

# ROI definitions (label, colour for plot line)
ROI_META = [
    ("Left Eye",    "#00e5ff"),
    ("Right Eye",   "#ff4081"),
    ("Forehead",    "#76ff03"),
    ("Nose Tip",    "#ffea00"),
    ("Left Cheek",  "#ff6d00"),
    ("Right Cheek", "#d500f9"),
]

ROI_KEYS = [m[0] for m in ROI_META]
ROI_COLORS = [m[1] for m in ROI_META]


# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
)


# ─────────────────────────────────────────────
#  SIGNAL STORAGE
# ─────────────────────────────────────────────

raw_signals  = {k: deque(maxlen=GRAPH_WINDOW) for k in ROI_KEYS}
ema_signals  = {k: deque(maxlen=GRAPH_WINDOW) for k in ROI_KEYS}
ema_prev     = {k: None for k in ROI_KEYS}   # EMA state


def ema_update(key: str, new_val: float) -> float:
    """Update Exponential Moving Average for one ROI."""
    prev = ema_prev[key]
    smoothed = new_val if prev is None else EMA_ALPHA * new_val + (1 - EMA_ALPHA) * prev
    ema_prev[key] = smoothed
    return smoothed


# ─────────────────────────────────────────────
#  MATPLOTLIB FIGURE (interactive / non-blocking)
# ─────────────────────────────────────────────

plt.ion()
fig = plt.figure(figsize=(14, 8), facecolor="#0d0d0d")
fig.canvas.manager.set_window_title("Thermal ROI — Live Signal Monitor")

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    hspace=0.55,
    wspace=0.35,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.07,
)

axes   = {}
lines_raw = {}
lines_ema = {}

for i, (label, color) in enumerate(ROI_META):
    row, col = divmod(i, 2)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor("#111111")
    ax.set_title(label, color=color, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xlim(0, GRAPH_WINDOW)
    ax.set_ylim(0, 255)
    ax.set_xlabel("Frames", color="#444444", fontsize=7)
    ax.set_ylabel("Intensity", color="#444444", fontsize=7)
    ax.grid(True, color="#1e1e1e", linewidth=0.6)

    # Raw signal — faint
    ln_raw, = ax.plot([], [], color=color, alpha=0.50, linewidth=0.8, label="raw")   # chnaged this  from 0.25 to 0.50 for better visibility
    # EMA signal — bold
    ln_ema, = ax.plot([], [], color=color, alpha=0.95, linewidth=1.6, label="EMA")

    ax.legend(
        fontsize=6,
        loc="upper right",
        facecolor="#111111",
        edgecolor="#333333",
        labelcolor=color,
    )

    axes[label]    = ax
    lines_raw[label] = ln_raw
    lines_ema[label] = ln_ema

fig.suptitle("Thermal Face ROI — Live Intensity Monitor", color="#cccccc",
             fontsize=11, fontweight="bold", y=0.98)


def refresh_plots():
    """Redraw all subplot lines from current deque data."""
    for label in ROI_KEYS:
        n_raw = len(raw_signals[label])
        n_ema = len(ema_signals[label])

        x_raw = list(range(n_raw))
        x_ema = list(range(n_ema))

        lines_raw[label].set_data(x_raw, list(raw_signals[label]))
        lines_ema[label].set_data(x_ema, list(ema_signals[label]))

        ax = axes[label]
        # Auto-scale Y with a small margin
        all_vals = list(raw_signals[label]) + list(ema_signals[label])
        if all_vals:
            lo = max(0,   min(all_vals) - 5)
            hi = min(255, max(all_vals) + 5)
            if hi > lo:
                ax.set_ylim(lo, hi)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ─────────────────────────────────────────────
#  VIDEO CAPTURE
# ─────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

frame_idx   = 0
flag        = False


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

while True:
    points_left  = []
    points_right = []

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _ = frame.shape

    # ── Enhance & detect landmarks ──────────────────────────
    sharpened = ut.get_transformed_image(grey)
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # ── Draw ROIs on frame ──────────────────────────
            tl, br, tr, bl = ut.get_eyes_coordinates(frame, grey, face_landmarks)
            lm, tm, rm, bm = ut.get_forehead_coordinates(frame, face_landmarks, flag)
            nx0, ny0, nx1, ny1 = ut.get_nose_coordinates(frame, face_landmarks)
            pl, pr = ut.get_cheeks_coordinates(frame, face_landmarks, points_left, points_right)

            # ── Extract raw mean intensities from GREY frame ─
            roi_vals = {}

            # Left Eye (inner corner box)
            le_crop = grey[tl[1]:br[1], tl[0]:br[0]]
            roi_vals["Left Eye"] = float(np.mean(le_crop)) if le_crop.size else 0.0

            # Right Eye (inner corner box)
            re_crop = grey[tr[1]:bl[1], bl[0]:tr[0]]
            roi_vals["Right Eye"] = float(np.mean(re_crop)) if re_crop.size else 0.0

            # Forehead (rectangle)
            fh_crop = grey[tm:bm, lm:rm]
            roi_vals["Forehead"] = float(np.mean(fh_crop)) if fh_crop.size else 0.0

            # Nose tip (rectangle)
            nose_crop = grey[ny0:ny1, nx0:nx1]
            roi_vals["Nose Tip"] = float(np.mean(nose_crop)) if nose_crop.size else 0.0

            # Cheeks (polygon mask on grey)
            mask_l = np.zeros((h, w), dtype=np.uint8)
            mask_r = np.zeros((h, w), dtype=np.uint8)

            if len(pl) >= 3:
                cv2.fillPoly(mask_l, [np.array(pl, dtype=np.int32)], 255)
                roi_vals["Left Cheek"] = float(np.mean(grey[mask_l > 0])) if mask_l.any() else 0.0
            else:
                roi_vals["Left Cheek"] = 0.0

            if len(pr) >= 3:
                cv2.fillPoly(mask_r, [np.array(pr, dtype=np.int32)], 255)
                roi_vals["Right Cheek"] = float(np.mean(grey[mask_r > 0])) if mask_r.any() else 0.0
            else:
                roi_vals["Right Cheek"] = 0.0

            # ── Update signals ──────────────────────────────
            for key in ROI_KEYS:
                val     = roi_vals[key]
                smoothed = ema_update(key, val)
                raw_signals[key].append(val)
                ema_signals[key].append(smoothed)

    # ── Overlay frame counter ────────────────────────────────
    cv2.putText(
        frame, f"Frame: {frame_idx}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )

    # ── Refresh live graph every 3 frames (balance speed/load) ─
    if frame_idx % 3 == 0:
        refresh_plots()

    # ── Show video window ────────────────────────────────────
    cv2.imshow("Thermal ROI Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:      # ESC to quit
        break

# ─────────────────────────────────────────────
#  CLEANUP
# ─────────────────────────────────────────────

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()       # keep final graph open after video ends