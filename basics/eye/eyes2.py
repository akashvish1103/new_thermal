import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore

# ============================================================
# CONFIGURATION
# ============================================================
PERCENTAGE_PIXEL_TO_KEEP = 0.10     # Top 10% hottest pixels
BASELINE_FRAMES          = 100      # Frames used for baseline normalization
SAVGOL_WINDOW            = 15       # Savitzky-Golay smoothing window (must be odd)
SAVGOL_POLYORDER         = 3        # Polynomial order for SG filter
ZSCORE_THRESHOLD         = 2.5      # Z-score threshold for outlier rejection
SUDDEN_JUMP_THRESHOLD    = 5.0      # Max allowed intensity jump between frames

LEFT_INNER_EYE  = 133
RIGHT_INNER_EYE = 362

# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ============================================================
# STEP 1 — EXTRACT RAW SIGNAL FROM VIDEO
# ============================================================
left_mean_values  = []
right_mean_values = []
frame_indices     = []   # track which frames had valid detections
frame_count       = 0

cap    = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _ = frame.shape
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # --- Landmark coordinates ---
            left_point  = face_landmarks.landmark[LEFT_INNER_EYE]
            lx = int(left_point.x * w)
            ly = int(left_point.y * h)

            right_point = face_landmarks.landmark[RIGHT_INNER_EYE]
            rx = int(right_point.x * w)
            ry = int(right_point.y * h)

            # --- Guard: skip if ROI goes out of frame bounds ---
            if (ly-10 < 0 or ly+10 > h or lx < 0 or lx+20 > w or
                ry-10 < 0 or ry+10 > h or rx-20 < 0 or rx > w):
                continue

            # --- ROI extraction ---
            box_left  = grey[ly-10:ly+10, lx:lx+20]
            box_right = grey[ry-10:ry+10, rx-20:rx]

            left_flat  = box_left.flatten()
            right_flat = box_right.flatten()

            # --- Top 10% hottest pixels ---
            left_sorted  = np.sort(left_flat)[::-1]
            right_sorted = np.sort(right_flat)[::-1]

            n_left  = max(1, int(len(left_sorted)  * PERCENTAGE_PIXEL_TO_KEEP))
            n_right = max(1, int(len(right_sorted) * PERCENTAGE_PIXEL_TO_KEEP))

            left_mean  = left_sorted[:n_left].mean()
            right_mean = right_sorted[:n_right].mean()

            left_mean_values.append(left_mean)
            right_mean_values.append(right_mean)
            frame_indices.append(frame_count)

            # --- Visualization ---
            cv2.circle(frame, (lx, ly), 2, (255, 255, 255), -1)
            cv2.circle(frame, (rx, ry), 2, (255, 255, 255), -1)
            cv2.rectangle(frame, (lx, ly-10), (lx+20, ly+10), (255, 0, 0), 2)
            cv2.rectangle(frame, (rx, ry-10), (rx-20, ry+10), (255, 0, 0), 2)

            # Live intensity overlay
            cv2.putText(frame, f"L:{left_mean:.1f}", (lx, ly-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            cv2.putText(frame, f"R:{right_mean:.1f}", (rx-40, ry-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    cv2.imshow("Thermal ROI Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Total frames processed: {frame_count}")
print(f"[INFO] Frames with valid detections: {len(left_mean_values)}")

# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================
left_raw  = np.array(left_mean_values,  dtype=np.float64)
right_raw = np.array(right_mean_values, dtype=np.float64)
frames    = np.array(frame_indices)

# ============================================================
# PHASE 1 — STEP A: OUTLIER REJECTION
# Detect and replace garbage frames (bad tracking, head out of frame)
# ============================================================
def reject_outliers(signal, zscore_thresh=ZSCORE_THRESHOLD, jump_thresh=SUDDEN_JUMP_THRESHOLD):
    """
    Marks frames as bad using two methods:
    1. Z-score: value is too far from mean
    2. Sudden jump: frame-to-frame change is too large
    Returns signal with outliers replaced by NaN.
    """
    clean = signal.copy()

    # Method 1: Z-score
    z = zscore(clean)
    clean[np.abs(z) > zscore_thresh] = np.nan

    # Method 2: Sudden frame-to-frame jump
    diff = np.abs(np.diff(clean, prepend=clean[0]))
    clean[diff > jump_thresh] = np.nan

    n_bad = np.sum(np.isnan(clean))
    print(f"[OUTLIER REJECTION] Removed {n_bad} bad frames ({100*n_bad/len(clean):.1f}%)")

    # Interpolate over NaN gaps
    s = pd.Series(clean)
    s = s.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return s.to_numpy()

left_clean  = reject_outliers(left_raw)
right_clean = reject_outliers(right_raw)

# ============================================================
# PHASE 1 — STEP B: SMOOTHING (Savitzky-Golay filter)
# Removes high-frequency noise while preserving real peaks/trends
# ============================================================
def smooth_signal(signal, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER):
    # window must be odd and > polyorder
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2)
    return savgol_filter(signal, window_length=window, polyorder=polyorder)

left_smooth  = smooth_signal(left_clean)
right_smooth = smooth_signal(right_clean)

print("[INFO] Smoothing applied (Savitzky-Golay)")

# ============================================================
# PHASE 1 — STEP C: BASELINE NORMALIZATION
# Express signal as deviation from subject's resting baseline
# ============================================================
n_baseline = min(BASELINE_FRAMES, len(left_smooth))

left_baseline  = np.mean(left_smooth[:n_baseline])
right_baseline = np.mean(right_smooth[:n_baseline])

left_norm  = left_smooth  - left_baseline
right_norm = right_smooth - right_baseline

print(f"[INFO] Baseline (Left):  {left_baseline:.2f}")
print(f"[INFO] Baseline (Right): {right_baseline:.2f}")

# Asymmetry signal (bonus — useful for stress/deception detection)
asymmetry = left_norm - right_norm

# ============================================================
# PLOTTING — 4 panel comparison
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)
fig.suptitle("Peri-Orbital Thermal Signal — Phase 1 Processing Pipeline", 
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: Raw signal
axes[0].plot(frames, left_raw,  label="Left Eye (raw)",  alpha=0.7, color='steelblue')
axes[0].plot(frames, right_raw, label="Right Eye (raw)", alpha=0.7, color='darkorange')
axes[0].set_title("① Raw Signal")
axes[0].set_ylabel("Mean Intensity")
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Panel 2: After outlier rejection
axes[1].plot(frames, left_clean,  label="Left Eye (outliers removed)",  color='steelblue')
axes[1].plot(frames, right_clean, label="Right Eye (outliers removed)", color='darkorange')
axes[1].set_title("② After Outlier Rejection")
axes[1].set_ylabel("Mean Intensity")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Panel 3: After smoothing
axes[2].plot(frames, left_smooth,  label="Left Eye (smoothed)",  color='steelblue', linewidth=1.5)
axes[2].plot(frames, right_smooth, label="Right Eye (smoothed)", color='darkorange', linewidth=1.5)
axes[2].set_title("③ After Savitzky-Golay Smoothing")
axes[2].set_ylabel("Mean Intensity")
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# Panel 4: Normalized + asymmetry
axes[3].plot(frames, left_norm,  label="Left (normalized)",  color='steelblue',  linewidth=1.5)
axes[3].plot(frames, right_norm, label="Right (normalized)", color='darkorange', linewidth=1.5)
axes[3].plot(frames, asymmetry,  label="Asymmetry (L−R)",   color='crimson',    linewidth=1.2, linestyle='--')
axes[3].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
axes[3].set_title("④ Normalized (deviation from baseline) + Thermal Asymmetry")
axes[3].set_ylabel("ΔIntensity")
axes[3].set_xlabel("Frame Number")
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("thermal_phase1_output.png", dpi=150, bbox_inches='tight')
plt.show()
