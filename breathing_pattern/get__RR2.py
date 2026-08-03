# Use this for Final BPM calculation

import cv2
import numpy as np
import mediapipe as mp
import utilities as ut
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARKS
# ─────────────────────────────────────────────────────────────────────────────

UPPER_LIPS_LANDMARK       = [13, 206, 426]
NOSE_HORIZONTAL_LANDMARKS = [64, 278]
NOSE_VERTICAL_LANDMARKS   = [4, 94]
NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK

LEFT_HORIZONTAL_LANDMARK  = 64
RIGHT_HORIZONTAL_LANDMARK = 278
UP_VERTICAL_LANDMARK      = 4
DOWN_VERTICAL_LANDMARK    = 94

# ─────────────────────────────────────────────────────────────────────────────
# INPUT VIDEO — uncomment the one you want
# ─────────────────────────────────────────────────────────────────────────────

# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\all_grey_manual_mp4\pratham_grey_manual.mp4"
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\pooja_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Swamini\swamini_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = r"D:\Lie Detection Data HTI\Girish\girish_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Dhruv\dhruv_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Prem\prem_grey_manual.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\Thermal_Clips\Q35.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\Thermal_Clips\Q44.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\62_2026-07-13\02_Psychometric_Tests\62_HDRS_grayscaled_Thermal.mpg"
# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_temp_from_pixel(pixel_value):
    """Linear mapping: pixel intensity → temperature (°C)."""
    m = 0.05891454
    b = 30.07676744
    return m * pixel_value + b


def compute_bpm_welch(window_signal, fps):
    """
    Compute breathing rate (BPM) from a signal chunk using Welch PSD.
    Returns np.nan if the chunk is too short or no valid frequency found.
    """
    if len(window_signal) < fps * 4:
        return np.nan
    sig = detrend(np.array(window_signal, dtype=float))
    nyq = fps / 2
    b, a = butter(4, [0.1 / nyq, 0.8 / nyq], btype='band')
    filtered = filtfilt(b, a, sig)
    freqs, psd = welch(filtered, fs=fps, nperseg=min(256, len(filtered) // 2))
    mask = (freqs >= 0.1) & (freqs <= 0.8)
    if not mask.any():
        return np.nan
    return freqs[mask][np.argmax(psd[mask])] * 60


def calculate_breathing_rate(signal, fps):
    """
    Global BPM over the full signal using Welch PSD.
    Returns: (bpm, filtered_signal, freqs, psd)
    """
    signal = np.array(signal, dtype=float)
    sig_detrended = detrend(signal)
    nyq = fps / 2
    b, a = butter(4, [0.1 / nyq, 0.8 / nyq], btype='band')
    filtered = filtfilt(b, a, sig_detrended)
    freqs, psd = welch(filtered, fs=fps, nperseg=min(256, len(filtered) // 2))
    mask = (freqs >= 0.1) & (freqs <= 0.8)
    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return peak_freq * 60, filtered, freqs, psd

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COLLECTION — main video loop
# ─────────────────────────────────────────────────────────────────────────────

lst      = []   # raw pixel intensity (breathing signal)
lst_temp = []   # temperature in °C

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps:.2f}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    grey_frame        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformed_frame = ut.get_transformed_image(grey_frame)
    rgb_frame         = cv2.cvtColor(transformed_frame, cv2.COLOR_GRAY2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            for idx in NOSE_LANDMARKS:
                nose_tip = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                nose_x = int(nose_tip.x * w)
                nose_y = int(nose_tip.y * h)

                if idx == LEFT_HORIZONTAL_LANDMARK:
                    left_margin = nose_x
                if idx == RIGHT_HORIZONTAL_LANDMARK:
                    right_margin = nose_x
                if idx == UP_VERTICAL_LANDMARK:
                    top_margin = nose_y
                if idx == DOWN_VERTICAL_LANDMARK:
                    height        = nose_y - top_margin
                    bottom_margin = nose_y + int(1.15 * height)

            dist         = right_margin - left_margin
            left_margin  = left_margin  - int(dist * 0.1)
            right_margin = right_margin + int(dist * 0.1)

            nose_crop  = frame[top_margin:bottom_margin, left_margin:right_margin]
            mean_value = np.mean(nose_crop)
            lst.append(mean_value)
            lst_temp.append(round(get_temp_from_pixel(mean_value), 3))

            cv2.rectangle(transformed_frame, (left_margin, top_margin),
                          (right_margin, bottom_margin), (255, 0, 0), 1)
            cv2.rectangle(frame, (left_margin, top_margin),
                          (right_margin, bottom_margin), (100, 0, 255), 2)

    cv2.imshow('Nose Tip Detection', frame)
    cv2.imshow('Transformed Frame',  transformed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL BPM (full video)
# ─────────────────────────────────────────────────────────────────────────────

global_bpm, filtered_signal, freqs, psd = calculate_breathing_rate(lst, fps)
print(f"Global Breathing Rate (full video): {global_bpm:.1f} breaths/min")

# ─────────────────────────────────────────────────────────────────────────────
# LIVE BPM TRACE — sliding window
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SEC = 20
STEP_SEC   = 1

window_frames = int(WINDOW_SEC * fps)
step_frames   = int(STEP_SEC   * fps)

bpm_values = []
bpm_times  = []

for start in range(0, len(lst) - window_frames + 1, step_frames):
    chunk = lst[start : start + window_frames]
    bpm   = compute_bpm_welch(chunk, fps)
    bpm_values.append(bpm)
    bpm_times.append((start + window_frames / 2) / fps)

bpm_values = np.array(bpm_values)
bpm_times  = np.array(bpm_times)

# EMA smoothing on BPM trace
alpha      = 0.3
bpm_smooth = np.full_like(bpm_values, np.nan)
for i, b in enumerate(bpm_values):
    if np.isnan(b):
        continue
    bpm_smooth[i] = b if (i == 0 or np.isnan(bpm_smooth[i - 1])) \
                      else alpha * b + (1 - alpha) * bpm_smooth[i - 1]

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE PRINT — BPM per time window
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 52)
print(f"  {'Time (s)':>10}  {'BPM (raw)':>10}  {'BPM (smooth)':>12}")
print("=" * 52)

for t, raw, smooth in zip(bpm_times, bpm_values, bpm_smooth):
    raw_str    = f"{raw:.1f}"    if not np.isnan(raw)    else "  ---"
    smooth_str = f"{smooth:.1f}" if not np.isnan(smooth) else "  ---"
    print(f"  {t:>10.1f}  {raw_str:>10}  {smooth_str:>12}")

print("=" * 52)
print(f"  Mean : {np.nanmean(bpm_values):.1f} bpm")
print(f"  Min  : {np.nanmin(bpm_values):.1f} bpm")
print(f"  Max  : {np.nanmax(bpm_values):.1f} bpm")
print("=" * 52)

# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHED SIGNAL + PEAK / BOTTOM DETECTION
# ─────────────────────────────────────────────────────────────────────────────

window   = 10
smoothed = np.convolve(lst, np.ones(window) / window, mode='valid')
slope    = np.diff(smoothed)

peak_x, peak_y, bottom_x, bottom_y = [], [], [], []
for i in range(1, len(slope)):
    if slope[i - 1] > 0 and slope[i] <= 0:
        peak_x.append(i);   peak_y.append(smoothed[i])
    elif slope[i - 1] < 0 and slope[i] >= 0:
        bottom_x.append(i); bottom_y.append(smoothed[i])

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

t_raw      = np.arange(len(lst))             / fps
t_filtered = np.arange(len(filtered_signal)) / fps
t_smoothed = np.arange(len(smoothed))        / fps

fig, axes = plt.subplots(6, 1, figsize=(15, 17))
fig.suptitle(f"Breathing Analysis  |  Global BPM: {global_bpm:.1f}  "
             f"|  Window: {WINDOW_SEC}s  Step: {STEP_SEC}s",
             fontsize=13, fontweight='bold')

# ── Graph 1: Raw signal ───────────────────────────────────────────────────
axes[0].plot(t_raw, lst, linewidth=0.7, color='steelblue')
axes[0].set_title("Graph 1 — Raw ROI Pixel Intensity")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Intensity")

# ── Graph 2: Bandpass filtered ────────────────────────────────────────────
axes[1].plot(t_filtered, filtered_signal, linewidth=1, color='royalblue')
axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
axes[1].set_title("Graph 2 — Bandpass Filtered Signal (0.1–0.8 Hz)  ← clean breathing waveform")
axes[1].set_xlabel("Time (s)")

# ── Graph 3: Welch PSD (global) ───────────────────────────────────────────
psd_mask = (freqs >= 0.05) & (freqs <= 1.2)
axes[2].plot(freqs[psd_mask], psd[psd_mask], color='darkorange')
axes[2].axvline(global_bpm / 60, color='red', linestyle='--', linewidth=1.5,
                label=f'Peak: {global_bpm:.1f} bpm  ({global_bpm/60:.3f} Hz)')
axes[2].set_title("Graph 3 — Welch PSD (full video)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].legend()

# ── Graph 4: Live BPM trace ───────────────────────────────────────────────
axes[3].plot(bpm_times, bpm_values, linewidth=1, color='silver', label='Raw BPM (per window)')
axes[3].plot(bpm_times, bpm_smooth, linewidth=2, color='tomato', label=f'Smoothed BPM (EMA α={alpha})')
axes[3].axhline(12, color='green', linestyle='--', linewidth=0.8, alpha=0.6)
axes[3].axhline(20, color='green', linestyle='--', linewidth=0.8, alpha=0.6, label='Normal range (12–20)')
axes[3].set_ylim(0, 40)
axes[3].set_title("Graph 4 — Live Breathing Rate Over Time")
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Breaths / min")
axes[3].legend()

# ── Graph 5: Live BPM only (standalone clean view) ───────────────────────
axes[4].plot(bpm_times, bpm_smooth, linewidth=2.5, color='tomato', label='BPM (smoothed)')
axes[4].fill_between(bpm_times, bpm_smooth, alpha=0.15, color='tomato')
axes[4].axhline(12, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Normal min (12)')
axes[4].axhline(20, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Normal max (20)')
axes[4].axhline(global_bpm, color='navy', linestyle=':', linewidth=1.2,
                label=f'Global mean ({global_bpm:.1f} bpm)')
axes[4].set_ylim(0, 40)
axes[4].set_title("Graph 5 — Live BPM (standalone view)")
axes[4].set_xlabel("Time (s)")
axes[4].set_ylabel("Breaths / min")
axes[4].legend()

# ── Graph 6: Smoothed signal + peaks / bottoms ───────────────────────────
axes[5].plot(t_smoothed, smoothed, linewidth=1, label='Smoothed signal (MA-10)')
axes[5].scatter(np.array(peak_x)   / fps, peak_y,   color='red',  s=12, zorder=3, label='Peaks')
axes[5].scatter(np.array(bottom_x) / fps, bottom_y, color='blue', s=12, zorder=3, label='Bottoms')
axes[5].set_title("Graph 6 — Moving Average with Peaks & Bottoms")
axes[5].set_xlabel("Time (s)")
axes[5].legend()

plt.tight_layout()
plt.show()