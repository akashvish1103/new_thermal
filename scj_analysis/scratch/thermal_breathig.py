import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utilities as ut
import utility_linear_mapping as ulm

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40.wmv"
video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\4.mp4"

WAVEFORM_OUT_PATH = video_path.rsplit(".", 1)[0] + "_breathing_waveform.png"

# =========================================================
# TUNING PARAMETERS (same meaning as your original script)
# =========================================================
SMOOTH_K_NOSE        = 7      # moving average window for nose signal (higher = smoother, more lag)
MIN_PEAK_VALLEY_DIFF = 0.08   # min temp drop (°C) after a peak to count as a real breath
REFRACTORY_PERIOD    = 2.0    # min seconds between two counted breaths (2.0s = 30 BPM max)

# ============================================================
# MediaPipe Face Mesh Setup
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



# =========================================================
# EMA landmark smoother -- stabilizes the breathing box so its
# location doesn't jitter frame to frame (reused from earlier)
# =========================================================
class LandmarkEMASmoother:
    def __init__(self, alpha=0.4, num_landmarks=478):
        self.alpha = alpha
        self.num_landmarks = num_landmarks
        self.prev = None

    def smooth(self, face_landmarks):
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float64
        )
        if self.prev is None or self.prev.shape != raw.shape:
            self.prev = raw.copy()
        else:
            self.prev = self.alpha * raw + (1 - self.alpha) * self.prev

        for i, lm in enumerate(face_landmarks.landmark):
            lm.x = float(self.prev[i, 0])
            lm.y = float(self.prev[i, 1])
            lm.z = float(self.prev[i, 2])
        return face_landmarks

    def reset(self):
        self.prev = None


landmark_smoother = LandmarkEMASmoother(alpha=0.4)


def moving_average(signal, k=7):
    """Centered moving average, window size k. Edges padded by repeating
    the first/last smoothed value. Larger k = smoother but more lag."""
    if len(signal) < k:
        return signal
    smooth    = np.convolve(signal, np.ones(k) / k, mode='valid')
    pad_left  = [smooth[0]]  * (k // 2)
    pad_right = [smooth[-1]] * (k // 2)
    return list(np.concatenate([pad_left, smooth, pad_right]))


def detect_breath_peaks(x, y_smooth):
    """Peak+valley cycle method. A breath counts only when:
       1. a local peak exists (rise then fall)
       2. a valley follows with drop >= MIN_PEAK_VALLEY_DIFF
       3. at least REFRACTORY_PERIOD seconds since the last counted breath
    Returns list of (time, value) tuples for confirmed breath peaks."""
    if len(y_smooth) < 5:
        return []

    y  = np.array(y_smooth)
    dy = np.diff(y)

    raw_peaks, raw_valleys = [], []
    for i in range(1, len(dy)):
        if dy[i-1] > 0 and dy[i] <= 0:
            raw_peaks.append(i)
        elif dy[i-1] < 0 and dy[i] >= 0:
            raw_valleys.append(i)

    if not raw_peaks or not raw_valleys:
        return []

    breath_markers  = []
    last_cycle_time = -999
    used_valleys    = set()

    for pi in raw_peaks:
        next_valleys = [vi for vi in raw_valleys if vi > pi and vi not in used_valleys]
        if not next_valleys:
            continue
        vi = next_valleys[0]

        peak_val, valley_val = y[pi], y[vi]
        if peak_val - valley_val < MIN_PEAK_VALLEY_DIFF:
            continue

        t_peak = x[pi]
        if t_peak - last_cycle_time < REFRACTORY_PERIOD:
            continue

        breath_markers.append((t_peak, peak_val))
        used_valleys.add(vi)
        last_cycle_time = t_peak

    return breath_markers


def extract_box_pixels(image, top_left, bottom_right):
    h, w = image.shape[:2]
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    if x2 <= x1 or y2 <= y1:
        return np.array([], dtype=image.dtype)
    return image[y1:y2, x1:x2]


# =========================================================
# Calibration (pixel <-> temperature), computed ONCE per video
# =========================================================
pixel_column = ulm.get_one_pixel_column(video_path)
min_temp, max_temp, first_pixel, last_pixel = \
    ulm.get_first_last_element_from_OnePixelColumn_AND_min_max_from_filename(
        pixel_column, video_path
    )

# =========================================================
# PASS 1 -- walk the whole video, collect the nose/breathing
# temperature signal using MediaPipe auto-detection
# =========================================================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0
    print(f"WARNING: could not read FPS from video, defaulting to {fps}")

x_data = []        # time in seconds (only appended when a face was found)
y_nose_raw = []     # raw (unsmoothed) breathing-ROI mean temperature

frame_number = -1
got_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    t = frame_number / fps

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformed_grey = ut.get_transformed_image(grey)
    rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2BGR)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_landmarks = landmark_smoother.smooth(face_landmarks)

            # coordinates from the enhanced frame (better detection),
            # pixel values from the raw `grey` frame (valid for calibration)
            top_left, bottom_right, got_frame = ut.get_breathing_roi_cords(transformed_grey, face_landmarks)
            roi_pixels = extract_box_pixels(grey, top_left, bottom_right)

            if roi_pixels.size > 0:
                mean_px = float(np.mean(roi_pixels))
                temp_nose = ulm.map_pixel_to_temperature(mean_px, min_temp, max_temp, first_pixel, last_pixel)

                x_data.append(t)
                y_nose_raw.append(temp_nose)

                cv2.putText(got_frame, f"Nose: {temp_nose:.2f} C", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        landmark_smoother.reset()

    if got_frame is not None:
        cv2.imshow("Breathing ROI (auto-detected)", got_frame)
    cv2.imshow("RGB frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Collected {len(x_data)} valid nose-temperature samples "
      f"out of {frame_number + 1} total frames.")

# =========================================================
# PASS 2 -- smooth, detect breaths, compute BPM
# =========================================================
y_nose_smooth = moving_average(y_nose_raw, SMOOTH_K_NOSE)
peaks = detect_breath_peaks(x_data, y_nose_smooth)

if len(peaks) >= 2:
    total_time = x_data[-1] - x_data[0]
    bpm = round((len(peaks) / total_time) * 60) if total_time > 0 else 0
else:
    bpm = 0

print(f"Detected {len(peaks)} breaths -> estimated BPM: {bpm}")

# =========================================================
# PLOT -- waveform + detected breath peaks, shown/saved at the end
# =========================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_data, y_nose_raw, color="lightgray", linewidth=1, label="raw")
ax.plot(x_data, y_nose_smooth, color="green", linewidth=2, label="smoothed")

if peaks:
    px, py = zip(*peaks)
    ax.scatter(px, py, color="red", s=60, zorder=5, label="confirmed breath")

ax.set_title(f"Breathing Waveform (Nose ROI, auto-detected)  |  Estimated BPM: {bpm}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(WAVEFORM_OUT_PATH, dpi=150)
print(f"Waveform saved to: {WAVEFORM_OUT_PATH}")
plt.show()