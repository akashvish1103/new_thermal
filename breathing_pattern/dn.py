# detecting breathingpettern using FFT (Fast Fourier Transform) and peak detection  , will imporve this....


import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import utilities as ut

# ----------------------------------------
# LANDMARKS
# ----------------------------------------
UPPER_LIPS_LANDMARK = [13, 206, 426]
NOSE_HORIZONTAL_LANDMARKS = [64, 278]
NOSE_VERTICAL_LANDMARKS = [4, 94]
LEFT_HORIZONTAL_LANDMARK = 64
RIGHT_HORIZONTAL_LANDMARK = 278
UP_VERTICAL_LANDMARK = 4
DOWN_VERTICAL_LANDMARK = 94
NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK

# ----------------------------------------
# PIXEL → TEMPERATURE
# ----------------------------------------
def get_temp_from_pixel(pixel_value):
    m = 0.05891454
    b = 30.07676744
    return m * pixel_value + b

# ----------------------------------------
# BANDPASS FILTER  (keeps only breathing frequencies)
# ----------------------------------------
def bandpass_filter(signal, lowcut=0.1, highcut=0.5, fs=25, order=3):
    """
    lowcut = 0.1 Hz → 6 breaths/min  (lower bound)
    highcut = 0.5 Hz → 30 breaths/min (upper bound)
    Removes heartbeat noise, slow drift, etc.
    """
    nyq = 0.5 * fs                         # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)          # zero-phase filtering (no time shift)

# ----------------------------------------
# FFT-based BREATH RATE
# ----------------------------------------
def get_breath_rate_fft(signal, fps):
    """
    Finds dominant frequency in breathing range using FFT.
    Returns breath rate in breaths/min.
    """
    N = len(signal)
    T = 1.0 / fps
    signal_centered = signal - np.mean(signal)     # remove DC offset (flat trend)
    
    yf = np.abs(fft(signal_centered))              # FFT magnitude
    xf = fftfreq(N, T)                             # frequency axis in Hz

    # Focus only on breathing frequency range
    mask = (xf >= 0.1) & (xf <= 0.5)
    
    if not np.any(mask):
        return None, None, None
    
    dominant_freq = xf[mask][np.argmax(yf[mask])] # strongest frequency
    breath_rate   = dominant_freq * 60             # Hz → breaths/min
    
    return breath_rate, xf, yf

# ----------------------------------------
# PEAK-BASED BREATH RATE  (cross-validation)
# ----------------------------------------
def get_breath_rate_peaks(smoothed_signal, fps):
    """
    Detects peaks and bottoms in smoothed signal.
    Filters out false peaks that are too close together.
    Returns breath rate and peak/bottom positions.
    """
    slope = np.diff(smoothed_signal)

    raw_peak_x, raw_bottom_x = [], []

    for i in range(1, len(slope)):
        if slope[i-1] > 0 and slope[i] <= 0:
            raw_peak_x.append(i)
        elif slope[i-1] < 0 and slope[i] >= 0:
            raw_bottom_x.append(i)

    # ---- Filter false peaks (too close = noise) ----
    MIN_DIST = int(fps * 1.5)           # minimum 1.5 sec between real breaths

    def filter_close_points(points):
        if not points:
            return []
        filtered = [points[0]]
        for p in points[1:]:
            if p - filtered[-1] >= MIN_DIST:
                filtered.append(p)
        return filtered

    peak_x   = filter_close_points(raw_peak_x)
    bottom_x = filter_close_points(raw_bottom_x)

    peak_y   = [smoothed_signal[i] for i in peak_x]
    bottom_y = [smoothed_signal[i] for i in bottom_x]

    # ---- Breath rate from peaks ----
    breath_rate = None
    if len(peak_x) >= 2:
        intervals_sec = np.diff(peak_x) / fps
        breath_rate   = 60 / np.mean(intervals_sec)

    return breath_rate, peak_x, peak_y, bottom_x, bottom_y

# ----------------------------------------
# BREATH PATTERN CLASSIFIER
# ----------------------------------------
def classify_breath_pattern(breath_rate):
    """
    Classifies breathing into clinical categories.
    Normal adult: 12-20 breaths/min
    """
    if breath_rate is None:
        return "Unknown"
    elif breath_rate < 8:
        return "BRADYPNEA (too slow - < 8)"      # abnormally slow
    elif breath_rate < 12:
        return "Slow (8-12)"
    elif breath_rate <= 20:
        return "NORMAL (12-20)"                  # healthy range
    elif breath_rate <= 25:
        return "Slightly Fast (20-25)"
    else:
        return "TACHYPNEA (too fast - > 25)"     # abnormally fast

# ============================================================
# MAIN VIDEO LOOP
# ============================================================
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

lst_pixel = []       # raw pixel intensity
lst_temp  = []       # converted temperature values

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

            h, w, _ = frame.shape
            left_margin = right_margin = top_margin = bottom_margin = 0

            for idx in NOSE_LANDMARKS:
                lm    = face_landmarks.landmark[idx]
                nx    = int(lm.x * w)
                ny    = int(lm.y * h)

                if idx == LEFT_HORIZONTAL_LANDMARK:
                    left_margin = nx
                if idx == RIGHT_HORIZONTAL_LANDMARK:
                    right_margin = nx
                if idx == UP_VERTICAL_LANDMARK:
                    top_margin = ny
                if idx == DOWN_VERTICAL_LANDMARK:
                    ht            = ny - top_margin
                    bottom_margin = ny + int(1.15 * ht)

            # Add 10% buffer to width
            dist         = right_margin - left_margin
            left_margin  = left_margin  - int(dist * 0.1)
            right_margin = right_margin + int(dist * 0.1)

            # Crop ROI and get mean
            nose_crop  = frame[top_margin:bottom_margin, left_margin:right_margin]
            mean_pixel = np.mean(nose_crop)

            lst_pixel.append(mean_pixel)
            lst_temp.append(round(get_temp_from_pixel(mean_pixel), 3))

            cv2.rectangle(frame, (left_margin, top_margin),
                          (right_margin, bottom_margin), (100, 0, 255), 2)

    cv2.imshow('Nose ROI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ============================================================
# SIGNAL PROCESSING AFTER VIDEO ENDS
# ============================================================
signal_raw = np.array(lst_pixel)

# Step 1: Smooth with moving average (same as your original)
WINDOW = 20
smoothed = np.convolve(signal_raw, np.ones(WINDOW)/WINDOW, mode='valid')

# Step 2: Bandpass filter on smoothed signal
filtered = bandpass_filter(smoothed, lowcut=0.1, highcut=0.5, fs=fps)

# Step 3: FFT breath rate
br_fft, xf, yf = get_breath_rate_fft(filtered, fps)

# Step 4: Peak-based breath rate
br_peaks, peak_x, peak_y, bottom_x, bottom_y = get_breath_rate_peaks(smoothed, fps)

# Step 5: Classify
pattern_fft   = classify_breath_pattern(br_fft)
pattern_peaks = classify_breath_pattern(br_peaks)

print("=" * 45)
print(f"  FFT  Breath Rate  : {br_fft:.1f}  breaths/min")
print(f"  Peak Breath Rate  : {br_peaks:.1f} breaths/min")
print(f"  FFT  Pattern      : {pattern_fft}")
print(f"  Peak Pattern      : {pattern_peaks}")
print("=" * 45)

# ============================================================
# PLOTTING
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("Breathing Pattern Analysis", fontsize=14, fontweight='bold')

# --- Plot 1: Raw pixel signal ---
axes[0].plot(signal_raw, color='steelblue')
axes[0].set_title("Raw Nose Pixel Intensity")
axes[0].set_ylabel("Pixel Value")

# --- Plot 2: Smoothed + Bandpass filtered ---
axes[1].plot(smoothed, color='gray',   alpha=0.5, label='Smoothed (Moving Avg)')
axes[1].plot(filtered, color='orange', linewidth=1.5, label='Bandpass Filtered')
axes[1].set_title("Smoothed vs Bandpass Filtered Signal")
axes[1].set_ylabel("Intensity")
axes[1].legend()

# --- Plot 3: Peak detection on smoothed ---
axes[2].plot(smoothed, color='steelblue', label='Smoothed Signal')
axes[2].scatter(peak_x,   peak_y,   color='red',  s=40, zorder=5, label=f'Peaks (Exhale)')
axes[2].scatter(bottom_x, bottom_y, color='blue', s=40, zorder=5, label=f'Bottoms (Inhale)')
axes[2].set_title(f"Peak Detection  |  Breath Rate: {br_peaks:.1f} breaths/min  |  Pattern: {pattern_peaks}")
axes[2].set_ylabel("Intensity")
axes[2].legend()

# --- Plot 4: FFT frequency spectrum ---
if xf is not None:
    mask = (xf >= 0.05) & (xf <= 0.8)     # show slightly wider range for context
    axes[3].plot(xf[mask] * 60, np.abs(yf[mask]), color='green')   # x-axis in breaths/min
    axes[3].axvline(x=br_fft, color='red', linestyle='--',
                    label=f'Dominant: {br_fft:.1f} br/min')
    axes[3].axvspan(12, 20, alpha=0.1, color='green', label='Normal range (12-20)')
    axes[3].set_title(f"FFT Frequency Spectrum  |  Breath Rate: {br_fft:.1f} breaths/min  |  Pattern: {pattern_fft}")
    axes[3].set_xlabel("Breaths per Minute")
    axes[3].set_ylabel("Magnitude")
    axes[3].legend()

plt.tight_layout()
plt.show()