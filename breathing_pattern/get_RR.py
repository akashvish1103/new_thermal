# # making NOSE ROI for breathing pattern detection using Mediapipe and elimanating the CSRT tracker failures.
# # Contaiend modules for UTILITIES file (to get TRANSFORMED frame)

# import cv2
# import numpy as np
# import mediapipe as mp
# import utilities as ut
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, welch, detrend

# UPPER_LIPS_LANDMARK = [13, 206, 426]
# NOSE_HORIZONTAL_LANDMARKS = [64, 278]
# NOSE_VERTICAL_LANDMARKS = [4, 94]

# LEFT_HORIZONTAL_LANDMARK = 64
# RIGHT_HORIZONTAL_LANDMARK = 278

# UP_VERTICAL_LANDMARK = 4
# DOWN_VERTICAL_LANDMARK = 94

# lst = []
# lst_temp = []


# # NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS
# NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK

# # Mapping from Pixel Intensity to Temperature (Celcieus)
# def get_temp_from_pixel(pixel_value):
#     m = 0.05891454 
#     b = 30.07676744
#     return m * pixel_value + b

# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# # video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub4_priyank\output_priyank_grey_manual.mp4"

# # -----------------------------
# # Input Video Path
# # -----------------------------
# # video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\all_grey_manual_mp4\aditi_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# # video_path = video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# # video_path = r"D:\Lie Detection Data HTI\Swamini\swamini_grey_manual.mpg"
# # video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"

# # video_path = r"C:\Users\Akash Vishwakarma\Pictures\Camera Roll\WIN_20260525_15_35_42_Pro.mpq4"
# cap = cv2.VideoCapture(video_path)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     transforemd_frame = ut.get_transformed_image(grey_frame)

#     # Convert the BGR image to RGB
#     rgb_frame = cv2.cvtColor(transforemd_frame, cv2.COLOR_GRAY2RGB)
#     # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame with MediaPipe Face Mesh
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for idx in NOSE_LANDMARKS:
#                 # Get the coordinates of the nose tip (landmark index 1)
#                 nose_tip = face_landmarks.landmark[idx]
#                 h, w, _ = frame.shape
#                 nose_x = int(nose_tip.x * w)
#                 nose_y = int(nose_tip.y * h)


#                 # For expanding the rectangular ROI in width
                

#                 if idx == LEFT_HORIZONTAL_LANDMARK:
#                     left_margin = nose_x                  # adding margin to the left and right of the nose tip
                
#                 if idx == RIGHT_HORIZONTAL_LANDMARK:
#                     right_margin = nose_x 

#                 if idx == UP_VERTICAL_LANDMARK:                # adding margin to the top and bottom of the nose tip
#                     top_margin = nose_y - 0
                                    
#                 if idx == DOWN_VERTICAL_LANDMARK:
#                     height = nose_y - top_margin
#                     bottom_margin = nose_y + int(1.15*height)        # shifting bottom edge of the rectangle ROI to downside to get the clear ROI
#                                                                      # shifting by the 15% of the height of the rectangular ROI.

#                 # nose_tip_crop = frame[top_margin:bottom_margin, left_margin:right_margin]

#                 # Draw a red dot on the nose tip
#                 # cv2.circle(transforemd_frame, (nose_x, nose_y), 3, (255, 255, 255), -1)
#                 # cv2.circle(transforemd_frame, (nose_x, nose_y), 3, (0, 255, 120), -1)

#             dist = right_margin  - left_margin                      # width of the rectangular ROI
#             left_margin = left_margin - int(dist*0.1)               # addding the buffer by 10% of the width
#             right_margin = right_margin + int(dist*0.1)             # adding the buffer by 10% of the width

#             nose_tip_crop = frame[top_margin:bottom_margin, left_margin:right_margin]
#             mean_value = np.mean(nose_tip_crop)
#             lst.append(np.mean(nose_tip_crop))                         # Raw Signal (Pixel Intensity)
#             lst_temp.append(round(get_temp_from_pixel(mean_value),3))  # Temperature Signal (Celcius) after mapping the pixel intensity to temperature using the linear equation.

#             cv2.rectangle(transforemd_frame, (left_margin, top_margin), (right_margin, bottom_margin), (255, 0, 0), 1)
#             cv2.rectangle(frame, (left_margin, top_margin), (right_margin, bottom_margin), (100, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('Nose Tip Detection', frame)
#     cv2.imshow('Transformed_frame', transforemd_frame)
    

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break    



# # ── PASTE THIS after your cap.release() / after lst is fully populated ────

# def calculate_breathing_rate(signal, fps):
#     """
#     Calculates breathing rate from raw nose ROI pixel intensity signal.
#     Returns: (bpm, filtered_signal, freqs, psd)
#     """
#     signal = np.array(signal, dtype=float)
    
#     # Step 1: Remove baseline drift (the slow wandering you see in Graph 1)
#     signal_detrended = detrend(signal)
    
#     # Step 2: Bandpass filter — keep only breathing frequencies (0.1–0.8 Hz)
#     nyq = fps / 2
#     b, a = butter(4, [0.1 / nyq, 0.8 / nyq], btype='band')
#     filtered = filtfilt(b, a, signal_detrended)   # zero-phase, no time shift
    
#     # Step 3: Welch PSD — robust against noise by averaging overlapping windows
#     freqs, psd = welch(filtered, fs=fps, nperseg=min(256, len(filtered) // 2))
    
#     # Step 4: Find dominant frequency in breathing band
#     mask = (freqs >= 0.1) & (freqs <= 0.8)
#     peak_freq = freqs[mask][np.argmax(psd[mask])]
#     bpm = peak_freq * 60
    
#     return bpm, filtered, freqs, psd

# # ── Get actual FPS from your video (don't assume 30!) ─────────────────────
# cap_temp = cv2.VideoCapture(video_path)
# fps = cap_temp.get(cv2.CAP_PROP_FPS)
# cap_temp.release()
# print(f"Video FPS: {fps}")

# # ── Calculate ─────────────────────────────────────────────────────────────
# bpm, filtered_signal, freqs, psd = calculate_breathing_rate(lst, fps)
# print(f"Estimated Breathing Rate: {bpm:.1f} breaths/min")

# # ── Your existing smoothed signal (keep as-is) ────────────────────────────
# window = 10
# smoothed = np.convolve(lst, np.ones(window) / window, mode='valid')

# # ── Updated plot layout (replaces your plt.subplot block) ─────────────────
# t_raw      = np.arange(len(lst))      / fps
# t_filtered = np.arange(len(filtered_signal)) / fps
# t_smoothed = np.arange(len(smoothed)) / fps

# fig, axes = plt.subplots(4, 1, figsize=(14, 10))
# fig.suptitle(f"Breathing Rate: {bpm:.1f} breaths/min", fontsize=13, fontweight='bold')

# # Graph 1 — Raw signal (your original)
# axes[0].plot(t_raw, lst, linewidth=0.8)
# axes[0].set_title("Graph 1 — Raw ROI Intensity")
# axes[0].set_xlabel("Time (s)")

# # Graph 2 — Bandpass filtered (the cleaned breathing waveform)
# axes[1].plot(t_filtered, filtered_signal, color='steelblue', linewidth=1)
# axes[1].set_title("Graph 2 — Bandpass Filtered (0.1–0.8 Hz) — Breathing Waveform")
# axes[1].set_xlabel("Time (s)")
# axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')

# # Graph 3 — Welch PSD (shows the dominant breathing frequency)
# mask = (freqs >= 0.05) & (freqs <= 1.2)
# axes[2].plot(freqs[mask], psd[mask], color='darkorange')
# axes[2].axvline(bpm / 60, color='red', linestyle='--', linewidth=1.5,
#                 label=f'Peak: {bpm:.1f} bpm ({bpm/60:.3f} Hz)')
# axes[2].set_title("Graph 3 — Welch PSD (Power Spectral Density)")
# axes[2].set_xlabel("Frequency (Hz)")
# axes[2].legend()

# # Graph 4 — Your existing smoothed + peaks/bottoms (unchanged)
# slope = np.diff(smoothed)
# peak_x, peak_y, bottom_x, bottom_y = [], [], [], []
# for i in range(1, len(slope)):
#     if slope[i-1] > 0 and slope[i] <= 0:
#         peak_x.append(i); peak_y.append(smoothed[i])
#     elif slope[i-1] < 0 and slope[i] >= 0:
#         bottom_x.append(i); bottom_y.append(smoothed[i])

# axes[3].plot(t_smoothed, smoothed, label="Smoothed Signal")
# axes[3].scatter(np.array(peak_x) / fps,   peak_y,   color='red',  s=10, label='Peaks')
# axes[3].scatter(np.array(bottom_x) / fps, bottom_y, color='blue', s=10, label='Bottoms')
# axes[3].set_title("Graph 4 — Rolling Average with Peaks & Bottoms")
# axes[3].set_xlabel("Time (s)")
# axes[3].legend()

# plt.tight_layout()
# plt.show()

# ###################

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.signal import detrend, butter, filtfilt, welch

# # # =====================================
# # # STEP 0: Create Dummy Breathing Signal
# # # =====================================

# # fps = 30
# # duration = 60

# # t = np.arange(0, duration, 1/fps)

# # # Breathing = 15 BPM = 0.25 Hz
# # breathing = 3 * np.sin(2*np.pi*0.25*t)

# # # Slow upward drift
# # trend = 0.1 * t

# # # Random noise
# # noise = np.random.normal(0, 1, len(t))

# # # Final raw signal
# # raw_signal = breathing + trend + noise

# # # =====================================
# # # STEP 1: Detrend
# # # =====================================

# # detrended_signal = detrend(raw_signal)

# # # =====================================
# # # STEP 2: Bandpass Filter
# # # =====================================

# # nyq = fps/2

# # b, a = butter(
# #     4,
# #     [0.1/nyq, 0.8/nyq],
# #     btype='band'
# # )

# # filtered_signal = filtfilt(
# #     b,
# #     a,
# #     detrended_signal
# # )

# # # =====================================
# # # STEP 3: Welch PSD
# # # =====================================

# # freqs, psd = welch(
# #     filtered_signal,
# #     fs=fps,
# #     nperseg=256
# # )

# # # =====================================
# # # STEP 4: Peak Frequency
# # # =====================================

# # mask = (freqs >= 0.1) & (freqs <= 0.8)

# # peak_freq = freqs[mask][np.argmax(psd[mask])]
# # bpm = peak_freq * 60

# # print(f"Detected BPM = {bpm:.2f}")

# # # =====================================
# # # PLOTS
# # # =====================================

# # plt.figure(figsize=(15,12))

# # # -------------------------------------
# # # Graph 1
# # # -------------------------------------
# # plt.subplot(4,1,1)
# # plt.plot(t, raw_signal)
# # plt.title("STEP 0 : Raw Signal (Breathing + Drift + Noise)")
# # plt.xlabel("Time (s)")

# # # -------------------------------------
# # # Graph 2
# # # -------------------------------------
# # plt.subplot(4,1,2)
# # plt.plot(t, detrended_signal)
# # plt.title("STEP 1 : After Detrend()")
# # plt.xlabel("Time (s)")

# # # -------------------------------------
# # # Graph 3
# # # -------------------------------------
# # plt.subplot(4,1,3)
# # plt.plot(t, filtered_signal)
# # plt.title("STEP 2 : After Bandpass Filter")
# # plt.xlabel("Time (s)")

# # # -------------------------------------
# # # Graph 4
# # # -------------------------------------
# # plt.subplot(4,1,4)

# # mask2 = (freqs >= 0.05) & (freqs <= 1)

# # plt.plot(freqs[mask2], psd[mask2])

# # plt.axvline(
# #     peak_freq,
# #     linestyle="--",
# #     label=f"{bpm:.1f} BPM"
# # )

# # plt.title("STEP 3 : Welch PSD")
# # plt.xlabel("Frequency (Hz)")
# # plt.legend()

# # plt.tight_layout()
# # plt.show()

############################################################

import cv2
import numpy as np
import mediapipe as mp
import utilities as ut
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARKS
# ─────────────────────────────────────────────────────────────────────────────

UPPER_LIPS_LANDMARK      = [13, 206, 426]
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
# video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Swamini\swamini_grey_manual.mpg"
video_path = r"D:\Lie Detection Data HTI\Prem\prem_grey_manual.mpg"

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
    if len(window_signal) < fps * 4:          # need at least 4 sec
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

            # Expand ROI width by 10% each side
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

WINDOW_SEC = 20       # seconds of signal per BPM estimate  (tune: 15–30)
STEP_SEC   = 1        # how often to compute a new BPM value (tune: 1–5)

window_frames = int(WINDOW_SEC * fps)
step_frames   = int(STEP_SEC   * fps)

bpm_values = []
bpm_times  = []

for start in range(0, len(lst) - window_frames + 1, step_frames):
    chunk = lst[start : start + window_frames]
    bpm   = compute_bpm_welch(chunk, fps)
    bpm_values.append(bpm)
    bpm_times.append((start + window_frames / 2) / fps)   # centre of window

bpm_values = np.array(bpm_values)
bpm_times  = np.array(bpm_times)

# Smooth the BPM trace with EMA (reduces frame-to-frame jitter)
alpha      = 0.3
bpm_smooth = np.full_like(bpm_values, np.nan)
for i, b in enumerate(bpm_values):
    if np.isnan(b):
        continue
    bpm_smooth[i] = b if (i == 0 or np.isnan(bpm_smooth[i - 1])) \
                      else alpha * b + (1 - alpha) * bpm_smooth[i - 1]

print(f"\nLive BPM stats  →  mean: {np.nanmean(bpm_values):.1f}"
      f"  min: {np.nanmin(bpm_values):.1f}"
      f"  max: {np.nanmax(bpm_values):.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHED SIGNAL + PEAK / BOTTOM DETECTION (your original logic, kept intact)
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

fig, axes = plt.subplots(5, 1, figsize=(15, 14))
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
axes[3].plot(bpm_times, bpm_values, linewidth=1,   color='silver',  label='Raw BPM (per window)')
axes[3].plot(bpm_times, bpm_smooth, linewidth=2,   color='tomato',  label=f'Smoothed BPM (EMA α={alpha})')
axes[3].axhline(12, color='green', linestyle='--', linewidth=0.8, alpha=0.6)
axes[3].axhline(20, color='green', linestyle='--', linewidth=0.8, alpha=0.6, label='Normal range (12–20)')
axes[3].set_ylim(0, 40)
axes[3].set_title("Graph 4 — Live Breathing Rate Over Time")
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Breaths / min")
axes[3].legend()

# ── Graph 5: Smoothed signal + peaks / bottoms (your original) ───────────
axes[4].plot(t_smoothed, smoothed, linewidth=1, label='Smoothed signal (MA-10)')
axes[4].scatter(np.array(peak_x)   / fps, peak_y,   color='red',  s=12, zorder=3, label='Peaks')
axes[4].scatter(np.array(bottom_x) / fps, bottom_y, color='blue', s=12, zorder=3, label='Bottoms')
axes[4].set_title("Graph 5 — Moving Average with Peaks & Bottoms")
axes[4].set_xlabel("Time (s)")
axes[4].legend()

plt.tight_layout()
plt.show()