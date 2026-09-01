# ============================================================
# DRIVER FILE
# Using all ROI functions from utilities.py
# With MediaPipe landmark smoothing
#
# IMPORTANT DESIGN NOTE (per your request):
#   - MediaPipe is used ONLY for face/landmark detection. It runs
#     on the CONTRAST-ENHANCED ("transformed") grey frame, because
#     that gives MediaPipe better detection odds on thermal footage.
#   - ALL pixel-intensity extraction (ROI means, CSV log, stats,
#     plots) is done on the PLAIN cv2-greyscaled frame ("grey"),
#     which is never touched by get_transformed_image(). No
#     temperature mapping is applied anywhere - everything below
#     is raw 0-255 pixel intensity.
# ============================================================

import os
import csv
from collections import defaultdict

import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt

import utilities as ut


# ============================================================
# VIDEO PATH
# ============================================================

# video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\7.mp4"
video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\4.mp4"
video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\8.mp4"

# Other examples:
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"


# ============================================================
# OUTPUT PATHS (auto-derived from the video's own folder/name)
# ============================================================

video_dir = os.path.dirname(video_path)
video_name = os.path.splitext(os.path.basename(video_path))[0]

CSV_PATH = os.path.join(video_dir, f"{video_name}_roi_pixel_log.csv")
PLOT_PATH = os.path.join(video_dir, f"{video_name}_roi_pixel_plot.png")


# ============================================================
# MEDIAPIPE FACE MESH SETUP
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# EMA SMOOTHING SETTINGS (LANDMARK SMOOTHING - unchanged)
# ============================================================

# Smaller alpha = smoother movement
#
# 0.05 -> very smooth / slow
# 0.10 -> smooth
# 0.15 -> good starting point
# 0.25 -> faster response
# 1.00 -> no smoothing

alpha = 0.08

# This will store the previous frame's smoothed landmarks
previous_landmarks = None


# ============================================================
# SMA SMOOTHING SETTINGS (FINAL PLOT SMOOTHING - separate knob)
# ============================================================

# Window size in FRAMES for the Simple Moving Average used only
# on the end-of-run plot (does not affect landmark smoothing above
# or the raw values written to the CSV).
#
# Smaller window -> follows raw signal more closely
# Larger window  -> smoother / slower curve
SMA_WINDOW = 50


# ============================================================
# PHASE MARKERS (SESSION TIMELINE) - EDIT PER VIDEO
# ============================================================
#
# Every video's protocol timing is different, so this is just a
# plain list you edit for each session. Each entry is:
#
#   (label, start_second, end_second)
#
# - Times are in SECONDS from the start of the video (0 = frame 1).
# - Segments do NOT need to be contiguous or start at 0 - e.g. if
#   the first 3 seconds are just setup/no protocol yet, simply
#   start your first segment at 3.
# - Leave the list empty ( PHASE_MARKERS = [] ) to skip markers
#   entirely and just get the plain raw+SMA plot.
#
# EXAMPLE (matches what you described):
#   Eyes open   : 3s   -> 93s   (1:33)
#   Eyes closed : 93s  -> 300s  (5:00)
#   Meditation  : 300s -> 600s  (10:00)
#   Eyes open   : 600s -> 660s  (11:00)

PHASE_MARKERS = [
    ("Eyes Open",   5,   95),
    ("Eyes Closed", 95,  185),
    ("Meditation",  185, 785),
    ("Eyes Open",   785, 875),
]


# ============================================================
# ROI NAMES (used for CSV header / stats table / dict keys)
# ============================================================

ROI_NAMES = [
    "breathing",
    "forehead",
    "left_cheek",
    "right_cheek",
    "left_eye",
    "right_eye",
    "nose_tip",
]


# ============================================================
# HELPERS: pixel-intensity extraction directly from PLAIN grey
# ============================================================

def mean_from_rect(grey_frame, top_left, bottom_right):
    """
    Mean pixel intensity inside a rectangle. Coordinates are sorted
    and clamped to the frame so it doesn't matter which corner
    ordering an ROI function happens to hand back.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(grey_frame.shape[1], x2)
    y2 = min(grey_frame.shape[0], y2)

    crop = grey_frame[y1:y2, x1:x2]

    if crop.size == 0:
        return float("nan")

    return float(np.mean(crop))


def mean_from_polygon(grey_frame, points_list):
    """
    Mean pixel intensity inside a polygon (used for the cheeks,
    and re-used for the forehead polygon so that the mean is always
    computed on the pristine grey frame, never the annotated one).
    """
    pts = np.array(points_list, dtype=np.int32)
    mask = np.zeros(grey_frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return float(cv2.mean(grey_frame, mask=mask)[0])


def sma_smooth(values, window):
    """
    Centered Simple Moving Average for the final plot
    (independent of landmark smoothing / CSV raw values).

    NaN gaps (frames with no face detected) are forward-filled
    first so a missing-face gap doesn't create a hole in the
    rolling window; edges are padded so the output is the same
    length as the input.
    """
    values = np.array(values, dtype=np.float64)

    # forward-fill NaNs
    filled = values.copy()
    last_valid = None
    for i in range(len(filled)):
        if np.isnan(filled[i]):
            if last_valid is not None:
                filled[i] = last_valid
        else:
            last_valid = filled[i]

    # back-fill any leading NaNs (no valid value seen yet)
    if np.any(np.isnan(filled)):
        first_valid_idx = np.argmax(~np.isnan(filled))
        filled[:first_valid_idx] = filled[first_valid_idx]

    if np.all(np.isnan(filled)):
        return filled  # no valid data at all

    window = max(1, int(window))
    kernel = np.ones(window) / window

    pad_left = window // 2
    pad_right = window - pad_left - 1
    padded = np.pad(filled, (pad_left, pad_right), mode="edge")

    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def add_phase_markers(axes, phases):
    """
    Draws session-phase markers across all given subplots:
      - a light shaded band per phase (so phases are visually distinct)
      - a dashed vertical line at every phase boundary (shared across
        all subplots so they line up)
      - a text label per phase, placed only on the TOP subplot so it
        isn't repeated 3 times

    `phases` is the PHASE_MARKERS list: (label, start_sec, end_sec)
    """
    if not phases:
        return

    colors = plt.cm.tab10.colors
    boundary_times = set()

    for i, (label, start, end) in enumerate(phases):
        color = colors[i % len(colors)]
        for ax in axes:
            ax.axvspan(start, end, color=color, alpha=0.08)
        boundary_times.add(start)
        boundary_times.add(end)

    for t in sorted(boundary_times):
        for ax in axes:
            ax.axvline(t, color="black", linestyle="--", linewidth=0.7, alpha=0.5)

    top_ax = axes[0]
    for label, start, end in phases:
        mid = (start + end) / 2
        top_ax.text(
            mid, 0.95, label,
            transform=top_ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.75)
        )


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0  # fallback if the container doesn't report a valid fps


# ============================================================
# FRAME COUNTER
# ============================================================

frame_number = 0


# ============================================================
# DATA STORAGE
# ============================================================

log_rows = []                     # one dict per frame -> written to CSV
roi_values = defaultdict(list)    # roi_name -> list of valid (face-detected) intensity values


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1
    timestamp_sec = frame_number / fps


    # ========================================================
    # ORIGINAL FRAME
    # ========================================================

    original_frame = frame.copy()


    # ========================================================
    # PLAIN GREYSCALE
    # This is the ONLY frame used for pixel-intensity extraction.
    # ========================================================

    grey_raw = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    grey = cv2.rotate(
        grey_raw,
        cv2.ROTATE_90_CLOCKWISE
    )


    # ========================================================
    # CONTRAST-ENHANCED GREYSCALE
    # Used ONLY to help MediaPipe detect the face/landmarks.
    # Never used for any intensity/temperature calculation.
    # ========================================================

    transformed_grey = ut.get_transformed_image(grey_raw)

    transformed_grey = cv2.rotate(
        transformed_grey,
        cv2.ROTATE_90_CLOCKWISE
    )


    # ========================================================
    # IMPORTANT:
    # DEFAULT FRAME
    #
    # If MediaPipe does NOT detect a face,
    # got_frame still exists. It's based on the PLAIN grey
    # frame (not the transformed one), per your requirement.
    # ========================================================

    got_frame = grey.copy()


    # ========================================================
    # CONVERT TO RGB FOR MEDIAPIPE (detection only)
    # ========================================================

    rgb = cv2.cvtColor(
        transformed_grey,
        cv2.COLOR_GRAY2RGB
    )


    # ========================================================
    # MEDIAPIPE FACE MESH
    # ========================================================

    results = face_mesh.process(rgb)


    # ========================================================
    # ROW SKELETON FOR THIS FRAME (CSV log)
    # ========================================================

    row = {
        "frame": frame_number,
        "timestamp_sec": round(timestamp_sec, 3),
        "face_detected": False,
    }
    for name in ROI_NAMES:
        row[name] = float("nan")


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if results.multi_face_landmarks:

        print(
            f"Frame {frame_number}: FACE = YES"
        )

        row["face_detected"] = True


        # ====================================================
        # ONLY ONE FACE BECAUSE max_num_faces=1
        # ====================================================

        for face_landmarks in results.multi_face_landmarks:


            # =================================================
            # LANDMARK SMOOTHING
            # =================================================

            # First detected frame
            if previous_landmarks is None:

                previous_landmarks = np.array(
                    [
                        [lm.x, lm.y, lm.z]
                        for lm in face_landmarks.landmark
                    ],
                    dtype=np.float32
                )


            # Subsequent frames
            else:

                current_landmarks = np.array(
                    [
                        [lm.x, lm.y, lm.z]
                        for lm in face_landmarks.landmark
                    ],
                    dtype=np.float32
                )


                # =============================================
                # EXPONENTIAL MOVING AVERAGE
                # =============================================

                previous_landmarks = (
                    alpha * current_landmarks
                    +
                    (1 - alpha) * previous_landmarks
                )


            # =================================================
            # PUT SMOOTHED LANDMARKS BACK INTO MEDIAPIPE
            # OBJECT
            # =================================================

            for i, lm in enumerate(
                face_landmarks.landmark
            ):

                lm.x = float(
                    previous_landmarks[i, 0]
                )

                lm.y = float(
                    previous_landmarks[i, 1]
                )

                lm.z = float(
                    previous_landmarks[i, 2]
                )


            # =================================================
            # ALL ROI FUNCTIONS RUN ON got_frame (PLAIN GREY,
            # ANNOTATED PROGRESSIVELY) FOR DISPLAY. THE ACTUAL
            # INTENSITY NUMBERS WE LOG/PLOT ARE RE-COMPUTED FROM
            # THE UNTOUCHED "grey" ARRAY SO ANNOTATIONS DRAWN BY
            # EARLIER ROIs NEVER LEAK INTO LATER ROI MEANS.
            # =================================================


            # -------------------------------------------------
            # BREATHING ROI
            # -------------------------------------------------

            (
                top_left_cords,
                bottom_right_cords,
                got_frame
            ) = ut.get_breathing_roi_cords(
                got_frame,
                face_landmarks
            )

            breathing_mean = mean_from_rect(
                grey, top_left_cords, bottom_right_cords
            )


            # -------------------------------------------------
            # FOREHEAD ROI
            # -------------------------------------------------

            (
                polygon_points,
                _mean_pixel_ignored,
                got_frame
            ) = ut.get_forhead_poly_coords(
                got_frame,
                face_landmarks
            )

            forehead_mean = mean_from_polygon(
                grey, polygon_points
            )


            # -------------------------------------------------
            # CHEEKS
            # -------------------------------------------------

            (
                l,
                r,
                got_frame
            ) = ut.get_cheeks_coordinates(
                got_frame,
                face_landmarks,
                [],
                []
            )

            left_cheek_mean = mean_from_polygon(grey, l)
            right_cheek_mean = mean_from_polygon(grey, r)


            # -------------------------------------------------
            # EYES
            # -------------------------------------------------

            (
                top_left_coords,
                bottom_right_coords,
                top_right_coords,
                bottom_left_coords,
                got_frame
            ) = ut.get_eyes_coordinates(
                got_frame,
                face_landmarks
            )

            left_eye_mean = mean_from_rect(
                grey, top_left_coords, bottom_right_coords
            )
            right_eye_mean = mean_from_rect(
                grey, bottom_left_coords, top_right_coords
            )


            # -------------------------------------------------
            # NOSE TIP
            # -------------------------------------------------

            (
                nose_top_left_coords,
                nose_bottom_right_coords,
                got_frame
            ) = ut.get_nose_tip_coordinates(
                got_frame,
                face_landmarks
            )

            nose_mean = mean_from_rect(
                grey, nose_top_left_coords, nose_bottom_right_coords
            )


            # =================================================
            # STORE THIS FRAME'S VALUES
            # =================================================

            row["breathing"]   = round(breathing_mean, 3)
            row["forehead"]    = round(forehead_mean, 3)
            row["left_cheek"]  = round(left_cheek_mean, 3)
            row["right_cheek"] = round(right_cheek_mean, 3)
            row["left_eye"]    = round(left_eye_mean, 3)
            row["right_eye"]   = round(right_eye_mean, 3)
            row["nose_tip"]    = round(nose_mean, 3)

            roi_values["breathing"].append(breathing_mean)
            roi_values["forehead"].append(forehead_mean)
            roi_values["left_cheek"].append(left_cheek_mean)
            roi_values["right_cheek"].append(right_cheek_mean)
            roi_values["left_eye"].append(left_eye_mean)
            roi_values["right_eye"].append(right_eye_mean)
            roi_values["nose_tip"].append(nose_mean)


            # =================================================
            # PRINT COORDINATES + INTENSITIES
            # =================================================

            print(
                "Breathing ROI:",
                top_left_cords,
                bottom_right_cords,
                "| Mean:", round(breathing_mean, 2)
            )

            print(
                "Forehead:",
                polygon_points,
                "| Mean:", round(forehead_mean, 2)
            )

            print(
                "Cheeks -> Left Mean:", round(left_cheek_mean, 2),
                " Right Mean:", round(right_cheek_mean, 2)
            )

            print(
                "Eyes -> Left Mean:", round(left_eye_mean, 2),
                " Right Mean:", round(right_eye_mean, 2)
            )

            print(
                "Nose -> Mean:", round(nose_mean, 2)
            )

            print("-" * 100)


    # ========================================================
    # NO FACE DETECTED
    # ========================================================

    else:

        print(
            f"Frame {frame_number}: FACE = NO"
        )

        # No ROI functions are called.
        #
        # got_frame already contains:
        #
        # grey.copy()
        #
        # Therefore the program continues normally, and this
        # frame is logged with face_detected=False and NaNs
        # for every ROI (excluded from the end-of-run stats).


    log_rows.append(row)


    # ========================================================
    # DISPLAY PLAIN GREY IMAGE WITH ROIs
    # ========================================================

    cv2.imshow(
        "Plain Grey - Smoothed ROIs (Pixel Intensity Frame)",
        got_frame
    )


    # ========================================================
    # DISPLAY ORIGINAL VIDEO
    # ========================================================

    cv2.imshow(
        "Original Frame",
        original_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

face_mesh.close()

cv2.destroyAllWindows()


# ============================================================
# WRITE PER-FRAME CSV LOG (ALL ROIs)
# ============================================================

fieldnames = ["frame", "timestamp_sec", "face_detected"] + ROI_NAMES

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\nPer-frame ROI pixel-intensity log written to:\n  {CSV_PATH}")


# ============================================================
# TERMINAL STATS: mean / min / max / std dev PER ROI
# ============================================================

print("\n" + "=" * 100)
print("SESSION SUMMARY - PIXEL INTENSITY STATS PER ROI (face-detected frames only)")
print("=" * 100)
print(f"{'ROI':<15}{'Frames':>10}{'Mean':>12}{'Min':>12}{'Max':>12}{'StdDev':>12}")
print("-" * 100)

for name in ROI_NAMES:
    values = np.array(roi_values[name], dtype=np.float64)
    values = values[~np.isnan(values)]

    if values.size == 0:
        print(f"{name:<15}{0:>10}{'--':>12}{'--':>12}{'--':>12}{'--':>12}")
        continue

    print(
        f"{name:<15}"
        f"{values.size:>10}"
        f"{np.mean(values):>12.2f}"
        f"{np.min(values):>12.2f}"
        f"{np.max(values):>12.2f}"
        f"{np.std(values):>12.2f}"
    )

print("=" * 100)
print(f"Total frames processed : {frame_number}")
print(f"Frames with face       : {sum(1 for r in log_rows if r['face_detected'])}")
print(f"Frames without face    : {sum(1 for r in log_rows if not r['face_detected'])}")
print("=" * 100)


# ============================================================
# FINAL PLOT: FOREHEAD + LEFT CHEEK + RIGHT CHEEK OVER TIME
# (raw + Simple Moving Average smoothed)
# ============================================================

timestamps = [r["timestamp_sec"] for r in log_rows]

forehead_series    = [r["forehead"]    for r in log_rows]
left_cheek_series  = [r["left_cheek"]  for r in log_rows]
right_cheek_series = [r["right_cheek"] for r in log_rows]

forehead_smooth    = sma_smooth(forehead_series, SMA_WINDOW)
left_cheek_smooth  = sma_smooth(left_cheek_series, SMA_WINDOW)
right_cheek_smooth = sma_smooth(right_cheek_series, SMA_WINDOW)

sma_label = f"Smoothed (SMA, window={SMA_WINDOW})"

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(timestamps, forehead_series, color="lightgray", linewidth=0.8, label="Raw")
axes[0].plot(timestamps, forehead_smooth, color="crimson", linewidth=1.8, label=sma_label)
axes[0].set_ylabel("Pixel Intensity")
axes[0].set_title("Forehead ROI")
axes[0].legend(loc="upper right")
axes[0].grid(alpha=0.3)

axes[1].plot(timestamps, left_cheek_series, color="lightgray", linewidth=0.8, label="Raw")
axes[1].plot(timestamps, left_cheek_smooth, color="royalblue", linewidth=1.8, label=sma_label)
axes[1].set_ylabel("Pixel Intensity")
axes[1].set_title("Left Cheek ROI")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

axes[2].plot(timestamps, right_cheek_series, color="lightgray", linewidth=0.8, label="Raw")
axes[2].plot(timestamps, right_cheek_smooth, color="seagreen", linewidth=1.8, label=sma_label)
axes[2].set_ylabel("Pixel Intensity")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_title("Right Cheek ROI")
axes[2].legend(loc="upper right")
axes[2].grid(alpha=0.3)

add_phase_markers(axes, PHASE_MARKERS)

fig.suptitle("ROI Pixel Intensity Over Time (Yoga Session)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(PLOT_PATH, dpi=150)
print(f"Plot saved to:\n  {PLOT_PATH}")

plt.show()