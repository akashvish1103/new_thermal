# ============================================================
# BREATHING PATTERN FROM NOSE ROI
# ============================================================
# 
# Pipeline:
#
# Thermal Video
#       ↓
# MediaPipe FaceMesh
#       ↓
# Breathing ROI
#       ↓
# ORIGINAL GREY IMAGE
#       ↓
# Mean Pixel Intensity
#       ↓
# Simple Moving Average (SMA)
#       ↓
# Breath Peak Detection
#       ↓
# Breathing Rate (BPM)
#
# NOTE:
# Only SMA is used for smoothing.
# No Savitzky-Golay / EWMA / other smoothing.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import mediapipe as mp
import numpy as np
import cv2
import utilities as ut
import matplotlib.pyplot as plt


# ============================================================
# VIDEO PATH
# ============================================================

video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40.wmv"


# ============================================================
# PARAMETERS
# ============================================================

# ------------------------------------------------------------
# Simple Moving Average
# ------------------------------------------------------------

SMA_WINDOW = 30

# Larger value:
#     smoother signal
#     more noise reduction
#
# Smaller value:
#     less smoothing
#     preserves faster changes


# ------------------------------------------------------------
# Breath Detection
# ------------------------------------------------------------

# Minimum difference between peak and following valley.
#
# IMPORTANT:
# This value depends on your pixel-intensity signal.
# Start with a small value and adjust based on your graph.

MIN_PEAK_VALLEY_DIFF = 2.0


# Minimum time between two detected breaths.

REFRACTORY_PERIOD = 2.0


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
# NOSE LANDMARKS
# ============================================================

UPPER_LIPS_LANDMARK = [13, 206, 426]

NOSE_HORIZONTAL_LANDMARKS = [64, 278]

NOSE_VERTICAL_LANDMARKS = [4, 94]

NOSE_LANDMARKS = (
    NOSE_HORIZONTAL_LANDMARKS
    + NOSE_VERTICAL_LANDMARKS
    + UPPER_LIPS_LANDMARK
)
   

# ============================================================
# SIGNAL STORAGE
# ============================================================

arr = []


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 60)
print("VIDEO INFORMATION")
print("=" * 60)

print(f"Video FPS: {fps}")


if fps <= 0:

    print("ERROR: Could not determine video FPS.")

    cap.release()
    face_mesh.close()

    exit()


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ========================================================
    # Convert frame to GREY
    # ========================================================

    grey = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # Create transformed image
    #
    # IMPORTANT:
    # This transformed image is ONLY used for
    # MediaPipe / landmark detection.
    #
    # Pixel intensity for breathing signal is taken
    # from ORIGINAL GREY image.
    # ========================================================

    transformed_grey = ut.get_transformed_image(
        grey
    )


    # ========================================================
    # MediaPipe expects 3-channel image
    # ========================================================

    rgb = cv2.cvtColor(
        transformed_grey,
        cv2.COLOR_GRAY2BGR
    )


    # ========================================================
    # Process face
    # ========================================================

    results = face_mesh.process(rgb)


    # ========================================================
    # If face detected
    # ========================================================

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:


            # =================================================
            # GET BREATHING ROI COORDINATES
            # =================================================

            (
                top_left_coords_bb,
                bottom_right_coords_bb,
                got_frame
            ) = ut.get_breathing_roi_cords(
                transformed_grey,
                face_landmarks
            )


            # =================================================
            # OTHER ROI FUNCTIONS
            #
            # These are retained from your original code.
            # They are not used for breathing signal.
            # =================================================

            polygon_points, mean_pixel, got_frame = (
                ut.get_forhead_poly_coords(
                    transformed_grey,
                    face_landmarks
                )
            )


            l, r, got_frame = (
                ut.get_cheeks_coordinates(
                    transformed_grey,
                    face_landmarks,
                    [],
                    []
                )
            )


            (
                top_left_coords,
                bottom_right_coords,
                top_right_coords,
                bottom_left_coords,
                got_frame
            ) = ut.get_eyes_coordinates(
                transformed_grey,
                face_landmarks
            )


            top_left_coords, bottom_right_coords, got_frame = (
                ut.get_nose_tip_coordinates(
                    transformed_grey,
                    face_landmarks
                )
            )


            # =================================================
            # EXTRACT BREATHING ROI
            #
            # IMPORTANT:
            # ORIGINAL GREY IMAGE
            # =================================================

            extracted_breathing_box_from_grey = grey[
                top_left_coords_bb[1]:
                bottom_right_coords_bb[1],

                top_left_coords_bb[0]:
                bottom_right_coords_bb[0]
            ]


            # =================================================
            # CALCULATE MEAN PIXEL INTENSITY
            # =================================================

            mean_pixel = ut.get_mean_of_ROI(
                extracted_breathing_box_from_grey
            )


            # =================================================
            # STORE SIGNAL
            # =================================================

            arr.append(
                mean_pixel
            )


            print(
                f"Average Pixel Value = {mean_pixel:.2f}"
            )


    # ========================================================
    # DISPLAY VIDEO / ROI
    # ========================================================

    if results.multi_face_landmarks:

        cv2.imshow(
            "Transformed Grey",
            got_frame
        )


        cv2.imshow(
            "RGB Frame",
            frame
        )


        cv2.imshow(
            "Extracted Breathing ROI",
            extracted_breathing_box_from_grey
        )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()

cv2.destroyAllWindows()

face_mesh.close()


# ============================================================
# CONVERT SIGNAL TO NUMPY ARRAY
# ============================================================

arr = np.array(
    arr,
    dtype=float
)


print("\n")
print("=" * 60)
print("SIGNAL INFORMATION")
print("=" * 60)

print(
    f"Total signal samples: {len(arr)}"
)


if len(arr) == 0:

    print("ERROR: No breathing signal was extracted.")

    exit()


# ============================================================
# SIMPLE MOVING AVERAGE FUNCTION
# ============================================================

def moving_average(signal, window):

    """
    Centered Simple Moving Average.

    Example:

        signal = [1, 2, 3, 4, 5]

        window = 3

    Each point is replaced by the average
    of neighboring points.

    Only SMA is used.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )


    # --------------------------------------------------------
    # If signal is shorter than window
    # --------------------------------------------------------

    if len(signal) < window:

        return signal.copy()


    # --------------------------------------------------------
    # Calculate moving average
    # --------------------------------------------------------

    sma = np.convolve(
        signal,
        np.ones(window) / window,
        mode='valid'
    )


    # --------------------------------------------------------
    # Padding
    #
    # This makes SMA length equal to original signal length.
    # --------------------------------------------------------

    pad_left = [
        sma[0]
    ] * (window // 2)


    pad_right = [
        sma[-1]
    ] * (window - 1 - window // 2)


    sma = np.concatenate(
        [
            pad_left,
            sma,
            pad_right
        ]
    )


    return sma


# ============================================================
# APPLY SIMPLE MOVING AVERAGE
# ============================================================

sma_signal = moving_average(
    arr,
    SMA_WINDOW
)


print(
    f"SMA window: {SMA_WINDOW}"
)


# ============================================================
# CREATE TIME AXIS
# ============================================================

time = np.arange(
    len(arr)
) / fps


# ============================================================
# BREATH DETECTION FUNCTION
# ============================================================

def detect_breath_peaks(
    time,
    signal,
    min_peak_valley_diff,
    refractory_period
):

    """
    Detect breathing cycles using:

        Peak → Valley

    A breath is counted when:

        1. A local peak is found.
        2. A valley follows the peak.
        3. Peak-to-valley difference is large enough.
        4. Enough time has passed since previous breath.
    """


    # --------------------------------------------------------
    # Not enough data
    # --------------------------------------------------------

    if len(signal) < 5:

        return []


    y = np.asarray(
        signal,
        dtype=float
    )


    # --------------------------------------------------------
    # First derivative / difference
    # --------------------------------------------------------

    dy = np.diff(y)


    # --------------------------------------------------------
    # Store peaks and valleys
    # --------------------------------------------------------

    raw_peaks = []

    raw_valleys = []


    # --------------------------------------------------------
    # Find local peaks and valleys
    # --------------------------------------------------------

    for i in range(1, len(dy)):

        # ----------------------------------------------------
        # Rising → Falling
        # ----------------------------------------------------

        if (
            dy[i - 1] > 0
            and
            dy[i] <= 0
        ):

            raw_peaks.append(i)


        # ----------------------------------------------------
        # Falling → Rising
        # ----------------------------------------------------

        elif (
            dy[i - 1] < 0
            and
            dy[i] >= 0
        ):

            raw_valleys.append(i)


    # --------------------------------------------------------
    # If no peaks or valleys
    # --------------------------------------------------------

    if (
        not raw_peaks
        or
        not raw_valleys
    ):

        return []


    # ========================================================
    # CONFIRM BREATHING CYCLES
    # ========================================================

    breath_markers = []

    last_cycle_time = -999

    used_valleys = set()


    for peak_index in raw_peaks:


        # ====================================================
        # Find first unused valley after peak
        # ====================================================

        next_valleys = [
            valley_index
            for valley_index in raw_valleys

            if (
                valley_index > peak_index
                and
                valley_index not in used_valleys
            )
        ]


        if not next_valleys:

            continue


        valley_index = next_valleys[0]


        # ====================================================
        # Peak and valley values
        # ====================================================

        peak_value = y[
            peak_index
        ]

        valley_value = y[
            valley_index
        ]


        # ====================================================
        # Peak → valley difference
        # ====================================================

        difference = (
            peak_value
            -
            valley_value
        )


        # ----------------------------------------------------
        # Ignore small fluctuations
        # ----------------------------------------------------

        if difference < min_peak_valley_diff:

            continue


        # ====================================================
        # Refractory period
        # ====================================================

        peak_time = time[
            peak_index
        ]


        if (
            peak_time
            -
            last_cycle_time
            <
            refractory_period
        ):

            continue


        # ====================================================
        # CONFIRMED BREATH
        # ====================================================

        breath_markers.append(
            (
                peak_time,
                peak_value
            )
        )


        used_valleys.add(
            valley_index
        )


        last_cycle_time = peak_time


    return breath_markers


# ============================================================
# DETECT BREATHS USING SMA SIGNAL
# ============================================================

breath_peaks = detect_breath_peaks(
    time,
    sma_signal,
    MIN_PEAK_VALLEY_DIFF,
    REFRACTORY_PERIOD
)


print(
    f"Detected breaths: {len(breath_peaks)}"
)


# ============================================================
# CALCULATE BREATHING RATE
# ============================================================

if len(breath_peaks) >= 2:

    first_breath_time = (
        breath_peaks[0][0]
    )

    last_breath_time = (
        breath_peaks[-1][0]
    )


    breathing_duration = (
        last_breath_time
        -
        first_breath_time
    )


    if breathing_duration > 0:

        bpm = (
            (len(breath_peaks) - 1)
            /
            breathing_duration
            *
            60
        )

    else:

        bpm = 0


else:

    bpm = 0


print(
    f"Breathing Rate: {bpm:.2f} BPM"
)


# ============================================================
# EXTRACT BREATH PEAK COORDINATES
# ============================================================

if breath_peaks:

    peak_times = [
        peak[0]
        for peak in breath_peaks
    ]


    peak_values = [
        peak[1]
        for peak in breath_peaks
    ]

else:

    peak_times = []

    peak_values = []


# ============================================================
# FINAL BREATHING PATTERN GRAPH
# ============================================================

plt.figure(
    figsize=(16, 7)
)


# ============================================================
# RAW SIGNAL
# ============================================================

plt.plot(
    time,
    arr,
    linewidth=0.7,
    alpha=0.30,
    label="Raw Nose ROI Signal"
)


# ============================================================
# SMA SIGNAL
# ============================================================

plt.plot(
    time,
    sma_signal,
    linewidth=2.5,
    label=f"Simple Moving Average (window={SMA_WINDOW})"
)


# ============================================================
# BREATH PEAKS
# ============================================================

plt.scatter(
    peak_times,
    peak_values,
    s=70,
    zorder=5,
    label="Detected Breath / Exhalation Peak"
)


# ============================================================
# BPM TEXT
# ============================================================

plt.text(
    0.02,
    0.95,

    f"Breathing Rate: {bpm:.1f} BPM",

    transform=plt.gca().transAxes,

    fontsize=13,

    verticalalignment='top',

    bbox=dict(
        boxstyle="round",
        alpha=0.8
    )
)


# ============================================================
# GRAPH LABELS
# ============================================================

plt.xlabel(
    "Time (seconds)",
    fontsize=12
)


plt.ylabel(
    "Mean Pixel Intensity",
    fontsize=12
)


plt.title(
    "Breathing Pattern from Nose ROI using Simple Moving Average",
    fontsize=15
)


# ============================================================
# GRID
# ============================================================

plt.grid(
    True,
    alpha=0.3
)


# ============================================================
# LEGEND
# ============================================================

plt.legend()


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SHOW GRAPH
# ============================================================

plt.show()