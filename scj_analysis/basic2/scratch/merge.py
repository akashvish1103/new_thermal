# cld merge

"""
multi_roi_thermal_analysis.py

Tracks multiple facial ROIs (breathing box, forehead, left/right cheek,
left/right inner-eye, nose tip) across a thermal video, converts each to
temperature (°C) using dynamic per-video calibration, and shows:

    1. A LIVE graph of every ROI's temperature, SMA-smoothed.
    2. A LIVE graph of left-vs-right cheek and left-vs-right inner-eye
       temperature differences, SMA-smoothed.
    3. An end-of-run printed summary of those two differences
       (mean / min / max / final).

Design notes
------------
- Landmark DETECTION runs on the sharpened frame (`ut.get_transformed_image`)
  for stability.
- The actual pixel values used for temperature are pulled from the PLAIN,
  unsharpened grayscale frame, using the coordinates returned by detection.
  This keeps temperature readings free of sharpening artifacts while still
  benefiting from sharpening for landmark stability.
- ROI extraction is registry-style (`extract_all_rois`) so adding a new ROI
  later is: write one `extract_x()` wrapper -> call it in `extract_all_rois`
  -> add its name to ROI_NAMES. Nothing else needs to change.
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

import utilities as ut
import utility_linear_mapping as ulm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_PATH = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"

SMA_WINDOW_SEC          = 1.0   # smoothing window, in seconds (converted to frames once fps is known)
LIVE_PLOT               = True  # set False to skip live plotting (still get end-of-run summary)
PLOT_UPDATES_PER_SECOND = 5     # how often the live plot redraws, in updates/sec of video time

ROI_NAMES = [
    'breathing',
    'forehead',
    'left_cheek',
    'right_cheek',
    'left_eye',
    'right_eye',
    'nose_tip',
]

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION (once per video — dynamic pixel → °C mapping from the legend)
# ─────────────────────────────────────────────────────────────────────────────

_pixel_column = ulm.get_one_pixel_column(VIDEO_PATH)
_min_temp, _max_temp, _first_pixel, _last_pixel = \
    ulm.get_first_last_element_from_OnePixelColumn_AND_min_max_from_filename(
        _pixel_column, VIDEO_PATH
    )

print(f"Calibration → min: {_min_temp}°C  max: {_max_temp}°C  "
      f"first_pixel: {_first_pixel}  last_pixel: {_last_pixel}")


def to_temp(pixel_value):
    """Pixel intensity -> temperature (°C), using this video's calibration."""
    if pixel_value is None or np.isnan(pixel_value):
        return np.nan
    return ulm.map_pixel_to_temperature(
        pixel_value, _min_temp, _max_temp, _first_pixel, _last_pixel
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROI EXTRACTION WRAPPERS  (one per utilities.py function)
# Each returns ('rect', (p1, p2)) or ('poly', points) plus the (mutated,
# overlay-drawn) display frame. Add new ROIs by writing one of these.
# ─────────────────────────────────────────────────────────────────────────────

def extract_breathing(display_frame, lm):
    tl, br, display_frame = ut.get_breathing_roi_cords(display_frame, lm)
    return {'breathing': ('rect', (tl, br))}, display_frame


def extract_forehead(display_frame, lm):
    polygon_points, _unused_mean, display_frame = ut.get_forhead_poly_coords(display_frame, lm)
    return {'forehead': ('poly', polygon_points)}, display_frame


def extract_cheeks(display_frame, lm):
    left_pts, right_pts, display_frame = ut.get_cheeks_coordinates(display_frame, lm, [], [])
    return {
        'left_cheek':  ('poly', left_pts),
        'right_cheek': ('poly', right_pts),
    }, display_frame


def extract_eyes(display_frame, lm):
    tl, br, tr, bl, display_frame = ut.get_eyes_coordinates(display_frame, lm)
    return {
        'left_eye':  ('rect', (tl, br)),
        'right_eye': ('rect', (tr, bl)),
    }, display_frame


def extract_nose(display_frame, lm):
    tl, br, display_frame = ut.get_nose_tip_coordinates(display_frame, lm)
    return {'nose_tip': ('rect', (tl, br))}, display_frame


def extract_all_rois(display_frame, lm):
    """Runs every registered extractor and merges results into one dict."""
    rois = {}

    for extractor in (extract_breathing, extract_forehead, extract_cheeks,
                       extract_eyes, extract_nose):
        partial, display_frame = extractor(display_frame, lm)
        rois.update(partial)

    return rois, display_frame


# ─────────────────────────────────────────────────────────────────────────────
# PIXEL EXTRACTION FROM PLAIN (UNSHARPENED) GREY FRAME
# ─────────────────────────────────────────────────────────────────────────────

def get_roi_mean_pixel(plain_grey, roi):
    """Mean pixel intensity of a ROI, read from the plain grey frame."""
    kind, data = roi
    h, w = plain_grey.shape

    if kind == 'rect':
        p1, p2 = data
        x_min, x_max = sorted((p1[0], p2[0]))
        y_min, y_max = sorted((p1[1], p2[1]))
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        if x_max <= x_min or y_max <= y_min:
            return np.nan
        return float(np.mean(plain_grey[y_min:y_max, x_min:x_max]))

    elif kind == 'poly':
        points = np.array(data, dtype=np.int32)
        mask = np.zeros(plain_grey.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return float(cv2.mean(plain_grey, mask=mask)[0])

    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE MOVING AVERAGE
# ─────────────────────────────────────────────────────────────────────────────

def sma(values, window):
    """SMA of the last `window` entries of `values`. NaN until enough data."""
    tail = values[-window:]
    if len(tail) < window:
        return np.nan
    return float(np.nanmean(tail))


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO SETUP
# ─────────────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0
    print("WARNING: could not read FPS from video, defaulting to 25.0")
print(f"Video FPS: {fps:.2f}")

SMA_WINDOW = max(1, int(SMA_WINDOW_SEC * fps))
PLOT_UPDATE_EVERY_N_FRAMES = max(1, int(fps / PLOT_UPDATES_PER_SECOND))

# ─────────────────────────────────────────────────────────────────────────────
# DATA STORAGE
# ─────────────────────────────────────────────────────────────────────────────

times = []                                                    # seconds, one entry per frame WITH a detected face
roi_data = {name: {'raw': [], 'smooth': []} for name in ROI_NAMES}

cheek_diff_raw, cheek_diff_smooth = [], []                    # left_cheek - right_cheek
eye_diff_raw,   eye_diff_smooth   = [], []                    # left_eye   - right_eye

# ─────────────────────────────────────────────────────────────────────────────
# LIVE PLOT SETUP
# ─────────────────────────────────────────────────────────────────────────────

if LIVE_PLOT:
    plt.ion()
    fig, (ax_rois, ax_diff) = plt.subplots(2, 1, figsize=(12, 8))

    lines_roi = {}
    for name in ROI_NAMES:
        (line,) = ax_rois.plot([], [], label=name, linewidth=1.3)
        lines_roi[name] = line
    ax_rois.set_title(f"ROI Temperatures — SMA window {SMA_WINDOW_SEC:.1f}s ({SMA_WINDOW} frames)")
    ax_rois.set_xlabel("Time (s)")
    ax_rois.set_ylabel("Temperature (°C)")
    ax_rois.legend(loc='upper right', fontsize=8, ncol=2)

    (line_cheek_diff,) = ax_diff.plot([], [], label='Left − Right Cheek Δ', color='tab:red', linewidth=1.6)
    (line_eye_diff,)   = ax_diff.plot([], [], label='Left − Right Inner-Eye Δ', color='tab:blue', linewidth=1.6)
    ax_diff.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_diff.set_title("Temperature Difference — SMA smoothed")
    ax_diff.set_xlabel("Time (s)")
    ax_diff.set_ylabel("Δ Temperature (°C)")
    ax_diff.legend(loc='upper right', fontsize=8)

    plt.tight_layout()


def update_live_plot():
    for name in ROI_NAMES:
        lines_roi[name].set_data(times, roi_data[name]['smooth'])
    ax_rois.relim()
    ax_rois.autoscale_view()

    line_cheek_diff.set_data(times, cheek_diff_smooth)
    line_eye_diff.set_data(times, eye_diff_smooth)
    ax_diff.relim()
    ax_diff.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    t = frame_idx / fps

    plain_grey       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # used for actual pixel/temperature extraction
    transformed_grey = ut.get_transformed_image(plain_grey)          # used ONLY for landmark detection + overlay

    rgb_for_detection = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2RGB)
    results = face_mesh.process(rgb_for_detection)

    display_frame = transformed_grey.copy()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]  # max_num_faces=1

        rois, display_frame = extract_all_rois(display_frame, face_landmarks)

        times.append(t)

        for name in ROI_NAMES:
            pixel_val = get_roi_mean_pixel(plain_grey, rois[name])   # <- plain grey, not sharpened
            temp_val = to_temp(pixel_val)
            roi_data[name]['raw'].append(temp_val)
            roi_data[name]['smooth'].append(sma(roi_data[name]['raw'], SMA_WINDOW))

        left_cheek_temp  = roi_data['left_cheek']['raw'][-1]
        right_cheek_temp = roi_data['right_cheek']['raw'][-1]
        left_eye_temp    = roi_data['left_eye']['raw'][-1]
        right_eye_temp   = roi_data['right_eye']['raw'][-1]

        cheek_diff_raw.append(left_cheek_temp - right_cheek_temp)
        eye_diff_raw.append(left_eye_temp - right_eye_temp)

        cheek_diff_smooth.append(sma(cheek_diff_raw, SMA_WINDOW))
        eye_diff_smooth.append(sma(eye_diff_raw, SMA_WINDOW))

        if LIVE_PLOT and frame_idx % PLOT_UPDATE_EVERY_N_FRAMES == 0:
            update_live_plot()

    cv2.imshow("ROI Overlay (sharpened, detection view)", display_frame)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# ─────────────────────────────────────────────────────────────────────────────
# END-OF-RUN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_diff_summary(label, diffs):
    diffs = np.array(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        print(f"  {label}: no valid data")
        return
    print(f"  {label}:")
    print(f"      Mean : {np.mean(diffs):+.3f} °C")
    print(f"      Min  : {np.min(diffs):+.3f} °C")
    print(f"      Max  : {np.max(diffs):+.3f} °C")
    print(f"      Final: {diffs[-1]:+.3f} °C")


print("\n" + "=" * 60)
print("TEMPERATURE DIFFERENCE SUMMARY (°C)")
print("=" * 60)
print_diff_summary("Left Cheek − Right Cheek", cheek_diff_raw)
print_diff_summary("Left Inner-Eye − Right Inner-Eye", eye_diff_raw)
print("=" * 60)

if LIVE_PLOT:
    update_live_plot()
    plt.ioff()
    plt.show()