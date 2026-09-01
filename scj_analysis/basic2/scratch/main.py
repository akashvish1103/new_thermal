# Using Logic of breathing_box.py


# Using this as a Driver of the Utility of Scratch
# this file using all the ROI from utility code
# + EMA landmark smoothing
# + pixel -> temperature mapping per ROI
# + per-frame statistical logging to CSV

import os
import csv
import numpy as np
import mediapipe as mp
import cv2
import utilities as ut
import utility_linear_mapping as ulm


# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40.wmv"
# video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\4.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\44_2026-07-07\01_Passive_Profiling\44_Passive_Thermal_25_40.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Exported_Thermal_IRSoft_Sabarmati\48_Passive_Thermal_30_40.mpg"

LOG_PATH = os.path.splitext(video_path)[0] + "_roi_log.csv"

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

UPPER_LIPS_LANDMARK       = [13, 206, 426]
NOSE_HORIZONTAL_LANDMARKS = [64, 278]
NOSE_VERTICAL_LANDMARKS   = [4, 94]
NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK


# ============================================================
# EMA Landmark Smoother (from previous step)
# ============================================================

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


# ============================================================
# ROI pixel extraction helpers
# ============================================================
# Two shapes are supported:
#   ("box", top_left, bottom_right)   -> axis-aligned rectangle
#   ("polygon", [pt1, pt2, ...])      -> arbitrary polygon (masked)
#
# Confirmed against utilities.py:
#   - breathing ROI              -> box:      (top_left, bottom_right)
#   - nose ROI                   -> box:      (top_left, bottom_right)
#   - forehead ROI                -> polygon:  polygon_points (9 landmark points)
#   - cheek_L / cheek_R           -> polygon:  7 landmark points each
#   - eye_L (left inner-eye box)  -> box:      (top_left_coords, bottom_right_coords)
#   - eye_R (right inner-eye box) -> box:      (top_right_coords, bottom_left_coords)
#
# IMPORTANT: coordinates are computed using transformed_grey (CLAHE/gamma/sharpen
# enhanced -- better for MediaPipe detection), but since get_transformed_image()
# only alters pixel intensities (no resize/rotate/crop), the coordinates line up
# 1:1 with the ORIGINAL grey frame. All pixel/temperature STATS are therefore
# extracted from `grey` (the raw, unenhanced frame) -- not transformed_grey --
# because the temperature calibration (legend readout) was built against raw
# pixel intensities, and CLAHE/gamma/sharpen would silently invalidate it.

def extract_pixels(image, shape_type, geometry):
    """Returns a 1D numpy array of pixel intensities inside the ROI."""
    h, w = image.shape[:2]

    if shape_type == "box":
        top_left, bottom_right = geometry
        x1, y1 = int(top_left[0]), int(top_left[1])
        x2, y2 = int(bottom_right[0]), int(bottom_right[1])
        x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
        y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
        if x2 <= x1 or y2 <= y1:
            return np.array([], dtype=image.dtype)
        return image[y1:y2, x1:x2].flatten()

    elif shape_type == "polygon":
        pts = np.array([[int(p[0]), int(p[1])] for p in geometry], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return image[mask == 255].flatten()

    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")


def pixel_and_temp_stats(pixel_array, min_temp, max_temp, first_pixel, last_pixel):
    """Given raw pixel intensities for one ROI, return a dict of pixel-space
    and temperature-space statistics. Returns NaNs if the ROI is empty."""
    if pixel_array.size == 0:
        nan_stats = dict(count=0, mean_px=np.nan, std_px=np.nan, min_px=np.nan,
                          max_px=np.nan, median_px=np.nan, mean_temp=np.nan,
                          std_temp=np.nan, min_temp=np.nan, max_temp=np.nan,
                          median_temp=np.nan)
        return nan_stats

    px = pixel_array.astype(np.float64)
    temp = ulm.map_pixel_to_temperature(px, min_temp, max_temp, first_pixel, last_pixel)

    return dict(
        count=int(px.size),
        mean_px=float(np.mean(px)),
        std_px=float(np.std(px)),
        min_px=float(np.min(px)),
        max_px=float(np.max(px)),
        median_px=float(np.median(px)),
        mean_temp=float(np.mean(temp)),
        std_temp=float(np.std(temp)),
        min_temp=float(np.min(temp)),
        max_temp=float(np.max(temp)),
        median_temp=float(np.median(temp)),
    )


ROI_NAMES = ["breathing", "forehead", "cheek_L", "cheek_R", "eye_L", "eye_R", "nose"]

STAT_SUFFIXES = ["count", "mean_px", "std_px", "min_px", "max_px", "median_px",
                  "mean_temp", "std_temp", "min_temp", "max_temp", "median_temp",
                  "delta_temp_from_baseline"]

CSV_FIELDNAMES = ["frame_number", "time_sec", "face_detected"] + [
    f"{roi}_{suffix}" for roi in ROI_NAMES for suffix in STAT_SUFFIXES
]


# ============================================================
# Calibration (pixel <-> temperature), computed ONCE per video
# ============================================================

pixel_column = ulm.get_one_pixel_column(video_path)
min_temp, max_temp, first_pixel, last_pixel = \
    ulm.get_first_last_element_from_OnePixelColumn_AND_min_max_from_filename(
        pixel_column, video_path
    )

# per-ROI baseline mean temp, captured on first frame a face is detected
roi_baseline_temp = {roi: None for roi in ROI_NAMES}


# ============================================================
# Main loop
# ============================================================

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0  # fallback if the container doesn't report fps correctly
    print(f"WARNING: could not read FPS from video, defaulting to {fps}")

frame_number = -1
got_frame = None

log_file = open(LOG_PATH, "w", newline="")
csv_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDNAMES)
csv_writer.writeheader()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    time_sec = frame_number / fps

    row = {"frame_number": frame_number, "time_sec": round(time_sec, 3), "face_detected": False}
    # pre-fill all ROI columns as blank in case no face is found this frame
    for roi in ROI_NAMES:
        for suffix in STAT_SUFFIXES:
            row[f"{roi}_{suffix}"] = ""

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformed_grey = ut.get_transformed_image(grey)

    rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2BGR)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        row["face_detected"] = True

        for face_landmarks in results.multi_face_landmarks:

            face_landmarks = landmark_smoother.smooth(face_landmarks)

            # Coordinates are computed on transformed_grey (better for MediaPipe),
            # but stats are pulled from `grey` (raw, uncalibrated-safe) below.
            top_left_cords, bottom_right_cords, got_frame = ut.get_breathing_roi_cords(transformed_grey, face_landmarks)
            polygon_points, _mean_pixel_unused, got_frame = ut.get_forhead_poly_coords(transformed_grey, face_landmarks)
            cheek_left_pts, cheek_right_pts, got_frame = ut.get_cheeks_coordinates(transformed_grey, face_landmarks, [], [])
            (eye_tl, eye_br, eye_tr, eye_bl, got_frame) = ut.get_eyes_coordinates(transformed_grey, face_landmarks)
            nose_tl, nose_br, got_frame = ut.get_nose_tip_coordinates(transformed_grey, face_landmarks)

            roi_geometry = {
                "breathing": ("box", (top_left_cords, bottom_right_cords)),
                "forehead":  ("polygon", polygon_points),
                "cheek_L":   ("polygon", cheek_left_pts),   # 7-point polygon
                "cheek_R":   ("polygon", cheek_right_pts),  # 7-point polygon
                "eye_L":     ("box", (eye_tl, eye_br)),     # left inner-eye box
                "eye_R":     ("box", (eye_tr, eye_bl)),     # right inner-eye box
                "nose":      ("box", (nose_tl, nose_br)),
            }

            for roi_name, (shape_type, geometry) in roi_geometry.items():
                # NOTE: stats computed from `grey` (raw frame), NOT transformed_grey --
                # CLAHE/gamma/sharpen would invalidate the temperature calibration.
                pixels = extract_pixels(grey, shape_type, geometry)
                stats = pixel_and_temp_stats(pixels, min_temp, max_temp, first_pixel, last_pixel)

                if roi_baseline_temp[roi_name] is None and not np.isnan(stats["mean_temp"]):
                    roi_baseline_temp[roi_name] = stats["mean_temp"]

                baseline = roi_baseline_temp[roi_name]
                delta = (stats["mean_temp"] - baseline) if (baseline is not None and not np.isnan(stats["mean_temp"])) else np.nan

                for suffix in STAT_SUFFIXES:
                    if suffix == "delta_temp_from_baseline":
                        row[f"{roi_name}_{suffix}"] = "" if np.isnan(delta) else round(delta, 3)
                    else:
                        val = stats[suffix]
                        row[f"{roi_name}_{suffix}"] = "" if (isinstance(val, float) and np.isnan(val)) else (round(val, 3) if isinstance(val, float) else val)

            # console preview (kept from your original prints, trimmed)
            print(f"frame {frame_number} | t={time_sec:.2f}s | "
                  f"forehead_mean_temp={row['forehead_mean_temp']} | "
                  f"nose_mean_temp={row['nose_mean_temp']}")
    else:
        # face lost this frame -> reset EMA so it doesn't blend garbage
        # history into the next detection
        landmark_smoother.reset()

    csv_writer.writerow(row)

    if got_frame is not None:
        cv2.imshow("Transformed Grey", got_frame)
    cv2.imshow("RGB frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()

print(f"\nDone. ROI log saved to: {LOG_PATH}")