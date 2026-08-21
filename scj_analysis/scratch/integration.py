# ============================================================
# main_pipeline.py
#
# Full pipeline:
#   1) Set ONE video path.
#   2) Auto-calibrate min/max temperature (from the video's legend +
#      filename) using utility_linear_mapping.py.
#   3) Run MediaPipe FaceMesh frame-by-frame.
#   4) Extract ROIs (forehead, cheeks, nose, breathing box, eyes)
#      using utilities.py.
#   5) Convert each ROI's mean pixel intensity -> mean temperature (°C).
#   6) Print per-frame results to console AND save them to a CSV.
#
# Place this file in the same folder as utilities.py and
# utility_linear_mapping.py (no changes were made to either of those
# two files).
# ============================================================

import os
import csv
import cv2
import numpy as np
import mediapipe as mp

import utilities as ut
import utility_linear_mapping as tm


# ============================================================
# 1) SET VIDEO PATH — the only thing you change per run
# ============================================================
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"


# ============================================================
# 2) CALIBRATE ONCE — read the color legend on the first frame
#    + parse min/max temperature from the filename.
#    (Uses utility_linear_mapping.py exactly as-is.)
# ============================================================
pixel_column = tm.get_one_pixel_column(video_path)

min_temp, max_temp, first_pixel, last_pixel = \
    tm.get_first_last_element_from_OnePixelColumn_AND_min_max_from_filename(
        pixel_column, video_path
    )

print(f"[CALIBRATION] min={min_temp}°C  max={max_temp}°C  "
      f"first_pixel={first_pixel}  last_pixel={last_pixel}")


def px_to_temp(pixel_value):
    """Maps a grayscale pixel intensity -> temperature (°C) using this video's calibration."""
    if pixel_value is None:
        return None
    return tm.map_pixel_to_temperature(
        pixel_value, min_temp, max_temp, first_pixel, last_pixel
    )


# ============================================================
# 3) HELPERS — mean pixel intensity for ROIs that utilities.py
#    returns coordinates for, but not a mean value directly.
#    (utilities.py itself is untouched — this logic lives here.)
# ============================================================

def mean_from_box(frame, top_left, bottom_right):
    """Crops a rectangular ROI (clamped to frame bounds) and returns its mean pixel intensity."""
    h, w = frame.shape[:2]
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return float(np.mean(crop))


def mean_from_polygon(frame, points):
    """Masks a polygon ROI (list of (x, y) points) and returns its mean pixel intensity."""
    if not points:
        return None
    polygon = np.array(points, dtype=np.int32)
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return float(cv2.mean(frame, mask=mask)[0])


# ============================================================
# 4) MEDIAPIPE FACE MESH SETUP
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
# 5) CSV SETUP
# ============================================================
csv_path = os.path.splitext(video_path)[0] + "_roi_temperatures.csv"
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_no",
    "forehead_temp_C",
    "left_cheek_temp_C",
    "right_cheek_temp_C",
    "nose_temp_C",
    "breathing_box_temp_C",
    "left_eye_temp_C",
    "right_eye_temp_C",
])


# ============================================================
# 6) MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(video_path)
frame_no = 0
got_frame = None   # holds the last annotated frame for display

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformed_grey = ut.get_transformed_image(grey)

    rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2BGR)
    results = face_mesh.process(rgb)

    # Default row (all None) in case no face is found this frame
    row = [frame_no, None, None, None, None, None, None, None]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # ---------------- Forehead (polygon) ----------------
            polygon_points, forehead_mean_px, got_frame = ut.get_forhead_poly_coords(
                transformed_grey, face_landmarks
            )
            forehead_temp = px_to_temp(forehead_mean_px)

            # ---------------- Cheeks (left + right) ----------------
            points_left, points_right, got_frame = ut.get_cheeks_coordinates(
                transformed_grey, face_landmarks, [], []
            )
            left_cheek_mean_px = mean_from_polygon(transformed_grey, points_left)
            right_cheek_mean_px = mean_from_polygon(transformed_grey, points_right)
            left_cheek_temp = px_to_temp(left_cheek_mean_px)
            right_cheek_temp = px_to_temp(right_cheek_mean_px)

            # ---------------- Nose tip ----------------
            nose_tl, nose_br, got_frame = ut.get_nose_tip_coordinates(
                transformed_grey, face_landmarks
            )
            nose_mean_px = mean_from_box(transformed_grey, nose_tl, nose_br)
            nose_temp = px_to_temp(nose_mean_px)

            # ---------------- Breathing box (nose/upper-lip) ----------------
            breath_tl, breath_br, got_frame = ut.get_breathing_roi_cords(
                transformed_grey, face_landmarks
            )
            breath_mean_px = mean_from_box(transformed_grey, breath_tl, breath_br)
            breath_temp = px_to_temp(breath_mean_px)

            # ---------------- Eyes (inner canthus, left + right) ----------------
            (eye_tl, eye_br, eye_tr, eye_bl, got_frame) = ut.get_eyes_coordinates(
                transformed_grey, face_landmarks
            )
            left_eye_mean_px = mean_from_box(transformed_grey, eye_tl, eye_br)
            right_eye_mean_px = mean_from_box(transformed_grey, eye_tr, eye_bl)
            left_eye_temp = px_to_temp(left_eye_mean_px)
            right_eye_temp = px_to_temp(right_eye_mean_px)

            row = [
                frame_no,
                round(forehead_temp, 2) if forehead_temp is not None else None,
                round(left_cheek_temp, 2) if left_cheek_temp is not None else None,
                round(right_cheek_temp, 2) if right_cheek_temp is not None else None,
                round(nose_temp, 2) if nose_temp is not None else None,
                round(breath_temp, 2) if breath_temp is not None else None,
                round(left_eye_temp, 2) if left_eye_temp is not None else None,
                round(right_eye_temp, 2) if right_eye_temp is not None else None,
            ]

            print(
                f"[Frame {frame_no}] "
                f"Forehead={row[1]}°C  L-Cheek={row[2]}°C  R-Cheek={row[3]}°C  "
                f"Nose={row[4]}°C  Breathing={row[5]}°C  "
                f"L-Eye={row[6]}°C  R-Eye={row[7]}°C"
            )

    csv_writer.writerow(row)

    # ---------------- Display ----------------
    if got_frame is not None:
        cv2.imshow("Transformed Grey (annotated)", got_frame)
    cv2.imshow("RGB frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print(f"\nDone. Per-frame ROI temperatures saved to:\n{csv_path}")