import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from forehead import utilities as ut

# ============================================================
# OUTPUT CSV PATH  — CHANGE THIS
# ============================================================

SUBJECT_NAME = "aditi2"                                         # CHANGE: subject identifier
OUTPUT_DIR   = r"C:\Users\Akash Vishwakarma\Desktop\new_thermal\data"   # CHANGE: destination folder
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_roi_temperatures.csv")

# ============================================================
# PIXEL TO TEMPERATURE CONVERSION
# ============================================================

def pixel_to_temperature(pixel_value):
    temperature = (
        0.05891454 * pixel_value
        + 30.07676744
    )
    return temperature

# ============================================================
# MEDIAPIPE FACE MESH SETUP
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ============================================================
# VARIABLES
# ============================================================

flag = False

# ============================================================
# INPUT VIDEO
# ============================================================

# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Swamini\swamini_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Girish\girish_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"   # CHANGE: input video path

video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\aditi_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\pooja_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\purva_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\priyank_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\pratham_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\sneha_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"

# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"


cap = cv2.VideoCapture(video_path)

# ============================================================
# PER-FRAME LOG  (list of dicts → DataFrame at end)
# ============================================================

frame_log = []

# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    points_left  = []
    points_right = []

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Row template — NaN means ROI was not detected this frame
    # --------------------------------------------------------
    row = {
        "frame":             frame_number,
        "left_eye_temp":     np.nan,
        "right_eye_temp":    np.nan,
        "forehead_temp":     np.nan,
        "nose_temp":         np.nan,
        "left_cheek_temp":   np.nan,
        "right_cheek_temp":  np.nan,
    }

    # ========================================================
    # CONVERT FRAME TO GRAYSCALE
    # ========================================================

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w, _ = frame.shape

    # ========================================================
    # ENHANCE FRAME FOR MEDIAPIPE
    # ========================================================

    sharpened_grey_frame = ut.get_transformed_image(grey)

    # ========================================================
    # CONVERT TO BGR FOR MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(sharpened_grey_frame, cv2.COLOR_GRAY2BGR)

    # ========================================================
    # FACEMESH DETECTION
    # ========================================================

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # =================================================
            # GET ROI COORDINATES
            # =================================================

            (
                top_left_coords,
                bottom_right_coords,
                top_right_coords,
                bottom_left_coords

            ) = ut.get_eyes_coordinates(
                frame, grey, face_landmarks
            )

            (
                a, b, c, d

            ) = ut.get_forehead_coordinates(
                frame, face_landmarks, flag
            )

            (
                x, y, z, w1

            ) = ut.get_nose_coordinates(
                frame, face_landmarks
            )

            (
                pl, pr

            ) = ut.get_cheeks_coordinates(
                frame, face_landmarks, points_left, points_right
            )

            # =================================================
            # LEFT EYE  — top 10% hottest pixels
            # =================================================

            left_eye_roi = grey[
                top_left_coords[1]:bottom_right_coords[1],
                top_left_coords[0]:bottom_right_coords[0]
            ]

            if left_eye_roi.size > 0:
                pixel_percentage      = 0.1
                flat_sorted           = np.sort(left_eye_roi.flatten())[::-1]
                top_pixels            = flat_sorted[:max(1, int(len(flat_sorted) * pixel_percentage))]
                row["left_eye_temp"]  = pixel_to_temperature(np.mean(top_pixels))

            # =================================================
            # RIGHT EYE  — top 10% hottest pixels
            # =================================================

            right_eye_roi = grey[
                top_right_coords[1]:bottom_left_coords[1],
                bottom_left_coords[0]:top_right_coords[0]
            ]

            if right_eye_roi.size > 0:
                pixel_percentage      = 0.1
                flat_sorted           = np.sort(right_eye_roi.flatten())[::-1]
                top_pixels            = flat_sorted[:max(1, int(len(flat_sorted) * pixel_percentage))]
                row["right_eye_temp"] = pixel_to_temperature(np.mean(top_pixels))

            # =================================================
            # FOREHEAD  — full mean
            # =================================================

            forehead_roi = grey[b:d, a:c]

            if forehead_roi.size > 0:
                row["forehead_temp"] = pixel_to_temperature(np.mean(forehead_roi))

            # =================================================
            # NOSE  — full mean
            # =================================================

            nose_roi = grey[y:w1, x:z]

            if nose_roi.size > 0:
                row["nose_temp"] = pixel_to_temperature(np.mean(nose_roi))

            # =================================================
            # LEFT CHEEK  — polygon masked mean
            # =================================================

            polygon_left = np.array(pl, dtype=np.int32)
            mask_left    = np.zeros(grey.shape, dtype=np.uint8)
            cv2.fillPoly(mask_left, [polygon_left], 255)
            left_pixels  = grey[mask_left == 255]

            if left_pixels.size > 0:
                row["left_cheek_temp"] = pixel_to_temperature(np.mean(left_pixels))

            # =================================================
            # RIGHT CHEEK  — polygon masked mean
            # =================================================

            polygon_right = np.array(pr, dtype=np.int32)
            mask_right    = np.zeros(grey.shape, dtype=np.uint8)
            cv2.fillPoly(mask_right, [polygon_right], 255)
            right_pixels  = grey[mask_right == 255]

            if right_pixels.size > 0:
                row["right_cheek_temp"] = pixel_to_temperature(np.mean(right_pixels))

    # --------------------------------------------------------
    # Append row (detected or NaN) and increment frame counter
    # --------------------------------------------------------
    frame_log.append(row)
    frame_number += 1

    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    cv2.imshow("Thermal ROI Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================================
# RELEASE RESOURCES
# ============================================================

cap.release()
cv2.destroyAllWindows()

# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.DataFrame(frame_log)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ CSV saved  →  {OUTPUT_CSV}")
print(f"   Frames logged : {len(df)}")
print(f"   Columns       : {list(df.columns)}")
print(f"\nPer-ROI detection rate (non-NaN frames):")
for col in df.columns[1:]:
    rate = df[col].notna().sum()
    print(f"   {col:<22} {rate} / {len(df)} frames  ({100*rate/len(df):.1f}%)")

# ============================================================
# PLOT TEMPERATURE SIGNALS
# ============================================================

plt.figure(figsize=(18, 8))

plt.plot(df["frame"], df["left_eye_temp"],    label="Left Eye")
plt.plot(df["frame"], df["right_eye_temp"],   label="Right Eye")
plt.plot(df["frame"], df["forehead_temp"],    label="Forehead")
plt.plot(df["frame"], df["nose_temp"],        label="Nose")
plt.plot(df["frame"], df["left_cheek_temp"],  label="Left Cheek")
plt.plot(df["frame"], df["right_cheek_temp"], label="Right Cheek")

plt.xlabel("Frame Number")
plt.ylabel("Temperature (°C)")
plt.title(f"Facial ROI Temperature Signals — {SUBJECT_NAME}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()