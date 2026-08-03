# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt

# # ============================================================
# #  CONFIGURATION  — tweak these values as needed
# # ============================================================
# PERCENTAGE_PIXEL_TO_KEEP = 0.5  # e.g. 0.10 = hottest 10 %

# # Padding (in pixels) added around the bounding box of each eye's landmarks
# PAD_TOP    = 15   # extra pixels above the eye
# PAD_BOTTOM = 15   # extra pixels below the eye
# PAD_LEFT   = 15   # extra pixels to the left  of the eye
# PAD_RIGHT  = 15   # extra pixels to the right of the eye

# # Input video
# # video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# # ============================================================

# # ------------------------------------------------------------------
# # MediaPipe landmark indices that trace the LEFT and RIGHT eye rims
# # (these 16 points per eye give a tight outline of the whole eye)
# # ------------------------------------------------------------------
# #  Left eye  (from the subject's perspective → right side of image)
# LEFT_EYE_LANDMARKS = [
#     33, 7, 163, 144, 145, 153, 154, 155,
#     133, 173, 157, 158, 159, 160, 161, 246
# ]
# # Right eye  (from the subject's perspective → left side of image)
# RIGHT_EYE_LANDMARKS = [
#     362, 382, 381, 380, 374, 373, 390, 249,
#     263, 466, 388, 387, 386, 385, 384, 398
# ]

# # -----------------------------
# # MediaPipe Face Mesh Setup
# # -----------------------------
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True
# )

# left_mean_values  = []
# right_mean_values = []

# cap = cv2.VideoCapture(video_path)

# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps    = cap.get(cv2.CAP_PROP_FPS)


# def get_eye_bbox(landmarks, indices, img_w, img_h,
#                  pad_top, pad_bottom, pad_left, pad_right):
#     """
#     Given a list of MediaPipe landmark indices for one eye,
#     return a padded bounding box (x1, y1, x2, y2) clamped to image bounds.
#     """
#     xs = [int(landmarks[i].x * img_w) for i in indices]
#     ys = [int(landmarks[i].y * img_h) for i in indices]

#     x1 = max(0,     min(xs) - pad_left)
#     y1 = max(0,     min(ys) - pad_top)
#     x2 = min(img_w, max(xs) + pad_right)
#     y2 = min(img_h, max(ys) + pad_bottom)

#     return x1, y1, x2, y2


# # -----------------------------
# # Process Video
# # -----------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = face_mesh.process(rgb)

#     filt_frame = np.zeros_like(grey)   # filtered visualisation frame

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             lms = face_landmarks.landmark

#             # ── Bounding boxes ──────────────────────────────────────────────
#             lx1, ly1, lx2, ly2 = get_eye_bbox(
#                 lms, LEFT_EYE_LANDMARKS,  w, h,
#                 PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT
#             )
#             rx1, ry1, rx2, ry2 = get_eye_bbox(
#                 lms, RIGHT_EYE_LANDMARKS, w, h,
#                 PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT
#             )

#             # ── Draw rectangles on the display frame ────────────────────────
#             cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)   # blue – left
#             cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)   # red  – right

#             cv2.putText(frame, "L", (lx1, ly1 - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#             cv2.putText(frame, "R", (rx1, ry1 - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#             # ── Extract grey ROIs ────────────────────────────────────────────
#             box_left  = grey[ly1:ly2, lx1:lx2]
#             box_right = grey[ry1:ry2, rx1:rx2]

#             # ── Skip if ROI is empty (edge-of-frame situations) ──────────────
#             if box_left.size == 0 or box_right.size == 0:
#                 continue

#             # ── Hottest N % pixels ───────────────────────────────────────────
#             left_flat  = box_left.flatten()
#             right_flat = box_right.flatten()

#             left_sorted  = np.sort(left_flat)[::-1]
#             right_sorted = np.sort(right_flat)[::-1]

#             n_left  = max(1, int(len(left_sorted)  * PERCENTAGE_PIXEL_TO_KEEP))
#             n_right = max(1, int(len(right_sorted) * PERCENTAGE_PIXEL_TO_KEEP))

#             left_hot  = left_sorted[:n_left]
#             right_hot = right_sorted[:n_right]

#             left_mean  = left_hot.mean()
#             right_mean = right_hot.mean()

#             left_mean_values.append(left_mean)
#             right_mean_values.append(right_mean)

#             print(f"Left mean: {left_mean:.2f}   Right mean: {right_mean:.2f}")

#             # ── Filtered visualisation (only hottest pixels shown) ───────────
#             left_threshold  = np.percentile(box_left,  (1 - PERCENTAGE_PIXEL_TO_KEEP) * 100)
#             right_threshold = np.percentile(box_right, (1 - PERCENTAGE_PIXEL_TO_KEEP) * 100)

#             left_hot_mask  = box_left  >= left_threshold
#             right_hot_mask = box_right >= right_threshold

#             filt_frame[ly1:ly2, lx1:lx2][left_hot_mask]  = box_left[left_hot_mask]
#             filt_frame[ry1:ry2, rx1:rx2][right_hot_mask] = box_right[right_hot_mask]

#             # Also draw rectangle outlines on filt_frame for reference
#             cv2.rectangle(filt_frame, (lx1, ly1), (lx2, ly2), 200, 1)
#             cv2.rectangle(filt_frame, (rx1, ry1), (rx2, ry2), 200, 1)

#     # ── Display ──────────────────────────────────────────────────────────────
#     cv2.imshow("Video – Eye ROI", frame)

#     colored = cv2.applyColorMap(filt_frame, cv2.COLORMAP_JET)
#     cv2.imshow("Filtered Frame (hottest pixels)", colored)

#     if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
#         break

# # -----------------------------
# # Release Resources
# # -----------------------------
# cap.release()
# cv2.destroyAllWindows()

# # -----------------------------
# # Plot
# # -----------------------------
# plt.figure(figsize=(20, 3))
# plt.plot(left_mean_values,  label="Left Eye")
# plt.plot(right_mean_values, label="Right Eye")
# plt.xlabel("Frame Number")
# plt.ylabel(f"Mean Intensity (hottest {int(PERCENTAGE_PIXEL_TO_KEEP*100)}%)")
# plt.title("Peri-Orbital Thermal Signal – Full Eye ROI")
# plt.legend()
# plt.tight_layout()
# plt.show()

#############################################################################################################
# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt

# # ============================================================
# #  CONFIGURATION  — tweak these values as needed
# # ============================================================
# PERCENTAGE_PIXEL_TO_KEEP = 0.10   # e.g. 0.10 = hottest 10%

# # Padding around each eye bounding box (pixels)
# PAD_TOP    = 10
# PAD_BOTTOM = 10
# PAD_LEFT   = 10
# PAD_RIGHT  = 10

# # Padding around face bounding box (pixels)
# FACE_PAD = 20

# # Input video
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# # ============================================================

# # ------------------------------------------------------------------
# # MediaPipe landmark indices for LEFT and RIGHT eye rims (16 pts each)
# # ------------------------------------------------------------------
# LEFT_EYE_LANDMARKS = [
#     33, 7, 163, 144, 145, 153, 154, 155,
#     133, 173, 157, 158, 159, 160, 161, 246
# ]
# RIGHT_EYE_LANDMARKS = [
#     362, 382, 381, 380, 374, 373, 390, 249,
#     263, 466, 388, 387, 386, 385, 384, 398
# ]

# # All 468 face mesh landmarks → used for face bounding box
# ALL_FACE_LANDMARKS = list(range(468))

# # -----------------------------
# # MediaPipe Face Mesh Setup
# # -----------------------------
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True
# )

# left_mean_values  = []
# right_mean_values = []

# cap = cv2.VideoCapture(video_path)
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps    = cap.get(cv2.CAP_PROP_FPS)


# def get_bbox(landmarks, indices, img_w, img_h,
#              pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):
#     """
#     Compute a padded bounding box from a set of landmark indices.
#     Returns (x1, y1, x2, y2) clamped to image bounds.
#     """
#     xs = [int(landmarks[i].x * img_w) for i in indices]
#     ys = [int(landmarks[i].y * img_h) for i in indices]
#     x1 = max(0,     min(xs) - pad_left)
#     y1 = max(0,     min(ys) - pad_top)
#     x2 = min(img_w, max(xs) + pad_right)
#     y2 = min(img_h, max(ys) + pad_bottom)
#     return x1, y1, x2, y2


# # -----------------------------
# # Process Video
# # -----------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = face_mesh.process(rgb)

#     # Black canvas — only hottest pixels will be written here
#     filt_frame = np.zeros((h, w), dtype=np.uint8)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             lms = face_landmarks.landmark

#             # ── 1. FACE bounding box ─────────────────────────────────────────
#             fx1, fy1, fx2, fy2 = get_bbox(
#                 lms, ALL_FACE_LANDMARKS, w, h,
#                 pad_top=FACE_PAD, pad_bottom=FACE_PAD,
#                 pad_left=FACE_PAD, pad_right=FACE_PAD
#             )
#             # Green rectangle around whole face
#             cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#             cv2.putText(frame, "Face", (fx1, fy1 - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#             # ── 2. EYE bounding boxes ────────────────────────────────────────
#             lx1, ly1, lx2, ly2 = get_bbox(
#                 lms, LEFT_EYE_LANDMARKS, w, h,
#                 PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT
#             )
#             rx1, ry1, rx2, ry2 = get_bbox(
#                 lms, RIGHT_EYE_LANDMARKS, w, h,
#                 PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT
#             )

#             # Blue = left eye,  Red = right eye
#             cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
#             cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
#             cv2.putText(frame, "L", (lx1, ly1 - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#             cv2.putText(frame, "R", (rx1, ry1 - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#             # ── 3. Extract grey ROIs ─────────────────────────────────────────
#             box_left  = grey[ly1:ly2, lx1:lx2]
#             box_right = grey[ry1:ry2, rx1:rx2]

#             if box_left.size == 0 or box_right.size == 0:
#                 continue

#             # ── 4. Hottest N% pixels → mean for graph ────────────────────────
#             left_flat  = box_left.flatten()
#             right_flat = box_right.flatten()

#             n_left  = max(1, int(len(left_flat)  * PERCENTAGE_PIXEL_TO_KEEP))
#             n_right = max(1, int(len(right_flat) * PERCENTAGE_PIXEL_TO_KEEP))

#             left_hot_vals  = np.sort(left_flat)[::-1][:n_left]
#             right_hot_vals = np.sort(right_flat)[::-1][:n_right]

#             left_mean  = left_hot_vals.mean()
#             right_mean = right_hot_vals.mean()

#             left_mean_values.append(left_mean)
#             right_mean_values.append(right_mean)

#             print(f"Left mean: {left_mean:.2f}   Right mean: {right_mean:.2f}")

#             # ── 5. Build filt_frame: only hottest pixels glow ────────────────
#             #  percentile threshold = (1 - keep%) * 100
#             #  e.g. keep 10%  →  90th percentile threshold
#             left_thresh  = np.percentile(box_left,  (1 - PERCENTAGE_PIXEL_TO_KEEP) * 100)
#             right_thresh = np.percentile(box_right, (1 - PERCENTAGE_PIXEL_TO_KEEP) * 100)

#             left_mask  = box_left  >= left_thresh
#             right_mask = box_right >= right_thresh

#             # Write ONLY those pixels onto the black canvas (rest stays 0)
#             left_roi_out  = filt_frame[ly1:ly2, lx1:lx2]
#             right_roi_out = filt_frame[ry1:ry2, rx1:rx2]

#             left_roi_out[left_mask]   = box_left[left_mask]
#             right_roi_out[right_mask] = box_right[right_mask]

#             filt_frame[ly1:ly2, lx1:lx2] = left_roi_out
#             filt_frame[ry1:ry2, rx1:rx2] = right_roi_out

#             # Draw eye box outlines on filt_frame too (dim white)
#             cv2.rectangle(filt_frame, (lx1, ly1), (lx2, ly2), 80, 1)
#             cv2.rectangle(filt_frame, (rx1, ry1), (rx2, ry2), 80, 1)

#     # ── Display ──────────────────────────────────────────────────────────────
#     # Window 1 — original frame with face + eye rectangles
#     cv2.imshow("Video – Face & Eye ROI", frame)

#     # Window 2 — colormap: black background, only hottest eye pixels glow
#     colored = cv2.applyColorMap(filt_frame, cv2.COLORMAP_JET)

#     # Make truly-zero pixels (background) stay black, not deep-blue from JET
#     bg_mask = filt_frame == 0
#     colored[bg_mask] = [0, 0, 0]

#     cv2.imshow(f"Hottest {int(PERCENTAGE_PIXEL_TO_KEEP*100)}% Eye Pixels", colored)

#     if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
#         break

# # -----------------------------
# # Release Resources
# # -----------------------------
# cap.release()
# cv2.destroyAllWindows()

# # -----------------------------
# # Plot
# # -----------------------------
# plt.figure(figsize=(20, 3))
# plt.plot(left_mean_values,  label="Left Eye")
# plt.plot(right_mean_values, label="Right Eye")
# plt.xlabel("Frame Number")
# plt.ylabel(f"Mean Intensity (hottest {int(PERCENTAGE_PIXEL_TO_KEEP*100)}%)")
# plt.title("Peri-Orbital Thermal Signal – Full Eye ROI")
# plt.legend()
# plt.tight_layout()
# plt.show()
###########################################################################################


import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  CONFIGURATION  — tweak these values as needed
# ============================================================
PERCENTAGE_PIXEL_TO_KEEP = 0.40   # hottest % shown on face (0.05 = 5%, 0.20 = 20%)

# Padding around each eye bounding box (pixels)
PAD_TOP    = 15
PAD_BOTTOM = 15
PAD_LEFT   = 15
PAD_RIGHT  = 15

# Padding around face bounding box (pixels)
FACE_PAD = 20

# Input video
video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
# ============================================================

# ------------------------------------------------------------------
# MediaPipe landmark indices for LEFT and RIGHT eye rims (16 pts each)
# ------------------------------------------------------------------
LEFT_EYE_LANDMARKS = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]
RIGHT_EYE_LANDMARKS = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

ALL_FACE_LANDMARKS = list(range(468))

# -----------------------------
# MediaPipe Face Mesh Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

left_mean_values  = []
right_mean_values = []

cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)


def get_bbox(landmarks, indices, img_w, img_h,
             pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):
    xs = [int(landmarks[i].x * img_w) for i in indices]
    ys = [int(landmarks[i].y * img_h) for i in indices]
    x1 = max(0,     min(xs) - pad_left)
    y1 = max(0,     min(ys) - pad_top)
    x2 = min(img_w, max(xs) + pad_right)
    y2 = min(img_h, max(ys) + pad_bottom)
    return x1, y1, x2, y2


# -----------------------------
# Process Video
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    # Black canvas — hottest face pixels will be written here
    filt_frame = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lms = face_landmarks.landmark

            # ── 1. FACE bounding box ─────────────────────────────────────────
            fx1, fy1, fx2, fy2 = get_bbox(
                lms, ALL_FACE_LANDMARKS, w, h,
                pad_top=FACE_PAD, pad_bottom=FACE_PAD,
                pad_left=FACE_PAD, pad_right=FACE_PAD
            )
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (fx1, fy1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ── 2. EYE bounding boxes (for eye mean signal only) ─────────────
            lx1, ly1, lx2, ly2 = get_bbox(
                lms, LEFT_EYE_LANDMARKS, w, h,
                PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT
            )
            rx1, ry1, rx2, ry2 = get_bbox(
                lms, RIGHT_EYE_LANDMARKS, w, h,
                PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT
            )
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(frame, "L", (lx1, ly1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(frame, "R", (rx1, ry1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # ── 3. Extract grey ROIs ─────────────────────────────────────────
            box_face  = grey[fy1:fy2, fx1:fx2]   # whole face ROI
            box_left  = grey[ly1:ly2, lx1:lx2]   # left eye ROI
            box_right = grey[ry1:ry2, rx1:rx2]   # right eye ROI

            if box_face.size == 0 or box_left.size == 0 or box_right.size == 0:
                continue

            # ── 4. Eye hottest N% → mean for graph ───────────────────────────
            n_left  = max(1, int(box_left.size  * PERCENTAGE_PIXEL_TO_KEEP))
            n_right = max(1, int(box_right.size * PERCENTAGE_PIXEL_TO_KEEP))

            left_mean  = np.sort(box_left.flatten())[::-1][:n_left].mean()
            right_mean = np.sort(box_right.flatten())[::-1][:n_right].mean()

            left_mean_values.append(left_mean)
            right_mean_values.append(right_mean)

            print(f"Left mean: {left_mean:.2f}   Right mean: {right_mean:.2f}")

            # ── 5. FACE hottest N% pixels glow on filt_frame ─────────────────
            face_thresh = np.percentile(box_face, (1 - PERCENTAGE_PIXEL_TO_KEEP) * 100)
            face_mask   = box_face >= face_thresh

            face_roi_out = filt_frame[fy1:fy2, fx1:fx2]
            face_roi_out[face_mask] = box_face[face_mask]
            filt_frame[fy1:fy2, fx1:fx2] = face_roi_out

            # Face box outline on filt_frame (dim)
            cv2.rectangle(filt_frame, (fx1, fy1), (fx2, fy2), 60, 1)

    # ── Display ──────────────────────────────────────────────────────────────
    cv2.imshow("Video – Face & Eye ROI", frame)

    # Apply JET colormap, then force background to pure black
    colored = cv2.applyColorMap(filt_frame, cv2.COLORMAP_JET)
    colored[filt_frame == 0] = [0, 0, 0]

    cv2.imshow(f"Hottest {int(PERCENTAGE_PIXEL_TO_KEEP * 100)}% Face Pixels", colored)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

# -----------------------------
# Release Resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(20, 3))
plt.plot(left_mean_values,  label="Left Eye")
plt.plot(right_mean_values, label="Right Eye")
plt.xlabel("Frame Number")
plt.ylabel(f"Mean Intensity (hottest {int(PERCENTAGE_PIXEL_TO_KEEP * 100)}%)")
plt.title("Peri-Orbital Thermal Signal – Full Eye ROI")
plt.legend()
plt.tight_layout()
plt.show()