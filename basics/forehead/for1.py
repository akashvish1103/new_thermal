# import cv2
# import mediapipe as mp
# import utilities as ut
# import numpy as np

# # ----------------------------
# # Video Path
# # ----------------------------
# video_path = rvideo_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"

# # ----------------------------
# # MediaPipe FaceMesh
# # ----------------------------
# mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ----------------------------
# # Upper Face / Forehead Area
# # ----------------------------
# FOREHEAD_LANDMARKS = [
#     10,
#     67,
#     69,
#     103,
#     104,
#     105,
#     107,
#     108,
#     109,
#     151,
#     297,
#     299,
#     332,
#     333,
#     334,
#     336,
#     337,
#     338
# ]

# cap = cv2.VideoCapture(video_path)

# while True:

#     ret, frame = cap.read()

#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     transformed_grey = ut.get_transformed_image(grey)

#     if not ret:
#         break

#     h, w = frame.shape[:2]

#     rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2RGB)

#     results = face_mesh.process(rgb)

#     if results.multi_face_landmarks:

#         face = results.multi_face_landmarks[0]

#         for idx in FOREHEAD_LANDMARKS:

#             lm = face.landmark[idx]

#             x = int(lm.x * w)
#             y = int(lm.y * h)

#             # Green dot
#             cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#             cv2.circle(transformed_grey, (x, y), 1, (0, 255, 0), -1)

#             # Landmark number
#             cv2.putText(
#                 transformed_grey,
#                 str(idx),
#                 (x + 5, y - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.35,
#                 (0, 0, 255),
#                 1,
#                 cv2.LINE_AA
#             )

#     cv2.imshow("Forehead Landmarks", frame)
#     cv2.imshow("Transformed Grey", transformed_grey)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

####################################################################################################

# import cv2
# import mediapipe as mp
# import utilities as ut
# import numpy as np

# # ----------------------------
# # Video Path
# # ----------------------------
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"

# # ----------------------------
# # MediaPipe FaceMesh
# # ----------------------------
# mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ----------------------------
# # Forehead Landmarks
# # ----------------------------
# FOREHEAD_LANDMARKS = [
#     10,
#     67,
#     69,
#     103,
#     104,
#     105,
#     107,
#     108,
#     109,
#     151,
#     297,
#     299,
#     332,
#     333,
#     334,
#     336,
#     337,
#     338
# ]

# # ----------------------------
# # Forehead ROI Polygon
# # (in the exact order you specified)
# # ----------------------------
# FOREHEAD_POLYGON = [
#     67,
#     109,
#     10,
#     338,
#     297,
#     333,
#     334,
#     107,
#     69,
#     67
# ]

# cap = cv2.VideoCapture(video_path)

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     transformed_grey = ut.get_transformed_image(grey)

#     h, w = frame.shape[:2]

#     rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2RGB)

#     results = face_mesh.process(rgb)

#     if results.multi_face_landmarks:

#         face = results.multi_face_landmarks[0]

#         # --------------------------------------
#         # Draw Forehead Landmarks
#         # --------------------------------------
#         for idx in FOREHEAD_LANDMARKS:

#             lm = face.landmark[idx]

#             x = int(lm.x * w)
#             y = int(lm.y * h)

#             # Small green dots
#             cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#             cv2.circle(transformed_grey, (x, y), 2, (0, 255, 0), -1)

#             # Landmark number
#             cv2.putText(
#                 transformed_grey,
#                 str(idx),
#                 (x + 3, y - 3),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.30,
#                 (255, 255, 255),
#                 1,
#                 cv2.LINE_AA
#             )

#         # --------------------------------------
#         # Draw Forehead ROI Polygon
#         # --------------------------------------
#         polygon_points = []

#         for idx in FOREHEAD_POLYGON:

#             lm = face.landmark[idx]

#             x = int(lm.x * w)
#             y = int(lm.y * h)

#             polygon_points.append([x, y])

#         polygon_points = np.array(polygon_points, dtype=np.int32)

#         # Polygon on original frame
#         cv2.polylines(
#             frame,
#             [polygon_points],
#             isClosed=True,
#             color=(255, 0, 0),   # Blue
#             thickness=2
#         )

#         # Polygon on transformed image
#         cv2.polylines(
#             transformed_grey,
#             [polygon_points],
#             isClosed=True,
#             color=(255, 255, 255),   # White
#             thickness=2
#         )

#     cv2.imshow("Forehead Landmarks", frame)
#     cv2.imshow("Transformed Grey", transformed_grey)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

################################################################################################################################################################

import cv2
import mediapipe as mp
import utilities as ut
import numpy as np

# ----------------------------
# Video Path
# ----------------------------
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"

# ----------------------------
# MediaPipe FaceMesh
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Forehead Landmarks
# ----------------------------
FOREHEAD_LANDMARKS = [
    10,
    67,
    69,
    103,
    104,
    105,
    107,
    108,
    109,
    151,
    297,
    299,
    332,
    333,
    334,
    336,
    337,
    338
]

# ----------------------------
# Forehead ROI Polygon
# ----------------------------
FOREHEAD_POLYGON = [
    67,
    109,
    10,
    338,
    297,
    333,
    334,
    107,
    104,
]

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    transformed_grey = ut.get_transformed_image(grey)

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        # ----------------------------
        # Draw Landmarks
        # ----------------------------
        for idx in FOREHEAD_LANDMARKS:

            lm = face.landmark[idx]

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.circle(transformed_grey, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(
                transformed_grey,
                str(idx),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # ----------------------------
        # Polygon Points
        # ----------------------------
        polygon_points = []

        for idx in FOREHEAD_POLYGON:

            lm = face.landmark[idx]

            x = int(lm.x * w)
            y = int(lm.y * h)

            polygon_points.append([x, y])

        polygon_points = np.array(polygon_points, dtype=np.int32)

        # ----------------------------
        # Draw Polygon
        # ----------------------------
        cv2.polylines(
            frame,
            [polygon_points],
            True,
            (255, 0, 0),
            2
        )

        cv2.polylines(
            transformed_grey,
            [polygon_points],
            True,
            255,
            2
        )

        # =====================================================
        # Create Mask
        # =====================================================
        mask = np.zeros(grey.shape, dtype=np.uint8)

        cv2.fillPoly(mask, [polygon_points], 255)

        # =====================================================
        # Extract ROI
        # =====================================================
        forehead_roi = cv2.bitwise_and(grey, grey, mask=mask)

        # =====================================================
        # Average Pixel Value
        # =====================================================
        mean_pixel = cv2.mean(grey, mask=mask)[0]

        print(f"Average Pixel Value = {mean_pixel:.2f}")

        # =====================================================
        # Display Average on Frame
        # =====================================================
        cv2.putText(
            frame,
            f"Avg = {mean_pixel:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # =====================================================
        # Display ROI
        # =====================================================
        cv2.imshow("Forehead ROI", forehead_roi)

    cv2.imshow("Forehead Landmarks", frame)
    cv2.imshow("Transformed Grey", transformed_grey)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()