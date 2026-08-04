import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -----------------------------
# Cheek Landmarks
# -----------------------------
# LEFT_CHEEK = [
#     50, 101, 118, 119, 120,
#     121, 123, 126, 142, 187,
#     203, 205, 206, 207, 213,
#     214, 216
# ]

# RIGHT_CHEEK = [
#     280, 330, 347, 348, 349,
#     350, 352, 355, 371, 411,
#     423, 425, 426, 427, 432,
#     433, 436
# ]

LEFT_CHEEK = [
    214, 216, 206, 120, 101, 50, 187
]

RIGHT_CHEEK = [
    432, 436, 426, 349, 330, 280, 411
]

# RIGHT_CHEEK = [
#     280, 330, 347, 348, 349,
#     350, 352, 355, 371, 411,
#     423, 425, 426, 427, 432,
#     433, 436
# ]


ALL_CHEEKS = LEFT_CHEEK + RIGHT_CHEEK

# -----------------------------
# Webcam
# -----------------------------
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub4_priyank\output_priyank_grey_manual.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape
    points_left = []
    points_right = []
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            for idx in ALL_CHEEKS:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                if idx in LEFT_CHEEK:
                    points_left.append((x, y))
                elif idx in RIGHT_CHEEK:
                    points_right.append((x, y))

                # Draw point
                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

                # Draw index label
                cv2.putText(
                    frame,
                    str(idx),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0,0,255),
                    1
                )

            polygon_left = np.array(points_left, dtype=np.int32)
            polygon_right = np.array(points_right, dtype=np.int32)
                # -----------------------------
                # Create empty mask
                # -----------------------------

            # Drawing the edges of the closed cure, i.e ploygon
            cv2.polylines(
                        frame,
                        [polygon_left],
                        isClosed=True,
                        color=(0,255,0),
                        thickness=1
                    )
            cv2.polylines(
                        frame,
                        [polygon_right],
                        isClosed=True,
                        color=(0,255,0),
                        thickness=1
                    )
            
            mask_left = np.zeros(frame.shape, dtype=np.uint8)     # making a ndarray of zeros with the same shape as frame, this will be used as a mask to extract the polygon region
            mask_right = np.zeros(frame.shape, dtype=np.uint8)

                # Fill polygon region with white
            cv2.fillPoly(mask_left, [polygon_left], 255)           # this line will modify mask to have the polygon region filled with 255 and remaining 0
            cv2.fillPoly(mask_right, [polygon_right], 255)

                # -----------------------------
                # Keep only polygon pixels
                # -----------------------------
            result_left = cv2.bitwise_and(frame, mask_left)
            result_right = cv2.bitwise_and(frame, mask_right)   



    cv2.imshow("Cheek Landmarks", frame)
    cv2.imshow("Mask Left", mask_left)
    cv2.imshow("Mask Right", mask_right)
    cv2.imshow("Polygon Extracted Left", result_left)
    cv2.imshow("Polygon Extracted Right", result_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()