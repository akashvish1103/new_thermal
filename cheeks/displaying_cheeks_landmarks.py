import cv2
import mediapipe as mp

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

# RIGHT_CHEEK = [
#     280, 330, 347, 348, 349,
#     350, 352, 355, 371, 411,
#     423, 425, 426, 427, 432,
#     433, 436
# ]

ALL_CHEEKS = LEFT_CHEEK 

# -----------------------------
# Webcam
# -----------------------------
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub4_priyank\output_priyank_grey_manual.mp4"
cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            for idx in ALL_CHEEKS:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

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

    cv2.imshow("Cheek Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()