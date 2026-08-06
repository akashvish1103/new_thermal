import cv2
import mediapipe as mp
import numpy as np
import utilities as ut

#Using the transofrmed frame from utility to give to mediapipe for getting the landmarks of the face and then extracting the forehead polygon from the landmarks and displaying it on the frame.

# --------------------------------------------------
# Video Path
# --------------------------------------------------
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"

# --------------------------------------------------
# MediaPipe Face Mesh
# --------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# frame = ut.get_transformed_image(frame)

# --------------------------------------------------
# Forehead Landmark Indices
# --------------------------------------------------
FOREHEAD_POINTS = [
    10,
    109,
    108,
    69,
    67,
    103,
    104,
    332,
    297,
    299,
    338,
    337
]

# --------------------------------------------------
# Open Video
# --------------------------------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

    frame = ut.get_transformed_image(grey_frame)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # -----------------------------
        # Draw all landmarks
        # -----------------------------
        for idx, lm in enumerate(face_landmarks.landmark):

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.putText(
                frame,
                str(idx),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 0, 255),
                1
            )

        # -----------------------------
        # Forehead Polygon
        # -----------------------------
        polygon = []

        for idx in FOREHEAD_POINTS:

            lm = face_landmarks.landmark[idx]

            x = int(lm.x * w)
            y = int(lm.y * h)

            polygon.append([x, y])

        polygon = np.array(polygon, dtype=np.int32)

        cv2.polylines(
            frame,
            [polygon],
            True,
            (255, 0, 0),
            2
        )

    cv2.imshow("Forehead Polygon", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()