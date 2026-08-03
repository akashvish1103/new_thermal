import cv2
import mediapipe as mp

# -----------------------------------
# MediaPipe FaceMesh Setup
# -----------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -----------------------------------
# Landmark IDs
# -----------------------------------

# Nose landmarks
NOSE = [1, 2, 98, 327, 168]

# Left eye landmarks
LEFT_EYE = [33, 133, 159, 145]

# Right eye landmarks
RIGHT_EYE = [362, 263, 386, 374]

# Forehead landmark
FOREHEAD = [10, 151, 9]

# -----------------------------------
# Input Video
# -----------------------------------
video_path = r"D:\Lie Detection Data HTI\Akash\aa.wmv"

cap = cv2.VideoCapture(video_path)

# -----------------------------------
# Process Video
# -----------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FaceMesh detection
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:             # Looping through all the faces obtained

            # -------------------------------
            # Draw Nose Landmarks
            # -------------------------------
            for idx in NOSE:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # -------------------------------
            # Draw Left Eye Landmarks
            # -------------------------------
            for idx in LEFT_EYE:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # -------------------------------
            # Draw Right Eye Landmarks
            # -------------------------------
            for idx in RIGHT_EYE:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            # -------------------------------
            # Draw Forehead Landmarks
            # -------------------------------
            for idx in FOREHEAD:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    # -----------------------------------
    # Display
    # -----------------------------------
    cv2.imshow("Facial Landmarks", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

# -----------------------------------
# Release
# -----------------------------------
cap.release()
cv2.destroyAllWindows()