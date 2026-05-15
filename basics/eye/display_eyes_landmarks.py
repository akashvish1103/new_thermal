import cv2
import mediapipe as mp

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

# ============================================================
# Landmark Indices
# ============================================================

# Eyes
LEFT_INNER_EYE  = 133
LEFT_OUTER_EYE  = 33

RIGHT_INNER_EYE = 362
RIGHT_OUTER_EYE = 263

# Nose Tip
NOSE_TIP = 1

# Mouth corners
MOUTH_LEFT  = 61
MOUTH_RIGHT = 291

# ============================================================
# Video Capture Setup
# ============================================================
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
video_path = rvideo_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face landmark detection
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ====================================================
            # Get Coordinates
            # ====================================================

            # LEFT INNER EYE
            left_inner = face_landmarks.landmark[LEFT_INNER_EYE]
            lix = int(left_inner.x * w)
            liy = int(left_inner.y * h)

            # LEFT OUTER EYE
            left_outer = face_landmarks.landmark[LEFT_OUTER_EYE]
            lox = int(left_outer.x * w)
            loy = int(left_outer.y * h)

            # RIGHT INNER EYE
            right_inner = face_landmarks.landmark[RIGHT_INNER_EYE]
            rix = int(right_inner.x * w)
            riy = int(right_inner.y * h)

            # RIGHT OUTER EYE
            right_outer = face_landmarks.landmark[RIGHT_OUTER_EYE]
            rox = int(right_outer.x * w)
            roy = int(right_outer.y * h)

            # NOSE TIP
            nose = face_landmarks.landmark[NOSE_TIP]
            nx = int(nose.x * w)
            ny = int(nose.y * h)

            # MOUTH LEFT
            mouth_left = face_landmarks.landmark[MOUTH_LEFT]
            mlx = int(mouth_left.x * w)
            mly = int(mouth_left.y * h)

            # MOUTH RIGHT
            mouth_right = face_landmarks.landmark[MOUTH_RIGHT]
            mrx = int(mouth_right.x * w)
            mry = int(mouth_right.y * h)

            # ====================================================
            # Draw Landmarks
            # ====================================================

            # Left Eye
            cv2.circle(frame, (lix, liy), 4, (0, 255, 0), -1)
            # cv2.putText(frame, "L Inner", (lix, liy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 1)

            cv2.circle(frame, (lox, loy), 4, (0, 255, 0), -1)
            # cv2.putText(frame, "L Outer", (lox, loy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 1)

            # Right Eye
            cv2.circle(frame, (rix, riy), 4, (255, 0, 0), -1)
            # cv2.putText(frame, "R Inner", (rix, riy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 0, 0), 1)

            cv2.circle(frame, (rox, roy), 4, (255, 0, 0), -1)
            # cv2.putText(frame, "R Outer", (rox, roy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 0, 0), 1)

            # Nose Tip
            cv2.circle(frame, (nx, ny), 5, (0, 0, 255), -1)
            # cv2.putText(frame, "Nose Tip", (nx, ny - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 0, 255), 1)

            # Mouth Corners
            cv2.circle(frame, (mlx, mly), 4, (255, 255, 0), -1)
            # cv2.putText(frame, "Mouth L", (mlx, mly - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 0), 1)

            cv2.circle(frame, (mrx, mry), 4, (255, 255, 0), -1)
            # cv2.putText(frame, "Mouth R", (mrx, mry - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 0), 1)

    # ============================================================
    # Show Frame
    # ============================================================
    cv2.imshow("Face Landmarks", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================================
# Release Resources
# ============================================================
cap.release()
cv2.destroyAllWindows()