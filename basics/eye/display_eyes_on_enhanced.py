import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# MediaPipe Setup
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
LEFT_INNER_EYE  = 133
LEFT_OUTER_EYE  = 33

RIGHT_INNER_EYE = 362
RIGHT_OUTER_EYE = 263

NOSE_TIP = 1

MOUTH_LEFT  = 61
MOUTH_RIGHT = 291

# ============================================================
# INPUT VIDEO PATH
# ============================================================
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\Thermal_Clips\Q47.mp4"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"

cap = cv2.VideoCapture(video_path)

# ============================================================
# CLAHE (Create Once)
# ============================================================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# ============================================================
# PROCESS VIDEO
# ============================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # ========================================================
    # ORIGINAL THERMAL GRAYSCALE (GROUND TRUTH)
    # ========================================================
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # CONTRAST ENHANCEMENT PIPELINE
    # ========================================================

    # Stretch contrast
    stretched = cv2.normalize(
        grey,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # CLAHE
    enhanced = clahe.apply(stretched)

    # Gamma correction
    gamma_corrected = (
        np.power(enhanced / 255.0, 0.6) * 255
    ).astype(np.uint8)

    # Unsharp mask
    blurred = cv2.GaussianBlur(
        gamma_corrected,
        (0, 0),
        3
    )

    # sharpened = cv2.addWeighted(
    #     gamma_corrected,
    #     1.5,
    #     blurred,
    #     -0.5,
    #     0
    # )

    sharpened = cv2.addWeighted(                        # sharpened the edges
    gamma_corrected,
    2,
    blurred,
    -1.5,
    0
)

    # ========================================================
    # FOR MEDIAPIPE
    # ========================================================
    
    rgb_for_mediapipe = cv2.cvtColor(
        sharpened,
        cv2.COLOR_GRAY2RGB
    )

    # ========================================================
    # MEDIAPIPE DETECTION
    # ========================================================
    results = face_mesh.process(rgb_for_mediapipe)

    # ========================================================
    # DRAW LANDMARKS
    # ========================================================
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # LEFT INNER EYE
            point = face_landmarks.landmark[LEFT_INNER_EYE]
            lix = int(point.x * w)
            liy = int(point.y * h)

            # LEFT OUTER EYE
            point = face_landmarks.landmark[LEFT_OUTER_EYE]
            lox = int(point.x * w)
            loy = int(point.y * h)

            # RIGHT INNER EYE
            point = face_landmarks.landmark[RIGHT_INNER_EYE]
            rix = int(point.x * w)
            riy = int(point.y * h)

            # RIGHT OUTER EYE
            point = face_landmarks.landmark[RIGHT_OUTER_EYE]
            rox = int(point.x * w)
            roy = int(point.y * h)

            # NOSE TIP
            point = face_landmarks.landmark[NOSE_TIP]
            nx = int(point.x * w)
            ny = int(point.y * h)

            # MOUTH LEFT
            point = face_landmarks.landmark[MOUTH_LEFT]
            mlx = int(point.x * w)
            mly = int(point.y * h)

            # MOUTH RIGHT
            point = face_landmarks.landmark[MOUTH_RIGHT]
            mrx = int(point.x * w)
            mry = int(point.y * h)

            # =================================================
            # DRAW ON ORIGINAL FRAME
            # =================================================

            # Left Eye
            cv2.circle(frame, (lix, liy), 4, (0, 255, 0), -1)
            cv2.circle(frame, (lox, loy), 4, (0, 255, 0), -1)

            cv2.circle(sharpened, (lix, liy), 4, (0, 255, 0), -1) # for enhanced image
            cv2.circle(sharpened, (lox, loy), 4, (0, 255, 0), -1)

            # Right Eye
            cv2.circle(frame, (rix, riy), 4, (255, 0, 0), -1)
            cv2.circle(frame, (rox, roy), 4, (255, 0, 0), -1)

            cv2.circle(sharpened, (rix, riy), 4, (255, 0, 0), -1) # for enhanced image
            cv2.circle(sharpened, (rox, roy), 4, (255, 0, 0), -1)


            # Nose
            cv2.circle(frame, (nx, ny), 5, (0, 0, 255), -1)

            cv2.circle(sharpened, (nx, ny), 5, (0, 0, 255), -1) # for enhanced image

            # Mouth
            cv2.circle(frame, (mlx, mly), 4, (255, 255, 0), -1)
            cv2.circle(frame, (mrx, mry), 4, (255, 255, 0), -1)

            cv2.circle(sharpened, (mlx, mly), 4, (255, 255, 0), -1) # for enhanced image
            cv2.circle(sharpened, (mrx, mry), 4, (255, 255, 0), -1) # for enhanced image

            # Labels
            cv2.putText(frame, "L Inner", (lix, liy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1)

            cv2.putText(frame, "R Inner", (rix, riy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,0,0), 1)

            cv2.putText(frame, "Nose", (nx, ny - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 1)

    # ========================================================
    # SHOW WINDOWS
    # ========================================================

    # Original thermal with landmarks
    cv2.imshow("Thermal + Landmarks", frame)

    # Enhanced image used by MediaPipe
    cv2.imshow("Enhanced for MediaPipe", sharpened)

    # ========================================================
    # EXIT
    # ========================================================
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================================
# RELEASE
# ============================================================
cap.release()
cv2.destroyAllWindows()