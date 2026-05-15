import cv2
import mediapipe as mp
import utilities as ut

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
# FOREHEAD LANDMARKS
# ============================================================

FOREHEAD_POINTS = [
    67,   # top-left
    297,  # top-right
    105,  # bottom-left
    334   # bottom-right
]

# ============================================================
# VIDEO PATH
# ============================================================

video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"

cap = cv2.VideoCapture(video_path)

# ============================================================
# TEMPORAL SMOOTHING VARIABLES
# ============================================================

prev_left = None
prev_right = None
prev_top = None
prev_bottom = None

# smoothing factor
# smaller = smoother but slower
alpha = 0.15

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    grey_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Thermal transformation
    # --------------------------------------------------------

    transformed_frame = ut.get_transformed_image(
        grey_frame
    )

    # --------------------------------------------------------
    # Height and width
    # --------------------------------------------------------

    h, w = transformed_frame.shape

    # --------------------------------------------------------
    # Convert grayscale -> BGR
    # MediaPipe requires 3 channels
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        transformed_frame,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Landmark detection
    # --------------------------------------------------------

    results = face_mesh.process(rgb)

    # ========================================================
    # FACE FOUND
    # ========================================================

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            forehead_coords = []

            # =================================================
            # EXTRACT FOREHEAD LANDMARKS
            # =================================================

            for idx in FOREHEAD_POINTS:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                forehead_coords.append((x, y))

                # ---------------------------------------------
                # Draw landmark point
                # ---------------------------------------------

                cv2.circle(
                    transformed_frame,
                    (x, y),
                    4,
                    (0,255,0),
                    -1
                )

                # ---------------------------------------------
                # Draw landmark index
                # ---------------------------------------------

                cv2.putText(
                    transformed_frame,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1
                )

            # =================================================
            # SAFETY CHECK
            # =================================================

            if len(forehead_coords) != 4:
                continue

            # =================================================
            # RAW FOREHEAD RECTANGLE
            # =================================================

            left_margin = max(
                forehead_coords[0][0],
                forehead_coords[2][0]
            )

            right_margin = min(
                forehead_coords[1][0],
                forehead_coords[3][0]
            )

            top_margin = max(
                forehead_coords[0][1],
                forehead_coords[1][1]
            )

            bottom_margin = min(
                forehead_coords[2][1],
                forehead_coords[3][1]
            )

            # =================================================
            # TEMPORAL SMOOTHING
            # =================================================

            if prev_left is None:

                prev_left = left_margin
                prev_right = right_margin
                prev_top = top_margin
                prev_bottom = bottom_margin

            else:

                left_margin = int(
                    alpha * left_margin +
                    (1 - alpha) * prev_left
                )

                right_margin = int(
                    alpha * right_margin +
                    (1 - alpha) * prev_right
                )

                top_margin = int(
                    alpha * top_margin +
                    (1 - alpha) * prev_top
                )

                bottom_margin = int(
                    alpha * bottom_margin +
                    (1 - alpha) * prev_bottom
                )

                # update previous coordinates

                prev_left = left_margin
                prev_right = right_margin
                prev_top = top_margin
                prev_bottom = bottom_margin

            # =================================================
            # REMOVE EYEBROW REGION
            # =================================================

            eyebrow_buffer = 5

            adjusted_bottom = bottom_margin - eyebrow_buffer

            # =================================================
            # CLAMP INSIDE IMAGE
            # =================================================

            left_margin = max(0, left_margin)
            top_margin = max(0, top_margin)

            right_margin = min(w, right_margin)
            adjusted_bottom = min(h, adjusted_bottom)

            # =================================================
            # VALIDITY CHECK
            # =================================================

            if left_margin >= right_margin:
                continue

            if top_margin >= adjusted_bottom:
                continue

            # =================================================
            # DRAW RECTANGLE
            # =================================================

            cv2.rectangle(
                transformed_frame,
                (left_margin, top_margin),
                (right_margin, adjusted_bottom),
                (255,255,255),
                2
            )

            cv2.rectangle(
                grey_frame,
                (left_margin, top_margin),
                (right_margin, adjusted_bottom),
                (255,255,255),
                2
            )

            # =================================================
            # FOREHEAD CROPS
            # =================================================

            forehead_crop = transformed_frame[
                top_margin:adjusted_bottom,
                left_margin:right_margin
            ]

            forehead_crop_grey = grey_frame[
                top_margin:adjusted_bottom,
                left_margin:right_margin
            ]

            # =================================================
            # SHOW CROPS
            # =================================================

            if forehead_crop.size != 0:

                cv2.imshow(
                    "Forehead Crop",
                    forehead_crop
                )

                cv2.imshow(
                    "Forehead Crop Original Grey",
                    forehead_crop_grey
                )

    # ========================================================
    # SHOW MAIN WINDOWS
    # ========================================================

    cv2.imshow(
        "Forehead Landmarks",
        transformed_frame
    )

    cv2.imshow(
        "Original Forehead Landmarks",
        grey_frame
    )

    # ========================================================
    # ESC TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================================
# RELEASE
# ============================================================

cap.release()
cv2.destroyAllWindows()