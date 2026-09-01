# import cv2
# import mediapipe as mp
# import utilities as ut

# video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\4.mp4"

# cap = cv2.VideoCapture(video_path)

# mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.3
# )

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Rotate 90 degrees RIGHT
#     frame = cv2.rotate(
#         frame,
#         cv2.ROTATE_90_CLOCKWISE
#     )
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = ut.get_transformed_image(grey)

#     h, w = frame.shape[:2]

#     # Convert to RGB
#     rgb = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2RGB
#     )

#     # MediaPipe
#     results = face_mesh.process(rgb)

#     # ==========================================
#     # FACE DETECTION
#     # ==========================================
#     if results.multi_face_landmarks:

#         print("YES")

#         # Draw landmarks
#         for face_landmarks in results.multi_face_landmarks:

#             for landmark in face_landmarks.landmark:

#                 x = int(landmark.x * w)
#                 y = int(landmark.y * h)

#                 if 0 <= x < w and 0 <= y < h:

#                     cv2.circle(
#                         frame,
#                         (x, y),
#                         2,
#                         (0, 255, 0),
#                         -1
#                     )

#         # Show YES on video
#         cv2.putText(
#             frame,
#             "FACE: YES",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )

#     else:

#         print("NO")

#         # Show NO on video
#         cv2.putText(
#             frame,
#             "FACE: NO",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )

#     # Display
#     cv2.imshow(
#         "Rotated 90 Right",
#         frame
#     )

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# face_mesh.close()
# cv2.destroyAllWindows()

###########################

# For YOGA Thermal

# This code is used to compare the 2 methods for face detection : 1-only rotate original frame,  2- rotate and then tranform
# Results: transfored grey frame works good Mediapipe face detection

import cv2
import mediapipe as mp
import utilities as ut

video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\4.mp4"

cap = cv2.VideoCapture(video_path)

mp_face_mesh = mp.solutions.face_mesh

# ============================================================
# TWO MEDIAPIPE MODELS
# ============================================================

face_mesh_original = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

face_mesh_transformed = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ============================================================
# COUNTERS
# ============================================================

total_frames = 0

detected_original = 0
detected_transformed = 0

# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    total_frames += 1

    # ========================================================
    # ROTATE 90 DEGREES RIGHT
    # ========================================================

    rotated = cv2.rotate(
        frame,
        cv2.ROTATE_90_CLOCKWISE
    )

    # ========================================================
    # METHOD 1
    # ROTATED FRAME DIRECTLY → MEDIAPIPE
    # ========================================================

    rgb_original = cv2.cvtColor(
        rotated,
        cv2.COLOR_BGR2RGB
    )

    results_original = face_mesh_original.process(
        rgb_original
    )

    if results_original.multi_face_landmarks:
        detected_original += 1

    # ========================================================
    # METHOD 2
    # ROTATED → GRAYSCALE → TRANSFORMED → MEDIAPIPE
    # ========================================================

    grey = cv2.cvtColor(
        rotated,
        cv2.COLOR_BGR2GRAY
    )

    transformed = ut.get_transformed_image(grey)

    rgb_transformed = cv2.cvtColor(
        transformed,
        cv2.COLOR_GRAY2RGB
    )

    results_transformed = face_mesh_transformed.process(
        rgb_transformed
    )

    if results_transformed.multi_face_landmarks:
        detected_transformed += 1

    # ========================================================
    # DISPLAY
    # ========================================================

    display = rotated.copy()

    cv2.putText(
        display,
        f"Original: {detected_original}/{total_frames}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Transformed: {detected_transformed}/{total_frames}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Face Detection Comparison",
        display
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================================
# FINAL RESULTS
# ============================================================

cap.release()

face_mesh_original.close()
face_mesh_transformed.close()

cv2.destroyAllWindows()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FACE DETECTION COMPARISON")
print("=" * 60)

print(f"\nTotal frames: {total_frames}")

print("\nMETHOD 1: Rotated frame directly")
print(f"Face detected: {detected_original} frames")
print(
    f"Detection rate: "
    f"{(detected_original / total_frames) * 100:.2f}%"
)

print("\nMETHOD 2: Rotated + Transformed")
print(f"Face detected: {detected_transformed} frames")
print(
    f"Detection rate: "
    f"{(detected_transformed / total_frames) * 100:.2f}%"
)

print("\n" + "=" * 60)

if detected_original > detected_transformed:

    print("BETTER METHOD: Original rotated frame")

elif detected_transformed > detected_original:

    print("BETTER METHOD: Transformed frame")

else:

    print("Both methods detected the same number of frames.")

print("=" * 60)