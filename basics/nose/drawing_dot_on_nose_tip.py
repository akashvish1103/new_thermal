import cv2
import numpy as np
import mediapipe as mp
import utilities as ut

NOSE_LANDMARKS = [19]

video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub4_priyank\output_priyank_grey_manual.mp4"
video_path = r"D:\Lie Detection Data HTI\Dhruv\dhruv_grey_manual.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\Thermal_Clips\Q47.mp4"

cap = cv2.VideoCapture(video_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transforemd_frame = ut.get_transformed_image(grey_frame)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(transforemd_frame, cv2.COLOR_GRAY2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in NOSE_LANDMARKS:
                # Get the coordinates of the nose tip (landmark index 1)
                nose_tip = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                nose_x = int(nose_tip.x * w)
                nose_y = int(nose_tip.y * h)

                left_margin = nose_x - 10                   # adding margin to the left and right of the nose tip
                right_margin = nose_x + 10

                top_margin = nose_y - 20                   # adding margin to the top and bottom of the nose tip
                bottom_margin = nose_y - 3

                nose_tip_crop = transforemd_frame[top_margin:bottom_margin, left_margin:right_margin]

                cv2.rectangle(transforemd_frame, (left_margin, top_margin), (right_margin, bottom_margin), (255, 0, 0), 1)

                # Draw a red dot on the nose tip
                cv2.circle(transforemd_frame, (nose_x, nose_y), 3, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow('Nose Tip Detection', transforemd_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    