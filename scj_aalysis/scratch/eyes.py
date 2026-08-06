import mediapipe as mp
import numpy as np
import cv2
import utilities as ut


video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg" 

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

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformed_grey = ut.get_transformed_image(grey)

    rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2BGR)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks: 

            (
                top_left_coords,
                bottom_right_coords,
                top_right_coords,
                bottom_left_coords, got_frame
            ) = ut.get_eyes_coordinates(
                transformed_grey,
                face_landmarks
            )                                                   
            # print(top_left_coords, bottom_right_coords, top_right_coords, bottom_left_coords)
            top_left_coords, bottom_right_coords, got_frame = ut.get_nose_coordinates(
                transformed_grey,
                face_landmarks
            )

            print(top_left_coords, bottom_right_coords)

            # cv2.rectangle(
            #     transformed_grey,
            #     top_left_coords,
            #     bottom_right_coords,
            #     (0,255,0),
            #     2
            # )

            # cv2.rectangle(
            #     transformed_grey,
            #     top_right_coords,
            #     bottom_left_coords,
            #     (0,255,0),
            #     2
            # )

    cv2.imshow("Transformed Grey", got_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
       
