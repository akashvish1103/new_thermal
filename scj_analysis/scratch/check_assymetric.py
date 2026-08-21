# Using this as a Driver of the Utility of Scratch
# this file using all the ROI from uitlity code

import mediapipe as mp
import numpy as np
import cv2
import utilities as ut
import utility_linear_mapping as ulm



# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg" 
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40.wmv"
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

UPPER_LIPS_LANDMARK       = [13, 206, 426]
NOSE_HORIZONTAL_LANDMARKS = [64, 278]
NOSE_VERTICAL_LANDMARKS   = [4, 94]
NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK


cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformed_grey = ut.get_transformed_image(grey)

    rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2BGR)  # converting single-channel image to 3-channel image, becasue mediapie expects a RGB (3-channel) image 

    results = face_mesh.process(rgb)                          # Processing the RGB image

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:              # loop for each face found in the video

           top_left_cords, bottom_right_cords, got_frame = ut.get_breathing_roi_cords(transformed_grey, face_landmarks)
           polygon_points, mean_pixel, got_frame = ut.get_forhead_poly_coords(transformed_grey, face_landmarks)
           l,r, got_frame = ut.get_cheeks_coordinates(transformed_grey, face_landmarks, [], [])

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
           top_left_coords, bottom_right_coords, got_frame = ut.get_nose_tip_coordinates(
                                                                                transformed_grey,
                                                                                face_landmarks
                                                                                )
           print(top_left_cords, bottom_right_cords, type(got_frame))
           print(polygon_points, mean_pixel, type(got_frame))
           print(l,r, type(got_frame))
           print(top_left_coords, bottom_right_coords, top_right_coords, bottom_left_coords, type(got_frame))
           print(top_left_coords, bottom_right_coords, type(got_frame))

           print("#"*150)
        #  cv2.circle(frame, top_left_cords, 5, (255, 255, 255), -5)   # trying to display DOT on original rgb frame, by getting the
                                                                        # coordinates from the utilities function.

        

    cv2.imshow("Transformed Grey", got_frame)    #just displaying the modified got_frame, which has been processed by these modular python FUNCTIONS()
    cv2.imshow("RGB framea", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

