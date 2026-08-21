import mediapipe as mp
import numpy as np
import cv2
import utilities as ut


# ============================================================
# MEDIAPIPE FACE MESH SETUP
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    points_left = []
    points_right = []

    if not ret:
        break

    h, w, _ = frame.shape

    # Convert the frame to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply transformations to the grey image
    transformed_grey = ut.get_transformed_image(grey)

    # Convert the original grey image to RGB format
    rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
    
            for face_landmarks in results.multi_face_landmarks:          # iterating over all the faces 
                flag = None
                # =================================================
                # GET ROI COORDINATES
                # =================================================
    
                (
                    top_left_coords,
                    bottom_right_coords,
                    top_right_coords,
                    bottom_left_coords
    
                ) = ut.get_eyes_coordinates(
                    frame,
                    grey,
                    face_landmarks
                )
    
                (
                    a,
                    b,
                    c,
                    d
    
                ) = ut.get_forehead_coordinates(
                    frame,
                    face_landmarks,
                    flag
                )
    
                (
                    x,
                    y,
                    z,
                    w1
    
                ) = ut.get_nose_coordinates(
                    frame,
                    face_landmarks
                )
    
                (
                    pl,
                    pr
    
                ) = ut.get_cheeks_coordinates(
                    frame,
                    face_landmarks,
                    points_left,
                    points_right
                )

                print("top_left_coords", top_left_coords)
                print("bottom_right_coords", bottom_right_coords)
                print("top_right_coords", top_right_coords)
                print("bottom_left_coords", bottom_left_coords)
                print("abcd",a, b, c, d)
                print("xyzw1",x, y, z, w1)
                print("plpr",pl, pr)

    cv2.circle(frame, top_left_coords, 2, (0, 255, 0), -1)
    cv2.circle(frame, bottom_right_coords, 2, (0, 255, 0), -1)
    cv2.circle(frame, top_right_coords, 2, (0, 255, 0), -1)
    cv2.circle(frame, bottom_left_coords, 2, (0, 255, 0), -1)   

    cv2.circle(frame, x, 2, (0, 255, 0), -1)
    cv2.circle(frame, y, 2, (0, 255, 0), -1)
    cv2.circle(frame, z, 2, (0, 255, 0), -1)
    cv2.circle(frame, w1, 2, (0, 255, 0), -1)
    cv2.circle(frame, pl, 2, (0, 255, 0), -1)
    cv2.circle(frame, pr, 2, (0, 255, 0), -1)  

    cv2.imshow("Frame with Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                
    


    

