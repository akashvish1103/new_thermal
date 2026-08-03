# Calculating the angle of the HEAD TILT

import cv2
import mediapipe as mp
import numpy as np
import math

def put_shades(frame):
    pass

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True)


index_list = [33, 133, 362, 263]
shades_index_lst = [145, 159]
shades = []

vid_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# vid_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# vid_path = r"D:\Lie Detection Data HTI\Girish\girish_grey_manual.mpg"
cap = cv2.VideoCapture(vid_path)

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

        
        for face_landmarks in results.multi_face_landmarks:   # Looping through all the faces obtained
                        
            x_lst = []
            y_lst = []
            # -------------------------------
            # Draw Nose Landmarks
            # -------------------------------
            for idx in index_list:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)
                

                x_lst.append(x)
                y_lst.append(y)

                # cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            for idx in shades_index_lst:
                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * y)

                shades.append((x,y))


            x_mid_left = int((x_lst[0] + x_lst[1])/2)
            x_mid_right = int(x_lst[2] + x_lst[3])/2

            p1 = (x_lst[0], y_lst[0])
            p2 = (x_lst[3], y_lst[3])

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            angle = str(round(math.degrees(math.atan2(dy, dx)), 1))                   # Angle b/w two points (in degrees)

            cv2.circle(frame, (x_mid_left, y_lst[0]), 3, (0, 0, 255), -1)

            cv2.line(frame, p1, p2, (0, 255, 0), 1)
            cv2.line(frame,p1,(x_lst[0] + 100, y_lst[0]),(0, 255, 0), 1 )

            cv2.putText(frame,
            angle,
            (x_lst[0] + 100, y_lst[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (128,0,128),
            2)

    cv2.imshow("my-win", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
# -----------------------------------
# Release
# -----------------------------------
cap.release()
cv2.destroyAllWindows()