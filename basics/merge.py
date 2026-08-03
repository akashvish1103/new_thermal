# Final file till now which have all the 6 ROIs , i.r. eyes inner corners, forehead, nose tip, cheeks polygon

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from forehead import utilities as ut                         # defined inside NOSE folder


           # how much hottest pixels to keep from eye ROI in the analysis (10% in this case)
# -----------------------------
# MediaPipe Face Mesh Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)
flag = False

left_mean_values = []
right_mean_values = []

# Inner eye corner landmarks
LEFT_INNER_EYE = 133                                      # mediapipe landmark for left eye inner corner
RIGHT_INNER_EYE = 362                                     # mediapipe landmark for right eye inner corner
PERCENTAGE_PIXEL_TO_KEEP = 0.80



# -----------------------------
# Input Video Path
# -----------------------------
video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# video_path = r"C:\Users\Akash Vishwakarma\Pictures\Camera Roll\WIN_20260525_15_35_42_Pro.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\2026-03-11 15-52-11.mp4"
# video_path = r"D:\Lie Detection Data HTI\Girish\girish_grey_manual.mpg"
cap = cv2.VideoCapture(video_path)

# ---------------------------------------------
# Video Writer (optional)
# ---------------------------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# -----------------------------
# Process Video
# -----------------------------
while True:
    points_left = []
    points_right = []
    ret, frame = cap.read()
    

    if not ret:
        break
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                # converted the 3 channel bgr frame to a single channel grey frame.

    h, w, _ = frame.shape


    # Enhancing the grey frame  to make better detection for MEDIAPIPE
    sharpened_grey_frame = ut.get_transformed_image(grey)

    # Convert grayscale -> BGR for MediaPipe   # Gibing transformed/enhanced frame to mediapipe for better detection of landmarks.
    rgb = cv2.cvtColor(
        sharpened_grey_frame,
        cv2.COLOR_GRAY2BGR
    )


    # Face Mesh Detection
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks: 
           
            top_left_coords, bottom_right_coords, top_right_coords, bottom_left_coords = ut.get_eyes_coordinates(frame, grey, face_landmarks)
            a,b,c,d = ut.get_forehead_coordinates(frame, face_landmarks, flag)
            x,y,z,w = ut.get_nose_coordinates(frame, face_landmarks)
            pl, pr = ut.get_cheeks_coordinates(frame, face_landmarks, points_left, points_right)

            


        


    # Show frame
    cv2.imshow("Video Landmark Detection", frame)



    # ESC key to exitc
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# Release Resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()