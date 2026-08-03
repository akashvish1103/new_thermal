# making NOSE ROI for breathing pattern detection using Mediapipe and elimanating the CSRT tracker failures.
# Contaiend modules for UTILITIES file (to get TRANSFORMED frame)
# this is new PRSG code for breathing.

import cv2
import numpy as np
import mediapipe as mp
import utilities as ut
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend

UPPER_LIPS_LANDMARK = [13, 206, 426]
NOSE_HORIZONTAL_LANDMARKS = [64, 278]
NOSE_VERTICAL_LANDMARKS = [4, 94]

LEFT_HORIZONTAL_LANDMARK = 64
RIGHT_HORIZONTAL_LANDMARK = 278

UP_VERTICAL_LANDMARK = 4
DOWN_VERTICAL_LANDMARK = 94

lst = []
lst_temp = []


# NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS
NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK

# Mapping from Pixel Intensity to Temperature (Celcieus)
def get_temp_from_pixel(pixel_value):
    m = 0.05891454 
    b = 30.07676744
    return m * pixel_value + b

# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub4_priyank\output_priyank_grey_manual.mp4"

# -----------------------------
# Input Video Path
# -----------------------------
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\all_grey_manual_mp4\aditi_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Swamini\swamini_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\purva_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = r"C:\Users\Akash Vishwakarma\Pictures\Camera Roll\WIN_20260525_15_35_42_Pro.mpq4"
video_path = r"D:\Lie Detection Data HTI\Prem\prem_grey_manual.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\Thermal_Clips\Q35.mp4"
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
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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


                # For expanding the rectangular ROI in width
                

                if idx == LEFT_HORIZONTAL_LANDMARK:
                    left_margin = nose_x                  # adding margin to the left and right of the nose tip
                
                if idx == RIGHT_HORIZONTAL_LANDMARK:
                    right_margin = nose_x 

                if idx == UP_VERTICAL_LANDMARK:                # adding margin to the top and bottom of the nose tip
                    top_margin = nose_y - 0
                                    
                if idx == DOWN_VERTICAL_LANDMARK:
                    height = nose_y - top_margin
                    bottom_margin = nose_y + int(1.15*height)        # shifting bottom edge of the rectangle ROI to downside to get the clear ROI
                                                                     # shifting by the 15% of the height of the rectangular ROI.

                # nose_tip_crop = frame[top_margin:bottom_margin, left_margin:right_margin]

                # Draw a red dot on the nose tip
                # cv2.circle(transforemd_frame, (nose_x, nose_y), 3, (255, 255, 255), -1)
                # cv2.circle(transforemd_frame, (nose_x, nose_y), 3, (0, 255, 120), -1)

            dist = right_margin  - left_margin                      # width of the rectangular ROI
            left_margin = left_margin - int(dist*0.1)               # addding the buffer by 10% of the width
            right_margin = right_margin + int(dist*0.1)             # adding the buffer by 10% of the width

            nose_tip_crop = frame[top_margin:bottom_margin, left_margin:right_margin]
            mean_value = np.mean(nose_tip_crop)
            lst.append(np.mean(nose_tip_crop))                         # Raw Signal (Pixel Intensity)
            lst_temp.append(round(get_temp_from_pixel(mean_value),3))  # Temperature Signal (Celcius) after mapping the pixel intensity to temperature using the linear equation.

            cv2.rectangle(transforemd_frame, (left_margin, top_margin), (right_margin, bottom_margin), (255, 0, 0), 1)
            cv2.rectangle(frame, (left_margin, top_margin), (right_margin, bottom_margin), (100, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Nose Tip Detection', frame)
    cv2.imshow('Transformed_frame', transforemd_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

# Making the SUBPLOTS
plt.subplot(4,1,1) 
plt.plot(lst)
plt.title("Graph 1")

# plt.subplot(4,1,2)
# plt.plot(lst_temp)
# plt.title("Graph 2")

#Making the signal smooth, using MOVING AVERAGE 
window = 10
smoothed = np.convolve(
    lst,
    np.ones(window)/window,
    mode='valid'
)

plt.subplot(4,1,4)
plt.plot(smoothed)
plt.title("Graph 4")

# plt.subplot(3,1,3)
# plt.plot(smoothed)
# plt.title("Rolling Average")
# plt.show()
# -----------------------------
# Find Peaks and Bottoms
# -----------------------------

# First derivative (slope)
slope = np.diff(smoothed)

peak_x = []
peak_y = []

bottom_x = []
bottom_y = []

# Checking sign change in slope
for i in range(1, len(slope)):

    # Peak condition (+ to -)
    if slope[i-1] > 0 and slope[i] <= 0:
        peak_x.append(i)
        peak_y.append(smoothed[i])

    # Bottom condition (- to +)
    elif slope[i-1] < 0 and slope[i] >= 0:
        bottom_x.append(i)
        bottom_y.append(smoothed[i])

# -----------------------------
# Plotting
# -----------------------------

plt.subplot(4,1,3)

# Smoothed signal
plt.plot(smoothed, label="Smoothed Signal")

# Peaks
plt.scatter(
    peak_x,
    peak_y,
    color='red',
    s=10,
    label='Peaks'
)

# Bottoms
plt.scatter(
    bottom_x,
    bottom_y,
    color='blue',
    s=10,
    label='Bottoms'
)

mean_signal = np.mean(smoothed)
std_signal = np.std(smoothed)

z_signal = (smoothed - mean_signal) / std_signal


plt.subplot(4,1,2)
plt.plot(z_signal, label="Z-Score Normalized Smoothed Signal")
plt.title("Graph 4")

plt.title("Rolling Average with Peaks and Bottoms")
plt.legend()

plt.show()





# plt.plot(smoothed)       # plotting the smoothed sign
# plt.plot(lst_temp)
# plt.show()