# # Making graph of the Nose ROI region from GREY image (and not TRANSFORMED IMAGE)

# import mediapipe as mp
# import numpy as np
# import cv2
# import utilities as ut
# import matplotlib.pyplot as plt

# # video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg" 
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40.wmv"

# # ============================================================
# # MediaPipe Face Mesh Setup
# # ============================================================

# mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# UPPER_LIPS_LANDMARK       = [13, 206, 426]
# NOSE_HORIZONTAL_LANDMARKS = [64, 278]
# NOSE_VERTICAL_LANDMARKS   = [4, 94]
# NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK

# arr = np.array([])
# cap = cv2.VideoCapture(video_path)

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     transformed_grey = ut.get_transformed_image(grey)

#     rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2BGR)  # converting single-channel image to 3-channel image, becasue mediapie expects a RGB (3-channel) image 

#     results = face_mesh.process(rgb)                          # Processing the RGB image

#     if results.multi_face_landmarks:

#         for face_landmarks in results.multi_face_landmarks:              # loop for each face found in the video

#            top_left_coords_bb, bottom_right_coords_bb, got_frame = ut.get_breathing_roi_cords(transformed_grey, face_landmarks)
#            polygon_points, mean_pixel, got_frame = ut.get_forhead_poly_coords(transformed_grey, face_landmarks)
#            l,r, got_frame = ut.get_cheeks_coordinates(transformed_grey, face_landmarks, [], [])

#            (
#             top_left_coords,
#                 bottom_right_coords,
#                 top_right_coords,
#                 bottom_left_coords, got_frame
#             ) = ut.get_eyes_coordinates(
#                 transformed_grey,
#                 face_landmarks
#             )                                                   
#             # print(top_left_coords, bottom_right_coords, top_right_coords, bottom_left_coords)
#            top_left_coords, bottom_right_coords, got_frame = ut.get_nose_tip_coordinates(
#                 transformed_grey,
#                 face_landmarks
#             )
#            print(top_left_coords_bb, bottom_right_coords_bb, type(got_frame))
#         #    print(polygon_points, mean_pixel, type(got_frame))
#         #    print(l,r, type(got_frame))
#         #    print(top_left_coords,
#         #                    bottom_right_coords,
#         #                    top_right_coords,
#         #                    bottom_left_coords, type(got_frame))
#         #    print(top_left_coords, bottom_right_coords, type(got_frame))
#         #    print("#"*150)
#         #    cv2.circle(frame, top_left_cords, 5, (255, 255, 255), -5)   # trying to display DOT on original rgb frame, by getting the
#                                                                         # coordinates from the utilities function.
#         # Extracting the ROI from grey_frame (convert from iron coded rgb to grey via cv2.cvtColor) 
        
#         # Using the GREY image (converted from Iron Coded)
#         extracted_breathing_box_from_grey = grey[top_left_coords_bb[1]:bottom_right_coords_bb[1], top_left_coords_bb[0]: bottom_right_coords_bb[0]]  

#         mean_pixel = ut.get_mean_of_ROI(extracted_breathing_box_from_grey)

#         arr = np.append(arr, mean_pixel)
#         print(f"Average Pixel Value = {mean_pixel}")
           

                                                                        

    
        
#     print(extracted_breathing_box_from_grey.shape)
#     cv2.imshow("Transformed Grey", got_frame)    #just displaying the modified got_frame, which has been processed by these modular python FUNCTIONS()
#     cv2.imshow("RGB framea", frame)
#     cv2.imshow("Extracted Box", extracted_breathing_box_from_grey)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# plt.figure(figsize=(12, 5))
# plt.plot(arr, linewidth=1)

# # plt.title("Mean Pixel Value of Breathing ROI")
# # plt.xlabel("Frame Number")
# # plt.ylabel("Mean Pixel Value")
# # plt.grid(True)

# plt.show()
# cap.release()
# cv2.destroyAllWindows()
       
#######################################


# Making graph of the Nose ROI region from GREY image (and not TRANSFORMED IMAGE)

import mediapipe as mp
import numpy as np
import cv2
import utilities as ut
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter   # <-- added

# ============================================================
# VIDEO PATH
# ============================================================

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"

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
NOSE_LANDMARKS = (
    NOSE_HORIZONTAL_LANDMARKS
    + NOSE_VERTICAL_LANDMARKS
    + UPPER_LIPS_LANDMARK
)

# ============================================================
# Signal storage
# ============================================================

arr = []

# ============================================================
# Open video
# ============================================================

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# ============================================================
# Main video loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    transformed_grey = ut.get_transformed_image(grey)

    # MediaPipe expects 3-channel image
    rgb = cv2.cvtColor(
        transformed_grey,
        cv2.COLOR_GRAY2BGR
    )

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ------------------------------------------------
            # Breathing ROI
            # ------------------------------------------------

            (
                top_left_coords_bb,
                bottom_right_coords_bb,
                got_frame
            ) = ut.get_breathing_roi_cords(
                transformed_grey,
                face_landmarks
            )

            # ------------------------------------------------
            # Other ROIs
            # ------------------------------------------------

            polygon_points, mean_pixel, got_frame = (
                ut.get_forhead_poly_coords(
                    transformed_grey,
                    face_landmarks
                )
            )

            l, r, got_frame = ut.get_cheeks_coordinates(
                transformed_grey,
                face_landmarks,
                [],
                []
            )

            (
                top_left_coords,
                bottom_right_coords,
                top_right_coords,
                bottom_left_coords,
                got_frame
            ) = ut.get_eyes_coordinates(
                transformed_grey,
                face_landmarks
            )

            top_left_coords, bottom_right_coords, got_frame = (
                ut.get_nose_tip_coordinates(
                    transformed_grey,
                    face_landmarks
                )
            )

            # ------------------------------------------------
            # Extract breathing ROI from ORIGINAL GREY image
            # ------------------------------------------------

            extracted_breathing_box_from_grey = grey[
                top_left_coords_bb[1]:bottom_right_coords_bb[1],
                top_left_coords_bb[0]:bottom_right_coords_bb[0]
            ]

            # ------------------------------------------------
            # Calculate mean pixel intensity
            # ------------------------------------------------

            mean_pixel = ut.get_mean_of_ROI(
                extracted_breathing_box_from_grey
            )

            # Store signal
            arr.append(mean_pixel)

            print(
                f"Average Pixel Value = {mean_pixel:.2f}"
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    if results.multi_face_landmarks:

        print(
            "ROI shape:",
            extracted_breathing_box_from_grey.shape
        )

        cv2.imshow(
            "Transformed Grey",
            got_frame
        )

        cv2.imshow(
            "RGB frame",
            frame
        )

        cv2.imshow(
            "Extracted Box",
            extracted_breathing_box_from_grey
        )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================================
# Release video
# ============================================================

cap.release()
cv2.destroyAllWindows()
face_mesh.close()


# ============================================================
# Convert signal to NumPy array
# ============================================================

arr = np.array(arr, dtype=float)

print("\nTotal signal samples:", len(arr))


# ============================================================
# Smoothing
# ============================================================

# ------------------------------------------------------------
# 1. Simple Moving Average (SMA)
# ------------------------------------------------------------

# Number of samples used for averaging
sma_window = 20

# Calculate Simple Moving Average
sma_signal = np.convolve(
    arr,
    np.ones(sma_window) / sma_window,
    mode='same'
)


# ------------------------------------------------------------
# 2. Savitzky-Golay Smoothing
# ------------------------------------------------------------

# Window length MUST be odd
sg_window_length = 11

# Polynomial order
sg_polyorder = 2

smoothed_signal = savgol_filter(
    arr,
    window_length=sg_window_length,
    polyorder=sg_polyorder
)


# ============================================================
# Plot RAW + SMA + SAVITZKY-GOLAY
# ============================================================

time = np.arange(len(arr)) / fps

plt.figure(figsize=(14, 6))


# ------------------------------------------------------------
# Raw signal
# ------------------------------------------------------------

plt.plot(
    time,
    arr,
    linewidth=0.7,
    alpha=0.35,
    label="Raw signal"
)


# ------------------------------------------------------------
# Simple Moving Average
# ------------------------------------------------------------

plt.plot(
    time,
    sma_signal,
    linewidth=2,
    label=f"Simple Moving Average (window={sma_window})"
)


# ------------------------------------------------------------
# Savitzky-Golay
# ------------------------------------------------------------

plt.plot(
    time,
    smoothed_signal,
    linewidth=2,
    label=f"Savitzky-Golay (window={sg_window_length}, "
           f"poly={sg_polyorder})"
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

plt.xlabel("Time (seconds)")
plt.ylabel("Mean Pixel Intensity")

plt.title("Nose ROI Signal: Raw vs SMA vs Savitzky-Golay")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()