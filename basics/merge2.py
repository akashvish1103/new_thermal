# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# from forehead import utilities as ut

# # ============================================================
# # MediaPipe Face Mesh Setup
# # ============================================================

# mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True
# )

# # ============================================================
# # VARIABLES
# # ============================================================

# flag = False

# # ============================================================
# # LISTS FOR STORING MEAN VALUES
# # ============================================================

# left_eye_means = []
# right_eye_means = []

# forehead_means = []

# nose_means = []

# left_cheek_means = []
# right_cheek_means = []

# # ============================================================
# # INPUT VIDEO
# # ============================================================

# # video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# # video_path = video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"



# cap = cv2.VideoCapture(video_path)

# # ============================================================
# # PROCESS VIDEO
# # ============================================================

# while True:

#     points_left = []
#     points_right = []

#     ret, frame = cap.read()

#     if not ret:
#         break

#     # --------------------------------------------------------
#     # Convert to grayscale
#     # --------------------------------------------------------

#     grey = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2GRAY
#     )

#     h, w, _ = frame.shape

#     # --------------------------------------------------------
#     # Enhance frame for MediaPipe
#     # --------------------------------------------------------

#     sharpened_grey_frame = ut.get_transformed_image(grey)

#     # --------------------------------------------------------
#     # Convert grayscale -> BGR for MediaPipe
#     # --------------------------------------------------------

#     rgb = cv2.cvtColor(
#         sharpened_grey_frame,
#         cv2.COLOR_GRAY2BGR
#     )

#     # --------------------------------------------------------
#     # Face Mesh Detection
#     # --------------------------------------------------------

#     results = face_mesh.process(rgb)

#     if results.multi_face_landmarks:

#         for face_landmarks in results.multi_face_landmarks:

#             # =================================================
#             # GET ROIs
#             # =================================================

#             (
#                 top_left_coords,
#                 bottom_right_coords,
#                 top_right_coords,
#                 bottom_left_coords

#             ) = ut.get_eyes_coordinates(
#                 frame,
#                 grey,
#                 face_landmarks
#             )

#             (
#                 a,
#                 b,
#                 c,
#                 d

#             ) = ut.get_forehead_coordinates(
#                 frame,
#                 face_landmarks,
#                 flag
#             )

#             (
#                 x,
#                 y,
#                 z,
#                 w1

#             ) = ut.get_nose_coordinates(
#                 frame,
#                 face_landmarks
#             )

#             (
#                 pl,
#                 pr

#             ) = ut.get_cheeks_coordinates(
#                 frame,
#                 face_landmarks,
#                 points_left,
#                 points_right
#             )

#             # =================================================
#             # LEFT EYE ROI
#             # =================================================

#             left_eye_roi = grey[
#                 top_left_coords[1]:bottom_right_coords[1],
#                 top_left_coords[0]:bottom_right_coords[0]
#             ]

#             if left_eye_roi.size > 0:

#                 left_eye_mean = np.mean(left_eye_roi)

#                 left_eye_means.append(left_eye_mean)

#             # =================================================
#             # RIGHT EYE ROI
#             # =================================================

#             right_eye_roi = grey[
#                 top_right_coords[1]:bottom_left_coords[1],
#                 bottom_left_coords[0]:top_right_coords[0]
#             ]

#             if right_eye_roi.size > 0:

#                 right_eye_mean = np.mean(right_eye_roi)

#                 right_eye_means.append(right_eye_mean)

#             # =================================================
#             # FOREHEAD ROI
#             # =================================================

#             forehead_roi = grey[
#                 b:d,
#                 a:c
#             ]

#             if forehead_roi.size > 0:

#                 forehead_mean = np.mean(forehead_roi)

#                 forehead_means.append(forehead_mean)

#             # =================================================
#             # NOSE ROI
#             # =================================================

#             nose_roi = grey[
#                 y:w1,
#                 x:z
#             ]

#             if nose_roi.size > 0:

#                 nose_mean = np.mean(nose_roi)

#                 nose_means.append(nose_mean)

#             # =================================================
#             # LEFT CHEEK POLYGON ROI
#             # =================================================

#             polygon_left = np.array(
#                 pl,
#                 dtype=np.int32
#             )

#             mask_left = np.zeros(
#                 grey.shape,
#                 dtype=np.uint8
#             )

#             cv2.fillPoly(
#                 mask_left,
#                 [polygon_left],
#                 255
#             )

#             left_cheek_only = cv2.bitwise_and(
#                 grey,
#                 grey,
#                 mask=mask_left
#             )

#             left_pixels = left_cheek_only[
#                 mask_left == 255
#             ]

#             if left_pixels.size > 0:

#                 left_cheek_mean = np.mean(left_pixels)

#                 left_cheek_means.append(left_cheek_mean)

#             # =================================================
#             # RIGHT CHEEK POLYGON ROI
#             # =================================================

#             polygon_right = np.array(
#                 pr,
#                 dtype=np.int32
#             )

#             mask_right = np.zeros(
#                 grey.shape,
#                 dtype=np.uint8
#             )

#             cv2.fillPoly(
#                 mask_right,
#                 [polygon_right],
#                 255
#             )

#             right_cheek_only = cv2.bitwise_and(
#                 grey,
#                 grey,
#                 mask=mask_right
#             )

#             right_pixels = right_cheek_only[
#                 mask_right == 255
#             ]

#             if right_pixels.size > 0:

#                 right_cheek_mean = np.mean(right_pixels)

#                 right_cheek_means.append(right_cheek_mean)

#     # =========================================================
#     # DISPLAY FRAME
#     # =========================================================

#     cv2.imshow(
#         "Video Landmark Detection",
#         frame
#     )

#     # ESC KEY TO EXIT

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# # ============================================================
# # RELEASE RESOURCES
# # ============================================================

# cap.release()

# cv2.destroyAllWindows()

# # ============================================================
# # PLOT ALL SIGNALS
# # ============================================================

# plt.figure(figsize=(18, 7))

# plt.plot(
#     left_eye_means,
#     label="Left Eye"
# )

# plt.plot(
#     right_eye_means,
#     label="Right Eye"
# )

# plt.plot(
#     forehead_means,
#     label="Forehead"
# )

# plt.plot(
#     nose_means,
#     label="Nose"
# )

# plt.plot(
#     left_cheek_means,
#     label="Left Cheek"
# )

# plt.plot(
#     right_cheek_means,
#     label="Right Cheek"
# )

# plt.xlabel("Frame Number")

# plt.ylabel("Mean Pixel Intensity")

# plt.title("Mean Intensity Signals of Facial ROIs")

# plt.legend()

# plt.grid(True)

# plt.show()

############################################################################################################_______________________

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from forehead import utilities as ut

# ============================================================
# PIXEL TO TEMPERATURE CONVERSION
# ============================================================

def pixel_to_temperature(pixel_value):

    temperature = (
        0.05891454 * pixel_value
        + 30.07676744
    )

    return temperature


# ============================================================
# MEDIAPIPE FACE MESH SETUP
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ============================================================
# VARIABLES
# ============================================================

flag = False

# ============================================================
# LISTS FOR TEMPERATURE VALUES
# ============================================================

left_eye_temperatures = []
right_eye_temperatures = []

forehead_temperatures = []

nose_temperatures = []

left_cheek_temperatures = []
right_cheek_temperatures = []

# ============================================================
# INPUT VIDEO
# ============================================================

# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"
video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\yetnak_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Swamini\swamini_grey_manual.mpg"
# video_path = r"D:\Lie Detection Data HTI\Girish\girish_grey_manual.mpg"


cap = cv2.VideoCapture(video_path)

# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    points_left = []
    points_right = []

    ret, frame = cap.read()

    if not ret:
        break

    # ========================================================
    # CONVERT FRAME TO GRAYSCALE
    # ========================================================

    grey = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    h, w, _ = frame.shape

    # ========================================================
    # ENHANCE FRAME FOR MEDIAPIPE
    # ========================================================

    sharpened_grey_frame = ut.get_transformed_image(grey)

    # ========================================================
    # CONVERT TO BGR FOR MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        sharpened_grey_frame,
        cv2.COLOR_GRAY2BGR
    )

    # ========================================================
    # FACEMESH DETECTION
    # ========================================================

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:          # iterating over all the faces 

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

            # =================================================
            # LEFT EYE ROI
            # =================================================

            left_eye_roi = grey[
                top_left_coords[1]:bottom_right_coords[1],
                top_left_coords[0]:bottom_right_coords[0]
            ]

            # if left_eye_roi.size > 0:

            #     left_eye_mean_pixel = np.mean(left_eye_roi)

            #     left_eye_temperature = pixel_to_temperature(
            #         left_eye_mean_pixel
            #     )

            #     left_eye_temperatures.append(
            #         left_eye_temperature
            #     )

            if left_eye_roi.size > 0:
                pixel_percentage = 0.1
                left_eye_roi_flatten = left_eye_roi.flatten()
                left_eye_roi_sorted = np.sort(left_eye_roi_flatten)[::-1]
                left_eye_roi_top_pixels = left_eye_roi_sorted[0:int(len(left_eye_roi_sorted)*pixel_percentage)]
                left_eye_mean_pixel = np.mean(left_eye_roi_top_pixels)
                left_eye_temperature = pixel_to_temperature(
                    left_eye_mean_pixel
                )

                left_eye_temperatures.append(
                    left_eye_temperature
                )

            # =================================================
            # RIGHT EYE ROI
            # =================================================

            right_eye_roi = grey[
                top_right_coords[1]:bottom_left_coords[1],
                bottom_left_coords[0]:top_right_coords[0]
            ]

            if right_eye_roi.size > 0:
                pixel_percentage = 0.1
                right_eye_roi_flatten = right_eye_roi.flatten()
                right_eye_roi_sorted = np.sort(right_eye_roi_flatten)[::-1]
                right_eye_roi_top_pixels = right_eye_roi_sorted[0:int(len(right_eye_roi_sorted)*pixel_percentage)]
                right_eye_mean_pixel = np.mean(right_eye_roi_top_pixels)
                right_eye_temperature = pixel_to_temperature(
                    right_eye_mean_pixel
                )

                right_eye_temperatures.append(
                    right_eye_temperature
                )

            #     right_eye_mean_pixel = np.mean(right_eye_roi)

            #     right_eye_temperature = pixel_to_temperature(
            #         right_eye_mean_pixel
            #     )

            #     right_eye_temperatures.append(
            #         right_eye_temperature
            #     )



            # =================================================
            # FOREHEAD ROI
            # =================================================

            forehead_roi = grey[
                b:d,
                a:c
            ]

            if forehead_roi.size > 0:

                forehead_mean_pixel = np.mean(
                    forehead_roi
                )

                forehead_temperature = (
                    pixel_to_temperature(
                        forehead_mean_pixel
                    )
                )

                forehead_temperatures.append(
                    forehead_temperature
                )

            # =================================================
            # NOSE ROI
            # =================================================

            nose_roi = grey[
                y:w1,
                x:z
            ]

            if nose_roi.size > 0:

                nose_mean_pixel = np.mean(nose_roi)

                nose_temperature = pixel_to_temperature(
                    nose_mean_pixel
                )

                nose_temperatures.append(
                    nose_temperature
                )

            # =================================================
            # LEFT CHEEK POLYGON ROI
            # =================================================

            polygon_left = np.array(
                pl,
                dtype=np.int32
            )

            mask_left = np.zeros(
                grey.shape,
                dtype=np.uint8
            )

            cv2.fillPoly(
                mask_left,
                [polygon_left],
                255
            )

            left_cheek_only = cv2.bitwise_and(
                grey,
                grey,
                mask=mask_left
            )

            left_pixels = left_cheek_only[
                mask_left == 255
            ]

            if left_pixels.size > 0:

                left_cheek_mean_pixel = np.mean(
                    left_pixels
                )

                left_cheek_temperature = (
                    pixel_to_temperature(
                        left_cheek_mean_pixel
                    )
                )

                left_cheek_temperatures.append(
                    left_cheek_temperature
                )

            # =================================================
            # RIGHT CHEEK POLYGON ROI
            # =================================================

            polygon_right = np.array(
                pr,
                dtype=np.int32
            )

            mask_right = np.zeros(
                grey.shape,
                dtype=np.uint8
            )

            cv2.fillPoly(
                mask_right,
                [polygon_right],
                255
            )

            right_cheek_only = cv2.bitwise_and(
                grey,
                grey,
                mask=mask_right
            )

            right_pixels = right_cheek_only[
                mask_right == 255
            ]

            if right_pixels.size > 0:

                right_cheek_mean_pixel = np.mean(
                    right_pixels
                )

                right_cheek_temperature = (
                    pixel_to_temperature(
                        right_cheek_mean_pixel
                    )
                )

                right_cheek_temperatures.append(
                    right_cheek_temperature
                )

    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    cv2.imshow(
        "Thermal ROI Detection",
        frame
    )

    # ========================================================
    # PRESS ESC TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================================
# RELEASE RESOURCES
# ============================================================

cap.release()

cv2.destroyAllWindows()

# ============================================================
# PLOT TEMPERATURE SIGNALS
# ============================================================

plt.figure(figsize=(18, 8))

plt.plot(
    left_eye_temperatures,
    label="Left Eye"
)

plt.plot(
    right_eye_temperatures,
    label="Right Eye"
)

plt.plot(
    forehead_temperatures,
    label="Forehead"
)

plt.plot(
    nose_temperatures,
    label="Nose"
)

plt.plot(
    left_cheek_temperatures,
    label="Left Cheek"
)

plt.plot(
    right_cheek_temperatures,
    label="Right Cheek"
)

plt.xlabel("Frame Number")

plt.ylabel("Temperature (°C)")

plt.title(
    "Facial ROI Temperature Signals"
)

plt.legend()

plt.grid(True)

plt.show()