# import cv2
# import mediapipe as mp
# import utilities as ut

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

# # FOREHEAD_POINTS = {
# #     top_left : 67,
# #     bottom_left : 105,
# #     top_right : 297,
# #     bottom_right : 334
# #     }

# # definig the function to fraw rectangle on forehead
# # def draw_forehead_rectangle(frame, landmarks, w, h):
# # ============================================================
# # FOREHEAD LANDMARK INDICES
# # ============================================================
# # These are upper-face / forehead-near landmarks

# FOREHEAD_POINTS = [
#     67,
    
#     # 103,
#     # 109,
#     # 338,
#     297,
#     # 332,
#     105, 334,
#     # 9,10
# ]

# # FOREHEAD_POINTS = [
# #     10,
# #     151,
# #     9,
# #     107,
# #     66,
# #     105,
# #     103,
# #     109,
# #     338,
# #     332,
# #     334,
# #     296,
# #     297
# # ]

# # ============================================================
# video_path = rvideo_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# # video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# # video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# cap = cv2.VideoCapture(video_path)



# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     h, w, _ = frame.shape
#     transformed_frame = ut.get_transformed_image(grey_frame)



#     # Converting BGR → RGB for MediaPipe processing (becoz it expects 3-channel RGB input)
#     rgb = cv2.cvtColor(transformed_frame, cv2.COLOR_GRAY2BGR)

  
#     # LANDMARK DETECTION
#     results = face_mesh.process(rgb)
#     # cords_dict = {}    # Dictionary to store the coordinates of forehead landmarks with their indices as keys

#     # DRAW FOREHEAD LANDMARKS
#     if results.multi_face_landmarks:

#         for face_landmarks in results.multi_face_landmarks:

#             forehead_coords = []

#             for idx in FOREHEAD_POINTS:
   
#                 landmark = face_landmarks.landmark[idx]

#                 x = int(landmark.x * w)
#                 y = int(landmark.y * h)

#                 forehead_coords.append((x, y))    # Order of coordinates is 67, 297, 105, 334

#                  # -----------------------------------------
#                 # SAFETY CHECK
#                 # -----------------------------------------
#                 if len(forehead_coords) != 4:
#                     continue

#                 # Draw point
#                 cv2.circle(
#                     transformed_frame,
#                     (x, y),
#                     4,
#                     (0, 255, 0),
#                     -1
#                 )

#                 # Display landmark index
#                 cv2.putText(
#                     transformed_frame,
#                     str(idx),
#                     (x + 5, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0,255,0), 
#                     1
#                 )
           
#             # Extracting the forehead Bounding Box Coordinates
#             left_margin =max(forehead_coords[0][0], forehead_coords[2][0])
#             right_margin =min(forehead_coords[1][0], forehead_coords[3][0])

#             top_margin = ( max(forehead_coords[0][1], forehead_coords[1][1]))
#             bottom_margin = ( min(forehead_coords[2][1], forehead_coords[3][1]))


#              # Draw rectangle
#             cv2.rectangle(
#                 transformed_frame,
#                 (left_margin, top_margin),
#                 (right_margin, bottom_margin),
#                 (255,255,255),
#                 2
#             )

#             # =====================================================
#             # CROP FOREHEAD
#             # =====================================================

#             forehead_crop = transformed_frame[
#                 top_margin:bottom_margin,
#                 left_margin:right_margin
#             ]

#             # Show cropped forehead
#             cv2.imshow("Forehead Crop", forehead_crop)           

#     # ========================================================
#     # SHOW FRAME
#     # ========================================================
#     cv2.imshow("Forehead Landmarks", transformed_frame)

#     # ESC to exit
#     if cv2.waitKey(1) & 0xFF == 27:
#         break


# # ============================================================
# # RELEASE
# # ============================================================
# cap.release()
# cv2.destroyAllWindows()

####################################################################################
####################################################################################
# This code will be used in final version.
# Implementing Forhead ROI tracking instread of CSRT Tracker to make the code faster.

# In tis code, we are detecting the forehead region using MediaPipe Face Mesh and 
# cropping it out , using 4 landmarks (67, 297, 105, 334) which are located near the forehead. 
# We are also drawing a rectangle around the detected forehead region and displaying the cropped forehead in a separate window.


import cv2
import mediapipe as mp
import utilities as ut

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

# ============================================================
# FOREHEAD LANDMARKS
# Order:
# 67  -> top-left
# 297 -> top-right
# 105 -> bottom-left
# 334 -> bottom-right
# ============================================================

FOREHEAD_POINTS = [
    67,
    297,
    105,
    334
]

# ============================================================
# VIDEO PATH
# ============================================================

video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = rvideo_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = rvideo_path = r"D:\Lie Detection Data HTI\Girish\girish_grey_manual.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"


cap = cv2.VideoCapture(video_path)

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Apply thermal transformation
    # --------------------------------------------------------

    transformed_frame = ut.get_transformed_image(grey_frame)

    # --------------------------------------------------------
    # Get height and width
    # IMPORTANT:
    # Use transformed frame dimensions
    # --------------------------------------------------------

    h, w = transformed_frame.shape

    # --------------------------------------------------------
    # Convert grayscale -> BGR for MediaPipe
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        transformed_frame,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Landmark detection
    # --------------------------------------------------------

    results = face_mesh.process(rgb)

    # ========================================================
    # DRAW LANDMARKS + FOREHEAD ROI
    # ========================================================

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            forehead_coords = []

            # ------------------------------------------------
            # Extract landmark coordinates
            # ------------------------------------------------

            for idx in FOREHEAD_POINTS:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                forehead_coords.append((x, y))

                # --------------------------------------------
                # Draw landmark point
                # --------------------------------------------

                cv2.circle(
                    transformed_frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

                # --------------------------------------------
                # Draw landmark index
                # --------------------------------------------

                cv2.putText(
                    transformed_frame,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if len(forehead_coords) != 4:
                continue

            # =================================================
            # FOREHEAD BOUNDING BOX
            # =================================================

            left_margin = max(
                forehead_coords[0][0],
                forehead_coords[2][0]
            )

            right_margin = min(
                forehead_coords[1][0],
                forehead_coords[3][0]
            )

            top_margin = max(
                forehead_coords[0][1],
                forehead_coords[1][1]
            )

            bottom_margin = min(
                forehead_coords[2][1],
                forehead_coords[3][1]
            )

            # ------------------------------------------------
            # Clamp coordinates inside image
            # ------------------------------------------------

            left_margin = max(0, left_margin)
            top_margin = max(0, top_margin)

            right_margin = min(w, right_margin)
            bottom_margin = min(h, bottom_margin)

            # ------------------------------------------------
            # Rectangle validity checks
            # ------------------------------------------------

            if left_margin >= right_margin:
                continue

            if top_margin >= bottom_margin:
                continue

            # =================================================
            # DRAW RECTANGLE
            # =================================================
            adjusted_bottom = bottom_margin - 2                  # added a small buffer to avoid including the eyebrow region
            cv2.rectangle(
                transformed_frame,
                (left_margin, top_margin),
                (right_margin, adjusted_bottom),
                (255, 255, 255),
                2
            )

            cv2.rectangle(
                grey_frame,
                (left_margin, top_margin),
                (right_margin, adjusted_bottom),
                (255, 255, 255),
                2
            )

            # =================================================
            # CROP FOREHEAD ROI
            # =================================================
            
            # Cropping the forehead region from the enhanced frame
            forehead_crop = transformed_frame[
                top_margin:adjusted_bottom,          
                left_margin:right_margin
            ]

            # Cropping the forehead region from the original grey frame 
            forehead_crop_grey = grey_frame[
                top_margin:adjusted_bottom,          
                left_margin:right_margin
            ]
            # =================================================
            # SHOW CROPPED FOREHEAD
            # =================================================

            if forehead_crop.size != 0:
                
                # showing the cropped forehead from the enhanced frame
                cv2.imshow(
                    "Forehead Crop",
                    forehead_crop
                )
                
                # showing    Bounding rectangle from the original frame
                cv2.imshow(
                    "Forehead Crop Original Grey",
                    forehead_crop_grey
                )


    # ========================================================
    # SHOW MAIN FRAME
    # ========================================================
     # For the transformed frame
    cv2.imshow(
        "Forehead Landmarks",
        transformed_frame
    )
     
    # For Original Frame
    cv2.imshow(
        "Original Forehead Landmarks",
        grey_frame
    )

    # ========================================================
    # ESC TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================================
# RELEASE
# ============================================================

cap.release()
cv2.destroyAllWindows()