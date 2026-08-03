# # this custom module will have all the intermediate procesing stuffs required.


# import cv2
# import numpy as np
# # ============================================================
# # CLAHE (Create Once)
# # ============================================================
# clahe = cv2.createCLAHE(
#     clipLimit=2.0,
#     tileGridSize=(8, 8)
# )

# def get_transformed_image(grey_frame):
#     """
#     Applies contrast enhancement pipeline to the input grayscale frame and returns the enhanced greyscaled frame.
#     """

#     # ========================================================
#     # CONTRAST ENHANCEMENT PIPELINE
#     # ========================================================

#     # Stretch contrast
#     stretched = cv2.normalize(
#         grey_frame,
#         None,
#         0,
#         255,
#         cv2.NORM_MINMAX
#     )

#     # CLAHE
#     enhanced = clahe.apply(stretched)

#     # Gamma correction
#     gamma_corrected = (
#         np.power(enhanced / 255.0, 0.6) * 255
#     ).astype(np.uint8)

#     # Unsharp mask
#     blurred = cv2.GaussianBlur(
#         gamma_corrected,
#         (0, 0),
#         3
#     )

#     # sharpened = cv2.addWeighted(
#     #     gamma_corrected,
#     #     1.5,
#     #     blurred,
#     #     -0.5,
#     #     0
#     # )

#     sharpened_grey_frame = cv2.addWeighted(                        # sharpened the edges
#     gamma_corrected,
#     2,
#     blurred,
#     -1.5,
#     0
#    )
    
#     return sharpened_grey_frame                                  # returning the enhanced greyscaled frame

##################################### now above code si of nose folder, and below code is of FOREHEAD flder

# this custom module will have all the intermediate procesing  required.
# This is Defined inside inside FOREHEAF folder.

import cv2
import numpy as np
# ============================================================
# CLAHE (Create Once)
# ============================================================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

def get_transformed_image(grey_frame):
    """
    Applies contrast enhancement pipeline to the input grayscale frame and returns the enhanced greyscaled frame.
    """

    # ========================================================
    # CONTRAST ENHANCEMENT PIPELINE
    # ========================================================

    # Stretch contrast
    stretched = cv2.normalize(
        grey_frame,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # CLAHE
    enhanced = clahe.apply(stretched)

    # Gamma correction
    gamma_corrected = (
        np.power(enhanced / 255.0, 0.6) * 255
    ).astype(np.uint8)

    # Unsharp mask
    blurred = cv2.GaussianBlur(
        gamma_corrected,
        (0, 0),
        3
    )

    # sharpened = cv2.addWeighted(
    #     gamma_corrected,
    #     1.5,
    #     blurred,
    #     -0.5,
    #     0
    # )

    sharpened_grey_frame = cv2.addWeighted(                        # sharpened the edges
    gamma_corrected,
    2,
    blurred,
    -1.5,
    0
   )
    
    return sharpened_grey_frame                                  # returning the enhanced greyscaled frame



################################################################################



def get_eyes_coordinates(frame, grey, face_landmarks):
            
            import mediapipe as mp
            import cv2

            # Inner eye corner landmarks
            LEFT_INNER_EYE = 133                                      # mediapipe landmark for left eye inner corner
            RIGHT_INNER_EYE = 362                                     # mediapipe landmark for right eye inner corner
                               
            PERCENTAGE_PIXEL_TO_KEEP = 0.80
            h, w, _ = frame.shape                           # mediapipe landmark for right eye inner corner

            # LEFT INNER EYE   --->   lx is the x-coordinate of inner corner for left eye and ly is the y coordinate of inner corner of left eye
            left_point = face_landmarks.landmark[LEFT_INNER_EYE]                     
            lx = int(left_point.x * w)
            ly = int(left_point.y * h)

            # RIGHT INNER EYE  --->  rx is the x-coordinate of inner corner for right eye and ry is the y coordinate of inner corner of right eye
            right_point = face_landmarks.landmark[RIGHT_INNER_EYE]
            rx = int(right_point.x * w)
            ry = int(right_point.y * h)
 
            # For LEFT INNER EYE
            top_left_coords = (lx, ly-10)
            bottom_right_coords = (lx+20, ly+10)
  
            # For RIGHT INNER EYE   
            top_right_coords = (rx, ry-10)
            bottom_left_coords = (rx-20, ry+10)

            # CV2.CIRCLE is modifyng the image array (Just changing, not DISPLAYING)
            # displaying the top left and bottom right corner of the box around inner eye corners LEFT EYE
            cv2.circle(frame, top_left_coords, 2, (0, 255, 0), -5)    #dot above left inner eye corner
            cv2.circle(frame, bottom_right_coords, 2, (0, 255, 0), -5)    #dot right to left inner eye corner

            # displaying the top left and bottom right corner of the box around inner eye corners RIGHT EYE
            cv2.circle(frame, top_right_coords, 2, (0, 255, 0), -5)    #dot above left inner eye corner
            cv2.circle(frame, bottom_left_coords, 2, (0, 255, 0), -5)    #dot right to left inner eye corner

            cv2.rectangle(frame, top_left_coords, bottom_right_coords, (255, 0, 0), 2)    #rectangle around left inner eye corner
            cv2.rectangle(frame, top_right_coords, bottom_left_coords, (255, 0, 0), 2)    #rectangle around right inner eye corner


            return top_left_coords, bottom_right_coords, top_right_coords, bottom_left_coords

#######################################

def get_forehead_coordinates(frame, face_landmarks, flag):

    FOREHEAD_POINTS = [
    67,
    297,
    105,
    334
]
    
    forehead_coords = []
    h, w, _ = frame.shape  

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
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

                # --------------------------------------------
                # Draw landmark index
                # --------------------------------------------

                cv2.putText(
                    frame,
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
                   flag = True

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
                    flag = True

    if top_margin >= bottom_margin:
                    flag = True

                # =================================================
                # DRAW RECTANGLE
                # =================================================
    adjusted_bottom = bottom_margin - 2                  # added a small buffer to excluding the eyebrow region

    # CV2.RECTANGLE is modifyng the image array (Just changing, not DISPLAYING)
    cv2.rectangle(
                    frame,
                    (left_margin, top_margin),
                    (right_margin, adjusted_bottom),
                    (255, 255, 255),
                    2
                )


    return left_margin, top_margin, right_margin, adjusted_bottom

#########################################################################################
# NOSE


def get_nose_coordinates(frame, face_landmarks):
            NOSE_LANDMARKS = [19] 
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

                # nose_tip_crop = transforemd_frame[top_margin:bottom_margin, left_margin:right_margin]
                
                # CV2.RECTANGLE is modifyng the image array (Just changing, not DISPLAYING)
                cv2.rectangle(frame, (left_margin, top_margin), (right_margin, bottom_margin), (100, 255, 150), 2)

                # Draw a red dot on the nose tip , # CV2.CIRCLE is modifyng the image array (Just changing, not DISPLAYING)
                cv2.circle(frame, (nose_x, nose_y), 3, (0, 0, 255), -1)

                return left_margin, top_margin, right_margin, bottom_margin
            

##############################################################################################

def get_cheeks_coordinates(frame, face_landmarks, points_left, points_right):

    h, w, _ = frame.shape  
    LEFT_CHEEK = [
                    214, 216, 206, 120, 101, 50, 187
                ]

    RIGHT_CHEEK = [
                    432, 436, 426, 349, 330, 280, 411
                ]
    
    ALL_CHEEKS = LEFT_CHEEK + RIGHT_CHEEK

    for idx in ALL_CHEEKS:       #face_landmarks contains the normalized landmark coordinates in a python list,  we can exess like - face_landmarks.landmark[214] to get the normalized coordinates of landmark 214.

                lm = face_landmarks.landmark[idx]

                # converting landmark into real cordinates
                x = int(lm.x * w)                    
                y = int(lm.y * h)

                if idx in LEFT_CHEEK:
                    points_left.append((x, y))
                elif idx in RIGHT_CHEEK:
                    points_right.append((x, y))


                # CV2.CIRCLE is modifyng the image array (Just changing, not DISPLAYING)
                # Draw point
                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

                # Draw index label
                cv2.putText(
                    frame,
                    str(idx),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0,0,255),
                    1
                )

    polygon_left = np.array(points_left, dtype=np.int32)
    polygon_right = np.array(points_right, dtype=np.int32)
                # -----------------------------
                # Create empty mask
                # -----------------------------

            # Just Drawing the edges of the closed cure, i.e ploygon 
    cv2.polylines(
                        frame,
                        [polygon_left],
                        isClosed=True,
                        color=(0,255,0),
                        thickness=1
                    )
    cv2.polylines(
                        frame,
                        [polygon_right],
                        isClosed=True,
                        color=(0,255,0),
                        thickness=1
                    )
            
    # mask_left = np.zeros(frame.shape, dtype=np.uint8)     # making a ndarray of zeros with the same shape as frame, this will be used as a mask to  extract the polygon region
    # mask_right = np.zeros(frame.shape, dtype=np.uint8)

    #             # Fill polygon region with white
    # cv2.fillPoly(mask_left, [polygon_left], 255)           # this line will modify mask to have the polygon region filled with 255 and remaining 0
    # cv2.fillPoly(mask_right, [polygon_right], 255)

    #             # -----------------------------
    #             # Keep only polygon pixels
    #             # -----------------------------
    # result_left = cv2.bitwise_and(frame, mask_left)
    # result_right = cv2.bitwise_and(frame, mask_right) 

    return points_left, points_right      # there are the lsit of corrdinates (tuple) arrying 7 (x,y) coordinates for each of 7 landmarks for left and right cheek.