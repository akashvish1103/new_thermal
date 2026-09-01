"""
# Facial ROI Processing and Image Enhancement Utilities

This custom module contains utility functions used for facial image
preprocessing and region-of-interest (ROI) extraction in the SCRATCH folder.

The module provides functionality for:
- Grayscale image contrast enhancement using normalization, CLAHE,
  gamma correction, and unsharp masking.
- Extracting facial coordinates using MediaPipe Face Mesh landmarks.
- Locating the inner eye corners, nose tip, cheeks, forehead, and
  breathing-related nasal ROI.
- Creating and visualizing facial ROI boundaries and polygons.
- Calculating mean pixel intensity values from selected facial ROIs.

The module is designed to be imported by the main analysis scripts,
where MediaPipe face detection/landmark processing is performed.
"""


# this custom module will have all the intermediate procesing  required.
# This is Defined inside inside SCRATCH folder of root folder "scj_analysis"

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
    2,                                                    # Make it larger for more sharpened image
    blurred,
    -1.5,
    0
   )
    
    return sharpened_grey_frame                                  # returning the enhanced greyscaled frame



###########################################################################################################


def get_mp_setup():
        """
        This fn will contain basic MediaPipe Set ups
        """
        pass


#######################################################################################################################


def get_eyes_coordinates(grey_frame, face_landmarks): 
            """
            This function takes a greyscaled frame and face_landmarks of already processed rgb image(in main python file) of mediapipe face mesh
            ["for face_landmarks in results.multi_face_landmarks: "]

            MediaPipe FaceMesh Hierarchy:

            results
            │
            └── multi_face_landmarks              # List of detected faces
                │
                ├── face_landmarks              # One NormalizedLandmarkList (one face)
                │      │
                │      ├── landmark[0]          # First facial landmark
                │      │      ├── x             # Normalized x-coordinate (0–1)
                │      │      ├── y             # Normalized y-coordinate (0–1)
                │      │      └── z             # Relative depth
                │      │
                │      ├── landmark[1]
                │      ├── landmark[2]
                │      ├── ...
                │      └── landmark[477]        # Last landmark (478 total with refine_landmarks=True)
                │
                ├── face_landmarks              # Second face (if detected)
                │      └── landmark...
                │
                └── ...

            The greyscaled image will be just used to get the height and width of the image, and get the cordinates of the eyes
            corners, and we are modifing this greyscaled image using cv2.circle and returning the modified image with the
            coordinates of the eyes corners. --> (x1,y1), (x2,y2), (x3,y3), (x4,y4)
            Note that there is not code for displaying the image or any cv2.imshow() function. 

            """
            
            import mediapipe as mp
            import cv2

            # Inner eye corner landmarks
            LEFT_INNER_EYE = 133                                      # mediapipe landmark for left eye inner corner
            RIGHT_INNER_EYE = 362                                     # mediapipe landmark for right eye inner corner
                               
            PERCENTAGE_PIXEL_TO_KEEP = 0.80
            h, w = grey_frame.shape                           

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
            cv2.circle(grey_frame, top_left_coords, 2, (0, 255, 0), -5)    #dot above left inner eye corner
            cv2.circle(grey_frame, bottom_right_coords, 2, (0, 255, 0), -5)    #dot right to left inner eye corner

            # displaying the top left and bottom right corner of the box around inner eye corners RIGHT EYE
            cv2.circle(grey_frame, top_right_coords, 2, (0, 255, 0), -5)    #dot above left inner eye corner
            cv2.circle(grey_frame, bottom_left_coords, 2, (0, 255, 0), -5)    #dot right to left inner eye corner

            cv2.rectangle(grey_frame, top_left_coords, bottom_right_coords, (255, 0, 0), 2)    #rectangle around left inner eye corner
            cv2.rectangle(grey_frame, top_right_coords, bottom_left_coords, (255, 0, 0), 2)    #rectangle around right inner eye corner


            return top_left_coords, bottom_right_coords, top_right_coords, bottom_left_coords, grey_frame   # --> (x1,y1), (x2,y2), (x3,y3), (x4,y4)

#######################################
 
#OLD FUNCTION (Depreciated!!) (We will not going to use this Funtion)
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


def get_nose_tip_coordinates(grey_frame, face_landmarks):
            """
            This function takes a greyscaled frame and face_landmarks of already processed rgb image(in main python file) of mediapipe face mesh
            ["for face_landmarks in results.multi_face_landmarks: "]
            The greyscaled image will be just used to get the height and width of the image, and get the cordinates of the eyes
            corners, and we are modifing this greyscaled image using cv2.circle and returning the modified image with the 
            coordinates of the eyes corners.
            Note that ther is not code for displaying the image or any cv2.imshow() function. 

            
        
            """
            NOSE_LANDMARKS = [19] 
            for idx in NOSE_LANDMARKS:
                # Get the coordinates of the nose tip (landmark index 1)
                nose_tip = face_landmarks.landmark[idx]
                h, w = grey_frame.shape
                nose_x = int(nose_tip.x * w)                 # It is the x-coordinate of the nose tip in pixels
                nose_y = int(nose_tip.y * h)                 # It is the y-coordinate of the nose tip in pixels

                left_margin = nose_x - 10                   # adding margin to the left and right of the nose tip
                right_margin = nose_x + 10

                top_margin = nose_y - 20                   # adding margin to the top and bottom of the nose tip
                bottom_margin = nose_y - 3

                # nose_tip_crop = transforemd_frame[top_margin:bottom_margin, left_margin:right_margin]
                
                # CV2.RECTANGLE is modifyng the image array (Just changing, not DISPLAYING)
                cv2.rectangle(grey_frame, (left_margin, top_margin), (right_margin, bottom_margin), (0, 255, 255), 2)

                # Draw a red dot on the nose tip , # CV2.CIRCLE is modifyng the image array (Just changing, not DISPLAYING)
                cv2.circle(grey_frame, (nose_x, nose_y), 3, (0, 0, 255), -1)

                top_left_coords = (left_margin, top_margin)                   # Cordinates of the top left corner of the rectangle around the nose tip
                bottom_right_coords = (right_margin, bottom_margin)           # Cordinates of the bottom right corner of the rectangle around the nose tip

                return top_left_coords, bottom_right_coords, grey_frame
            

##############################################################################################

def get_cheeks_coordinates(frame, face_landmarks, points_left_lst, points_right_lst):
    """
    This Function will return the coordinates of the both right cheek polygon and left cheek polygon.
    There are the lsit of corrdinates (tuple) arrying 7 (x,y) coordinates for each of 7 landmarks for left and right cheek.
    [(103, 308), (105, 289), (106, 278), (107, 238), (98, 252), (86, 265), (88, 283)]  AND 
    [(207, 274), (198, 259), (188, 252), (168, 217), (184, 224), (206, 227), (219, 242)]

    """

    h, w= frame.shape  
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
                    points_left_lst.append((x, y))
                elif idx in RIGHT_CHEEK:
                    points_right_lst.append((x, y))


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

    polygon_left = np.array(points_left_lst, dtype=np.int32)
    polygon_right = np.array(points_right_lst, dtype=np.int32)
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

    return points_left_lst, points_right_lst, frame      # there are the lsit of corrdinates (tuple) arrying 7 (x,y) coordinates for each of 7 landmarks for left and right cheek.

#############################################################################################




def get_breathing_roi_cords(grey_frame, face_landmarks):
    """
    This function will return the top right and bottom left coordinates of the Breathing Box ROI --> (x1,y1), (x2,y2) and a modified grey_frame
    """

    h,w = grey_frame.shape

    UPPER_LIPS_LANDMARK       = [13, 206, 426]
    NOSE_HORIZONTAL_LANDMARKS = [64, 278]
    NOSE_VERTICAL_LANDMARKS   = [4, 94]
    NOSE_LANDMARKS = NOSE_HORIZONTAL_LANDMARKS + NOSE_VERTICAL_LANDMARKS + UPPER_LIPS_LANDMARK

    LEFT_HORIZONTAL_LANDMARK  = 64
    RIGHT_HORIZONTAL_LANDMARK = 278
    UP_VERTICAL_LANDMARK      = 4
    DOWN_VERTICAL_LANDMARK    = 94

    for idx in NOSE_LANDMARKS:
                nose_tip = face_landmarks.landmark[idx]
                # h, w, _ = grey_frame.shape
                nose_x = int(nose_tip.x * w)
                nose_y = int(nose_tip.y * h)

                if idx == LEFT_HORIZONTAL_LANDMARK:
                    left_margin = nose_x
                if idx == RIGHT_HORIZONTAL_LANDMARK:
                    right_margin = nose_x
                if idx == UP_VERTICAL_LANDMARK:
                    top_margin = nose_y
                if idx == DOWN_VERTICAL_LANDMARK:
                    height        = nose_y - top_margin
                    bottom_margin = nose_y + int(1.15 * height)

    dist         = right_margin - left_margin
    left_margin  = left_margin  - int(dist * 0.1)
    right_margin = right_margin + int(dist * 0.1)

    nose_crop  = grey_frame[top_margin:bottom_margin, left_margin:right_margin]
    mean_value = np.mean(nose_crop)
    # lst.append(mean_value)
    # lst_temp.append(round(get_temp_from_pixel(mean_value), 3))

    cv2.rectangle(grey_frame, (left_margin, top_margin),
                          (right_margin, bottom_margin), (255, 255, 255), 1)

    cv2.circle(grey_frame, (left_margin, top_margin), 2, (0, 0, 0), -5)
    cv2.circle(grey_frame, (right_margin, bottom_margin), 2, (0, 0, 0), -5) 

    top_left_cords = (left_margin, top_margin)              # Coordinates of the top left point of the ROI Box that will be used for breathing pattern
    bottom_right_cords = (right_margin, bottom_margin)      # Coordinates of the bottom point of the ROI Box that will be used for breathing pattern

    return top_left_cords, bottom_right_cords, grey_frame



#####################################################################################################################


def get_forhead_poly_coords(grey_frame, face_landmarks):     # grey_frame need to be br RGB, becasue there is no face detection(mp.process(rgb_image))) here, it alpready happened in the main python file
    """
    This function takes the grey_frame, and face_landmark of mp, and spits out :

    1. polygon_points --> a List containing [x,y] coordinates of all  the Forehead polygon points.
    Function is returning "polygon_points" which are (x, y) coordinates of all the polygon points like
    [ [245, 98], [260, 95], [280, 92], [315, 94], [340, 99],...]

    2. mean_pixel --> Mean of the pixel intensities(values) of the polygon ROI.

    3. grey_frame --> Modified grey_frame, containing the circle or rectangle or polyLines.(It just need "cv2.imshow()" to display.)
    """

    # ----------------------------
    # Forehead Landmarks                       # These are all the landmarks of the Entire forehead, we are not using them rn, but can use it in future..
    # ----------------------------
    # FOREHEAD_LANDMARKS = [
    #     10,
    #     67,
    #     69,
    #     103,
    #     104,
    #     105,
    #     107,
    #     108,
    #     109,
    #     151,
    #     297,
    #     299,
    #     332,
    #     333,
    #     334,
    #     336,
    #     337,
    #     338
    # ]


    # ----------------------------
    # Forehead ROI Polygon
    # ----------------------------
    FOREHEAD_POLYGON = [
        67,
        109,
        10,
        338,
        297,
        333,
        334,
        107,
        104,
    ]

    h,w = grey_frame.shape
    polygon_points = []

    for idx in FOREHEAD_POLYGON:

                lm = face_landmarks.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                polygon_points.append([x, y])

    polygon_points = np.array(polygon_points, dtype=np.int32)

            # ----------------------------
            # Draw Polygon
            # ----------------------------
    cv2.polylines(
                grey_frame,
                [polygon_points],
                True,
                (255, 0, 0),
                2
            )

    cv2.polylines(
                grey_frame,
                [polygon_points],
                True,
                255,
                2
            )

    # =====================================================
    # Create Mask
    # =====================================================
    mask = np.zeros(grey_frame.shape, dtype=np.uint8)

    cv2.fillPoly(mask, [polygon_points], 255)

    # =====================================================
    # Extract ROI
    # =====================================================

    forehead_roi = cv2.bitwise_and(grey_frame, grey_frame, mask=mask)   # Extracted numpy image of the Forehead Polygon

    # =====================================================
    # Average Pixel Value
    # =====================================================
    mean_pixel = cv2.mean(grey_frame, mask=mask)[0]

    print(f"Average Pixel Value = {mean_pixel:.2f}")

    return polygon_points, mean_pixel, grey_frame        # List of lists of the coordinates of all polygon  


#########################################################################################################################


# Function to get mean temperature for the given numpy array(roi_image single channel image in this case)
def get_mean_of_ROI(roi_numpy_array):

    return round(np.mean(roi_numpy_array), 2)


########################################################################################################################################



        