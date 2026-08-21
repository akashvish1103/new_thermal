# Removed the Jitter ROI and make it SMOOTH using EMA (Exponentail Moving Average)
# Using this as a Driver of the Utility of Scratch
# this file using all the ROI from uitlity code

import mediapipe as mp
import numpy as np
import cv2
import utilities as ut


# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40.wmv"
# video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\4.mp4"

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


# ============================================================
# EMA Landmark Smoother
# ============================================================
# Smooths every (x, y, z) of every landmark with an exponential
# moving average BEFORE any ROI is derived from them. This means
# every downstream ut.get_* function automatically inherits smooth,
# jitter-free coordinates -- no need to smooth each ROI separately.
#
#   smoothed = alpha * raw + (1 - alpha) * smoothed_prev
#
# alpha closer to 1.0 -> less smoothing, more responsive (more jitter)
# alpha closer to 0.0 -> more smoothing, more lag (less jitter)
# 0.3-0.5 is a good starting range for face landmarks at ~25-30 fps.

class LandmarkEMASmoother:
    def __init__(self, alpha=0.4, num_landmarks=478):
        self.alpha = alpha
        self.num_landmarks = num_landmarks
        self.prev = None  # np.ndarray shape (num_landmarks, 3)

    def smooth(self, face_landmarks):
        """
        face_landmarks: mediapipe NormalizedLandmarkList (results.multi_face_landmarks[i])
        Mutates face_landmarks in place with smoothed values and returns it,
        so it can be dropped straight into your existing ut.get_* calls unchanged.
        """
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float64
        )

        if self.prev is None or self.prev.shape != raw.shape:
            # first frame (or face count changed) -> no history yet
            self.prev = raw.copy()
        else:
            self.prev = self.alpha * raw + (1 - self.alpha) * self.prev

        for i, lm in enumerate(face_landmarks.landmark):
            lm.x = float(self.prev[i, 0])
            lm.y = float(self.prev[i, 1])
            lm.z = float(self.prev[i, 2])

        return face_landmarks

    def reset(self):
        """Call this if the face is lost for a frame or more, so the EMA
        doesn't try to interpolate across a gap / a different face."""
        self.prev = None


landmark_smoother = LandmarkEMASmoother(alpha=0.4)


cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformed_grey = ut.get_transformed_image(grey)

    #     # -----------------------------------------
    #     # ROTATE 90° RIGHT
    #     # -----------------------------------------
    # transformed_grey = cv2.rotate(
    #         transformed_grey,
    #         cv2.ROTATE_90_CLOCKWISE
    #     )

    rgb = cv2.cvtColor(transformed_grey, cv2.COLOR_GRAY2BGR)  # converting single-channel image to 3-channel image, becasue mediapie expects a RGB (3-channel) image

    results = face_mesh.process(rgb)                          # Processing the RGB image

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:              # loop for each face found in the video

           # smooth the raw landmarks BEFORE deriving any ROI from them
           face_landmarks = landmark_smoother.smooth(face_landmarks)

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
        #    cv2.circle(frame, top_left_cords, 5, (255, 255, 255), -5)   # trying to display DOT on original rgb frame, by getting the
                                                                         # coordinates from the utilities function.
    else:
        # face lost this frame -> reset EMA so it doesn't blend garbage
        # history into the next detection
        landmark_smoother.reset()

    cv2.imshow("Transformed Grey", got_frame)    #just displaying the modified got_frame, which has been processed by these modular python FUNCTIONS()
    cv2.imshow("RGB framea", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()