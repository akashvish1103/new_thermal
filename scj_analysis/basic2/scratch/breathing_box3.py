# This Code works fine, Time Duration issue solved...
# Solved jitter ROI issue


import os
import subprocess
import tempfile

import mediapipe as mp
import numpy as np
import cv2

import utilities as ut


# ============================================================
# VIDEO PATH
# ============================================================

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_30_40.mpg"

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40.wmv"

# video_path = r"D:\2026-08-20_Thermal_Meditation\TopInfrared\4.mp4"

video_path = r"D:\000_ofc_thermalData\sorted_data\48\48_HDRS_Thermal_30_40.mpg"
video_path = r"D:\000_ofc_thermalData\sorted_data\48\48_Passive_Thermal_30_40.mpg"


# ============================================================
# FFmpeg conversion settings
# ============================================================

# FFmpeg must be available from the command line.
#
# You already verified:
#
#     ffmpeg -version
#
# works on your system.

FFMPEG_EXE = r"C:\ffmpeg-2026-08-20-git-7d77562d2a-essentials_build\bin\ffmpeg.exe"


# ============================================================
# Convert MPG to temporary MP4
# ============================================================

def convert_to_mp4(input_video):

    """
    Convert the input video to a temporary MP4 using FFmpeg.

    The original video is NOT modified.

    Returns:
        path to temporary MP4
    """

    # --------------------------------------------------------
    # Check that input exists
    # --------------------------------------------------------

    if not os.path.isfile(input_video):

        raise FileNotFoundError(
            f"Input video does not exist:\n{input_video}"
        )


    # --------------------------------------------------------
    # Create temporary MP4 filename
    #
    # delete=False because OpenCV needs to open the file after
    # FFmpeg finishes creating it.
    # --------------------------------------------------------

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )

    temp_mp4 = temp_file.name

    temp_file.close()


    # --------------------------------------------------------
    # FFmpeg command
    #
    # -y
    #     overwrite temporary file if necessary
    #
    # -i
    #     input video
    #
    # -c:v libx264
    #     encode video as H.264
    #
    # -pix_fmt yuv420p
    #     standard pixel format supported well by OpenCV
    #
    # -vsync 0
    #     do not duplicate/drop frames unnecessarily
    #
    # -an
    #     no audio; your analysis is video-only
    #
    # --------------------------------------------------------

    command = [
    FFMPEG_EXE,
    "-y",
    "-i",
    input_video,
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-an",
    temp_mp4
]

    print("=" * 70)
    print("CONVERTING INPUT VIDEO USING FFMPEG")
    print("=" * 70)

    print("Input:")
    print(input_video)

    print()
    print("Temporary MP4:")
    print(temp_mp4)

    print("=" * 70)


    # --------------------------------------------------------
    # Run FFmpeg
    #
    # stdout/stderr are inherited so FFmpeg progress/errors
    # can be seen in the console.
    # --------------------------------------------------------

    try:

        subprocess.run(
            command,
            check=True
        )

    except FileNotFoundError:

        # ----------------------------------------------------
        # FFmpeg itself could not be found
        # ----------------------------------------------------

        if os.path.exists(temp_mp4):

            os.remove(temp_mp4)

        raise RuntimeError(
            "FFmpeg was not found.\n\n"
            "Open a new Command Prompt and verify:\n\n"
            "    ffmpeg -version\n\n"
            "FFmpeg must be available in PATH."
        )

    except subprocess.CalledProcessError as e:

        # ----------------------------------------------------
        # FFmpeg returned an error
        # ----------------------------------------------------

        if os.path.exists(temp_mp4):

            os.remove(temp_mp4)

        raise RuntimeError(
            f"FFmpeg conversion failed with exit code "
            f"{e.returncode}."
        )


    # --------------------------------------------------------
    # Verify that the output file actually exists
    # --------------------------------------------------------

    if not os.path.isfile(temp_mp4):

        raise RuntimeError(
            "FFmpeg finished, but the temporary MP4 "
            "was not created."
        )


    print()
    print("FFmpeg conversion completed successfully.")
    print("=" * 70)


    return temp_mp4


# ============================================================
# Determine whether conversion is necessary
# ============================================================

input_extension = os.path.splitext(
    video_path
)[1].lower()


temporary_video = None


if input_extension in [
    ".mpg",
    ".mpeg",
    ".mpv"
]:

    # --------------------------------------------------------
    # Your thermal MPEG video
    #
    # Convert it automatically before OpenCV processing.
    # --------------------------------------------------------

    temporary_video = convert_to_mp4(
        video_path
    )

    processing_video_path = temporary_video


else:

    # --------------------------------------------------------
    # For MP4 / other formats, process directly.
    # --------------------------------------------------------

    processing_video_path = video_path


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
# Landmark definitions
# ============================================================

UPPER_LIPS_LANDMARK = [13, 206, 426]

NOSE_HORIZONTAL_LANDMARKS = [64, 278]

NOSE_VERTICAL_LANDMARKS = [4, 94]

NOSE_LANDMARKS = (
    NOSE_HORIZONTAL_LANDMARKS
    + NOSE_VERTICAL_LANDMARKS
    + UPPER_LIPS_LANDMARK
)


# ============================================================
# EMA Landmark Smoother
# ============================================================

class LandmarkEMASmoother:

    def __init__(
        self,
        alpha=0.4,
        num_landmarks=478
    ):

        self.alpha = alpha

        self.num_landmarks = num_landmarks

        # Previous smoothed landmark coordinates
        #
        # Shape:
        #
        #     (num_landmarks, 3)
        #
        # where 3 = x, y, z

        self.prev = None


    def smooth(self, face_landmarks):

        """
        Apply EMA to every MediaPipe landmark.

        Formula:

            smoothed =
                alpha * raw
                +
                (1 - alpha) * previous_smoothed
        """

        # ----------------------------------------------------
        # Convert MediaPipe landmarks to NumPy array
        # ----------------------------------------------------

        raw = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in face_landmarks.landmark
            ],
            dtype=np.float64
        )


        # ----------------------------------------------------
        # First frame / no history
        # ----------------------------------------------------

        if (
            self.prev is None
            or self.prev.shape != raw.shape
        ):

            self.prev = raw.copy()


        # ----------------------------------------------------
        # Subsequent frames
        # ----------------------------------------------------

        else:

            self.prev = (
                self.alpha * raw
                +
                (1.0 - self.alpha) * self.prev
            )


        # ----------------------------------------------------
        # Put smoothed coordinates back into MediaPipe object
        # ----------------------------------------------------

        for i, lm in enumerate(
            face_landmarks.landmark
        ):

            lm.x = float(
                self.prev[i, 0]
            )

            lm.y = float(
                self.prev[i, 1]
            )

            lm.z = float(
                self.prev[i, 2]
            )


        return face_landmarks


    def reset(self):

        """
        Reset EMA history.

        Called when no face is detected.

        This prevents the next detected face from being
        blended with the previous face.
        """

        self.prev = None


# ============================================================
# Create EMA smoother
# ============================================================

landmark_smoother = LandmarkEMASmoother(
    alpha=0.4
)


# ============================================================
# Open converted video with OpenCV
# ============================================================

cap = cv2.VideoCapture(
    processing_video_path
)


# ============================================================
# Check video opening
# ============================================================

if not cap.isOpened():

    face_mesh.close()

    if temporary_video is not None:
        os.remove(temporary_video)

    raise RuntimeError(
        "OpenCV could not open the video:\n"
        f"{processing_video_path}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = cap.get(
    cv2.CAP_PROP_FRAME_COUNT
)


# ------------------------------------------------------------
# Check FPS
# ------------------------------------------------------------

if fps <= 0:

    cap.release()
    face_mesh.close()

    if temporary_video is not None:
        os.remove(temporary_video)

    raise RuntimeError(
        "Could not determine video FPS."
    )


# ============================================================
# Calculate video duration
# ============================================================

total_duration_seconds = (
    total_frames / fps
)


total_minutes = int(
    total_duration_seconds // 60
)

total_seconds = (
    total_duration_seconds % 60
)


total_duration = (
    f"{total_minutes:02d}:"
    f"{total_seconds:05.2f}"
)


# ============================================================
# Print video information
# ============================================================

print()
print("=" * 70)
print("VIDEO INFORMATION USED BY OPENCV")
print("=" * 70)

print(
    f"Processing video: "
    f"{processing_video_path}"
)

print(
    f"FPS: {fps}"
)

print(
    f"Total Frames: {total_frames}"
)

print(
    f"Calculated Duration: "
    f"{total_duration}"
)

print("=" * 70)


# ============================================================
# Main Video Loop
# ============================================================

frame_number = 0

video_time = "00:00.00"


while True:

    # ========================================================
    # Read next frame
    # ========================================================

    ret, frame = cap.read()


    # ========================================================
    # End of video
    # ========================================================

    if not ret:

        print()
        print("=" * 70)
        print("VIDEO READ ENDED")
        print(
            f"Last readable frame: "
            f"{frame_number}"
        )
        print(
            f"Last video time: "
            f"{video_time}"
        )
        print("=" * 70)

        break


    # ========================================================
    # Increment frame number
    # ========================================================
    #
    # This is our own sequential frame counter.
    #
    # Since we read frames sequentially:
    #
    #     1, 2, 3, 4, ...
    #
    # we can calculate the video position directly.
    #
    # ========================================================

    frame_number += 1


    # ========================================================
    # Calculate INPUT VIDEO TIME
    # ========================================================
    #
    # This is NOT real processing time.
    #
    # Example:
    #
    # FPS = 25
    #
    # frame 25:
    #
    #     25 / 25 = 1 second
    #
    # frame 2450:
    #
    #     2450 / 25 = 98 seconds
    #                  = 01:38
    #
    # ========================================================

    video_time_seconds = (
        frame_number / fps
    )


    # ========================================================
    # Convert seconds → MM:SS.xx
    # ========================================================

    minutes = int(
        video_time_seconds // 60
    )

    seconds = (
        video_time_seconds % 60
    )


    video_time = (
        f"{minutes:02d}:"
        f"{seconds:05.2f}"
    )


    # ========================================================
    # Print video time
    # ========================================================

    print(
        f"Frame: {frame_number} | "
        f"Video Time: {video_time}"
    )


    # ========================================================
    # Convert BGR → Grayscale
    # ========================================================

    grey = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # Apply your transformation pipeline
    # ========================================================

    transformed_grey = (
        ut.get_transformed_image(grey)
    )


    # ========================================================
    # IMPORTANT:
    #
    # Always initialize got_frame.
    #
    # This prevents:
    #
    #     NameError: got_frame is not defined
    #
    # when no face is detected.
    # ========================================================

    got_frame = (
        transformed_grey.copy()
    )


    # ========================================================
    # OPTIONAL 90° ROTATION
    # ========================================================

    # transformed_grey = cv2.rotate(
    #     transformed_grey,
    #     cv2.ROTATE_90_CLOCKWISE
    # )


    # ========================================================
    # Convert grayscale → 3-channel image
    #
    # MediaPipe requires a 3-channel image.
    # ========================================================

    rgb = cv2.cvtColor(
        transformed_grey,
        cv2.COLOR_GRAY2BGR
    )


    # ========================================================
    # MediaPipe Face Mesh
    # ========================================================

    results = face_mesh.process(
        rgb
    )


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if results.multi_face_landmarks:


        for face_landmarks in (
            results.multi_face_landmarks
        ):


            # ==================================================
            # EMA SMOOTHING
            # ==================================================

            face_landmarks = (
                landmark_smoother.smooth(
                    face_landmarks
                )
            )


            # ==================================================
            # BREATHING ROI
            # ==================================================

            (
                top_left_cords,
                bottom_right_cords,
                got_frame
            ) = ut.get_breathing_roi_cords(
                transformed_grey,
                face_landmarks
            )


            # ==================================================
            # FOREHEAD ROI
            # ==================================================

            (
                polygon_points,
                mean_pixel,
                got_frame
            ) = ut.get_forhead_poly_coords(
                transformed_grey,
                face_landmarks
            )


            # ==================================================
            # CHEEKS ROI
            # ==================================================

            (
                l,
                r,
                got_frame
            ) = ut.get_cheeks_coordinates(
                transformed_grey,
                face_landmarks,
                [],
                []
            )


            # ==================================================
            # EYES ROI
            # ==================================================

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


            # ==================================================
            # NOSE TIP ROI
            # ==================================================

            (
                top_left_coords,
                bottom_right_coords,
                got_frame
            ) = ut.get_nose_tip_coordinates(
                transformed_grey,
                face_landmarks
            )


            # ==================================================
            # PRINT ROI INFORMATION
            # ==================================================

            print(
                top_left_cords,
                bottom_right_cords,
                type(got_frame)
            )


            print(
                polygon_points,
                mean_pixel,
                type(got_frame)
            )


            print(
                l,
                r,
                type(got_frame)
            )


            print(
                top_left_coords,
                bottom_right_coords,
                top_right_coords,
                bottom_left_coords,
                type(got_frame)
            )


            print(
                top_left_coords,
                bottom_right_coords,
                type(got_frame)
            )


            print(
                "#" * 150
            )


    # ========================================================
    # NO FACE DETECTED
    # ========================================================

    else:

        # ----------------------------------------------------
        # Reset EMA history
        # ----------------------------------------------------

        landmark_smoother.reset()


        print(
            f"No face detected | "
            f"Video Time: {video_time}"
        )


        # ----------------------------------------------------
        # got_frame already contains:
        #
        # transformed_grey.copy()
        #
        # Therefore this frame is simply displayed without
        # ROI processing.
        # ----------------------------------------------------


    # ========================================================
    # ADD INPUT VIDEO TIME TO IMAGE
    # ========================================================

    cv2.putText(
        got_frame,
        f"Video Time: {video_time}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # DISPLAY TRANSFORMED GREY
    # ========================================================

    cv2.imshow(
        "Transformed Grey",
        got_frame
    )


    # ========================================================
    # DISPLAY ORIGINAL FRAME
    # ========================================================

    cv2.imshow(
        "RGB framea",
        frame
    )


    # ========================================================
    # Press Q to quit
    # ========================================================

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):

        print()
        print("Processing stopped by user.")

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

face_mesh.close()

cv2.destroyAllWindows()


# ============================================================
# DELETE TEMPORARY MP4
# ============================================================

if temporary_video is not None:

    try:

        os.remove(
            temporary_video
        )

        print()
        print(
            "Temporary converted MP4 deleted."
        )

    except OSError as e:

        print()
        print(
            "WARNING: Could not delete "
            "temporary MP4:"
        )

        print(
            temporary_video
        )

        print(e)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)