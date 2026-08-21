# This Code works fine, Time Duration issue solved...
# Solved jitter ROI issue
# Use this ONLY for Saving the ROI LOG file
# Final Code by Akash V.


import os
import csv
import subprocess
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

import mediapipe as mp
import numpy as np
import cv2

import utilities as ut
import utility_linear_mapping as ulm


def process_video(input_video):

        # ============================================================
        # VIDEO PATH
        # ============================================================
    # The batch runner supplies video_path automatically.
    # ============================================================

    video_path = input_video

    # ============================================================
    # CSV LOG FILE
    #
    # The log is saved next to the ORIGINAL input video.
    #
    # Example:
    #
    #   48_HDRS_Thermal_30_40.mpg
    #              ↓
    #   48_HDRS_Thermal_30_40_log_stats.csv
    #
    # The temporary FFmpeg MP4 is NEVER used to name the log.
    # ============================================================

    video_filename_without_extension = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    LOG_PATH = os.path.join(
        os.path.dirname(video_path),
        video_filename_without_extension + "_log_stats.csv"
    )


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
    # ROI LOGGING SETTINGS
    # ============================================================

    ROI_NAMES = [
        "breathing",
        "forehead",
        "cheek_L",
        "cheek_R",
        "eye_L",
        "eye_R",
        "nose"
    ]

    STAT_NAMES = [
        "count",
        "mean_px",
        "std_px",
        "min_px",
        "max_px",
        "median_px",
        "mean_temp",
        "std_temp",
        "min_temp",
        "max_temp",
        "median_temp",
        "delta_temp_from_baseline"
    ]


    def extract_roi_pixels(image, shape_type, geometry):
        """
        Extract RAW grayscale pixels belonging to an ROI.

        shape_type:
            "box"      -> geometry = (top_left, bottom_right)
            "polygon"  -> geometry = list of (x, y)
        """

        height, width = image.shape[:2]

        if shape_type == "box":

            top_left, bottom_right = geometry

            x1 = int(top_left[0])
            y1 = int(top_left[1])
            x2 = int(bottom_right[0])
            y2 = int(bottom_right[1])

            # Keep ROI inside image boundaries.
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if x2 <= x1 or y2 <= y1:
                return np.array([], dtype=image.dtype)

            return image[y1:y2, x1:x2].flatten()

        elif shape_type == "polygon":

            points = np.array(
                [[int(p[0]), int(p[1])] for p in geometry],
                dtype=np.int32
            )

            if len(points) < 3:
                return np.array([], dtype=image.dtype)

            mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            cv2.fillPoly(
                mask,
                [points],
                255
            )

            return image[mask == 255].flatten()

        else:
            raise ValueError(
                f"Unknown ROI shape type: {shape_type}"
            )


    def calculate_roi_stats(
        pixels,
        min_temp,
        max_temp,
        first_pixel,
        last_pixel
    ):
        """
        Calculate pixel statistics and corresponding
        temperature statistics for one ROI.
        """

        if pixels.size == 0:
            return {
                "count": 0,
                "mean_px": np.nan,
                "std_px": np.nan,
                "min_px": np.nan,
                "max_px": np.nan,
                "median_px": np.nan,
                "mean_temp": np.nan,
                "std_temp": np.nan,
                "min_temp": np.nan,
                "max_temp": np.nan,
                "median_temp": np.nan
            }

        px = pixels.astype(np.float64)

        temp = ulm.map_pixel_to_temperature(
            px,
            min_temp,
            max_temp,
            first_pixel,
            last_pixel
        )

        temp = np.asarray(temp, dtype=np.float64)

        return {
            "count": int(px.size),

            "mean_px": float(np.mean(px)),
            "std_px": float(np.std(px)),
            "min_px": float(np.min(px)),
            "max_px": float(np.max(px)),
            "median_px": float(np.median(px)),

            "mean_temp": float(np.mean(temp)),
            "std_temp": float(np.std(temp)),
            "min_temp": float(np.min(temp)),
            "max_temp": float(np.max(temp)),
            "median_temp": float(np.median(temp))
        }


    # ============================================================
    # CALIBRATION FOR TEMPERATURE LOGGING
    #
    # This uses the ORIGINAL thermal video path, not the temporary
    # FFmpeg MP4. Therefore the original thermal calibration
    # information remains unchanged.
    # ============================================================

    print()
    print("=" * 70)
    print("PREPARING TEMPERATURE CALIBRATION FOR CSV LOG")
    print("=" * 70)

    pixel_column = ulm.get_one_pixel_column(video_path)

    (
        min_temp,
        max_temp,
        first_pixel,
        last_pixel
    ) = ulm.get_first_last_element_from_OnePixelColumn_AND_min_max_from_filename(
        pixel_column,
        video_path
    )

    print(f"Temperature range: {min_temp} - {max_temp} °C")
    print(f"First calibration pixel: {first_pixel}")
    print(f"Last calibration pixel: {last_pixel}")
    print("=" * 70)


    # ============================================================
    # BASELINE TEMPERATURE
    #
    # First valid mean temperature for each ROI becomes its
    # baseline. Subsequent rows contain:
    #
    # current_mean_temp - baseline_mean_temp
    # ============================================================

    roi_baseline_temp = {
        roi: None
        for roi in ROI_NAMES
    }


    # ============================================================
    # CSV COLUMNS
    # ============================================================

    CSV_FIELDNAMES = [
        "frame_number",
        "time_sec",
        "video_time",
        "face_detected"
    ]

    for roi in ROI_NAMES:
        for stat in STAT_NAMES:
            CSV_FIELDNAMES.append(
                f"{roi}_{stat}"
            )


    # ============================================================
    # OPEN CSV LOG
    # ============================================================

    print()
    print("=" * 70)
    print("CSV LOG")
    print("=" * 70)
    print(f"Saving statistics to:")
    print(LOG_PATH)
    print("=" * 70)

    log_file = open(
        LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.DictWriter(
        log_file,
        fieldnames=CSV_FIELDNAMES
    )

    csv_writer.writeheader()


    # ============================================================
    # Main Video Loop
    # ============================================================

    frame_number = 0

    video_time = "00:00.00"


    try:

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
                print(f"Last readable frame: {frame_number}")
                print(f"Last video time: {video_time}")
                print("=" * 70)

                break

            # ========================================================
            # Increment frame number
            # ========================================================

            frame_number += 1

            # ========================================================
            # INPUT VIDEO TIME
            #
            # IMPORTANT:
            #
            # Keep this exactly based on the FPS of the converted
            # processing video. This preserves the video-duration
            # behavior that solved the previous MPEG cut problem.
            # ========================================================

            video_time_seconds = (
                frame_number / fps
            )

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
            # Start a blank CSV row.
            #
            # Even when NO FACE is detected, this row will still be
            # written. ROI values remain blank.
            # ========================================================

            row = {
                "frame_number": frame_number,
                "time_sec": round(video_time_seconds, 3),
                "video_time": video_time,
                "face_detected": False
            }

            for roi in ROI_NAMES:
                for stat in STAT_NAMES:
                    row[f"{roi}_{stat}"] = ""

            # ========================================================
            # Convert BGR → Grayscale
            # ========================================================

            grey = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # ========================================================
            # Apply your existing transformation pipeline
            # ========================================================

            transformed_grey = (
                ut.get_transformed_image(grey)
            )

            # ========================================================
            # IMPORTANT:
            #
            # got_frame is initialized before face detection.
            # Therefore no-face frames cannot cause a NameError.
            # ========================================================

            got_frame = transformed_grey.copy()

            # ========================================================
            # OPTIONAL 90° ROTATION
            # ========================================================

            # transformed_grey = cv2.rotate(
            #     transformed_grey,
            #     cv2.ROTATE_90_CLOCKWISE
            # )

            # ========================================================
            # MediaPipe expects 3 channels
            # ========================================================

            rgb = cv2.cvtColor(
                transformed_grey,
                cv2.COLOR_GRAY2BGR
            )

            # ========================================================
            # MediaPipe Face Mesh
            # ========================================================

            results = face_mesh.process(rgb)

            # ========================================================
            # FACE DETECTED
            # ========================================================

            if results.multi_face_landmarks:

                row["face_detected"] = True

                for face_landmarks in results.multi_face_landmarks:

                    # ==================================================
                    # EMA LANDMARK SMOOTHING
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
                        nose_top_left,
                        nose_bottom_right,
                        got_frame
                    ) = ut.get_nose_tip_coordinates(
                        transformed_grey,
                        face_landmarks
                    )

                    # ==================================================
                    # Define ROI geometries
                    #
                    # IMPORTANT:
                    #
                    # Coordinates are generated exactly as before
                    # using utilities.py.
                    #
                    # Pixel extraction is performed on RAW 'grey',
                    # NOT transformed_grey.
                    # ==================================================

                    roi_geometries = {

                        "breathing": (
                            "box",
                            (
                                top_left_cords,
                                bottom_right_cords
                            )
                        ),

                        "forehead": (
                            "polygon",
                            polygon_points
                        ),

                        "cheek_L": (
                            "polygon",
                            l
                        ),

                        "cheek_R": (
                            "polygon",
                            r
                        ),

                        "eye_L": (
                            "box",
                            (
                                top_left_coords,
                                bottom_right_coords
                            )
                        ),

                        "eye_R": (
                            "box",
                            (
                                top_right_coords,
                                bottom_left_coords
                            )
                        ),

                        "nose": (
                            "box",
                            (
                                nose_top_left,
                                nose_bottom_right
                            )
                        )
                    }

                    # ==================================================
                    # Calculate statistics for every ROI
                    # ==================================================

                    for (
                        roi_name,
                        (
                            shape_type,
                            geometry
                        )
                    ) in roi_geometries.items():

                        # ------------------------------------------------
                        # RAW grayscale thermal pixels
                        #
                        # Do NOT use transformed_grey here.
                        # ------------------------------------------------

                        pixels = extract_roi_pixels(
                            grey,
                            shape_type,
                            geometry
                        )

                        stats = calculate_roi_stats(
                            pixels,
                            min_temp,
                            max_temp,
                            first_pixel,
                            last_pixel
                        )

                        # ------------------------------------------------
                        # Establish baseline
                        # ------------------------------------------------

                        if (
                            roi_baseline_temp[roi_name]
                            is None
                            and
                            not np.isnan(
                                stats["mean_temp"]
                            )
                        ):

                            roi_baseline_temp[roi_name] = (
                                stats["mean_temp"]
                            )

                        baseline = (
                            roi_baseline_temp[roi_name]
                        )

                        # ------------------------------------------------
                        # Temperature change from baseline
                        # ------------------------------------------------

                        if (
                            baseline is not None
                            and
                            not np.isnan(
                                stats["mean_temp"]
                            )
                        ):

                            delta_temp = (
                                stats["mean_temp"]
                                -
                                baseline
                            )

                        else:

                            delta_temp = np.nan

                        # ------------------------------------------------
                        # Store statistics in CSV row
                        # ------------------------------------------------

                        for stat in [
                            "count",
                            "mean_px",
                            "std_px",
                            "min_px",
                            "max_px",
                            "median_px",
                            "mean_temp",
                            "std_temp",
                            "min_temp",
                            "max_temp",
                            "median_temp"
                        ]:

                            value = stats[stat]

                            if (
                                isinstance(value, float)
                                and np.isnan(value)
                            ):

                                row[
                                    f"{roi_name}_{stat}"
                                ] = ""

                            elif isinstance(value, float):

                                row[
                                    f"{roi_name}_{stat}"
                                ] = round(
                                    value,
                                    4
                                )

                            else:

                                row[
                                    f"{roi_name}_{stat}"
                                ] = value

                        if np.isnan(delta_temp):

                            row[
                                f"{roi_name}_delta_temp_from_baseline"
                            ] = ""

                        else:

                            row[
                                f"{roi_name}_delta_temp_from_baseline"
                            ] = round(
                                delta_temp,
                                4
                            )

                    # ==================================================
                    # Console summary
                    # ==================================================

                    print(
                        f"Frame: {frame_number} | "
                        f"Video Time: {video_time} | "
                        f"Nose Mean Temp: "
                        f"{row['nose_mean_temp']} | "
                        f"Forehead Mean Temp: "
                        f"{row['forehead_mean_temp']}"
                    )

            # ========================================================
            # NO FACE DETECTED
            # ========================================================

            else:

                # ----------------------------------------------------
                # Reset EMA history.
                # ----------------------------------------------------

                landmark_smoother.reset()

                print(
                    f"Frame: {frame_number} | "
                    f"Video Time: {video_time} | "
                    f"No face detected"
                )

            # ========================================================
            # WRITE THIS FRAME TO CSV
            #
            # This happens whether a face was detected or not.
            # ========================================================

            csv_writer.writerow(row)

            # --------------------------------------------------------
            # Flush periodically.
            # --------------------------------------------------------

            if frame_number % 100 == 0:
                log_file.flush()

            # ========================================================
            # Add VIDEO TIME to displayed transformed image
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
            # ========================================================
            # OPTIONAL DISPLAY
            # ========================================================

            if SHOW_WINDOWS:

                cv2.imshow(
                    "Transformed Grey",
                    got_frame
                )

                cv2.imshow(
                    "RGB framea",
                    frame
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):

                    print()
                    print("Processing stopped by user.")

                    break

    finally:

        # ========================================================
        # Always flush and close CSV
        # ========================================================

        log_file.flush()
        log_file.close()

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
    print()
    print("ROI statistics log saved to:")
    print(LOG_PATH)
    print("=" * 70)


# ============================================================
# GRAPH / QUESTION-MARKER FUNCTIONS
# ============================================================

SMA_WINDOW = 50

ROIS = {
    "Forehead": "forehead_mean_temp",
    "Left Cheek": "cheek_L_mean_temp",
    "Right Cheek": "cheek_R_mean_temp",
    "Nose": "nose_mean_temp",
}


def find_marker_file(log_path):
    """
    Automatically select the matching marker file from the log name.

    Example:
        48_HDRS_Thermal_30_40_log_stats.csv
            -> 48_HDRS_Markers.csv

        48_Passive_Thermal_30_40_log_stats.csv
            -> 48_Passive_Markers.csv
    """

    log_filename = os.path.basename(log_path)
    folder = os.path.dirname(log_path)

    if "_Passive_" in log_filename:
        marker_filename = (
            log_filename.split("_Passive_")[0]
            + "_Passive_Markers.csv"
        )

    elif "_HDRS_" in log_filename:
        marker_filename = (
            log_filename.split("_HDRS_")[0]
            + "_HDRS_Markers.csv"
        )

    else:
        raise ValueError(
            "Cannot determine Passive/HDRS from log filename:\n"
            + log_filename
        )

    marker_path = os.path.join(folder, marker_filename)

    if not os.path.isfile(marker_path):
        raise FileNotFoundError(
            "Matching marker file was not found:\n"
            + marker_path
        )

    return marker_path


def get_question_markers(markers_df):
    """
    Supports the two marker formats used in this dataset.

    Passive:
        start_time
        next_question_time

    HDRS:
        marker_type
        question_id
        question_order
        timestamp_iso
        timestamp_ms

    Q1 is forced to video time 0 because the user confirmed
    that the first question starts when video recording starts.
    """

    # --------------------------------------------------------
    # HDRS
    # --------------------------------------------------------

    if "question_order" in markers_df.columns:

        start_rows = markers_df[
            markers_df["marker_type"]
            .astype(str)
            .str.lower()
            == "start"
        ]

        if start_rows.empty:
            raise ValueError(
                "HDRS marker file does not contain a 'start' marker."
            )

        video_start_time = pd.to_datetime(
            start_rows.iloc[0]["timestamp_iso"],
            errors="coerce",
            utc=True
        )

        if pd.isna(video_start_time):
            raise ValueError(
                "Could not parse the HDRS video start timestamp."
            )

        lock_rows = markers_df[
            markers_df["marker_type"]
            .astype(str)
            .str.lower()
            == "lock"
        ].copy()

        lock_rows["question_order"] = pd.to_numeric(
            lock_rows["question_order"],
            errors="coerce"
        )

        lock_rows = lock_rows.dropna(
            subset=["question_order"]
        )

        lock_rows = lock_rows.sort_values(
            "question_order"
        )

        question_markers = []

        for _, row in lock_rows.iterrows():

            q_number = int(row["question_order"])

            question_time = pd.to_datetime(
                row["timestamp_iso"],
                errors="coerce",
                utc=True
            )

            if pd.isna(question_time):
                continue

            relative_seconds = (
                question_time - video_start_time
            ).total_seconds()

            question_markers.append({
                "question": f"Q{q_number}",
                "video_time_sec": relative_seconds,
                "absolute_time": question_time,
            })

        # User-confirmed rule: Q1 starts at video time 0.
        for marker in question_markers:
            if marker["question"] == "Q1":
                marker["video_time_sec"] = 0.0
                break

        return question_markers

    # --------------------------------------------------------
    # Passive
    # --------------------------------------------------------

    if (
        "start_time" in markers_df.columns
        and "next_question_time" in markers_df.columns
    ):

        start_times = pd.to_datetime(
            markers_df["start_time"],
            errors="coerce",
            utc=True
        )

        next_question_times = pd.to_datetime(
            markers_df["next_question_time"],
            errors="coerce",
            utc=True
        )

        valid_start_times = start_times.dropna()

        if valid_start_times.empty:
            raise ValueError(
                "Passive marker file contains no valid start_time."
            )

        video_start_time = valid_start_times.iloc[0]

        question_markers = [{
            "question": "Q1",
            "video_time_sec": 0.0,
            "absolute_time": video_start_time,
        }]

        valid_next_times = next_question_times.dropna()

        # Final next_question_time is the end of the final question,
        # so it is not used as a new question marker.
        for i, next_time in enumerate(
            valid_next_times.iloc[:-1],
            start=2
        ):

            relative_seconds = (
                next_time - video_start_time
            ).total_seconds()

            question_markers.append({
                "question": f"Q{i}",
                "video_time_sec": relative_seconds,
                "absolute_time": next_time,
            })

        return question_markers

    raise ValueError(
        "Unknown marker-file format. Columns found:\n"
        + str(list(markers_df.columns))
    )


def save_temperature_graph(log_path):
    """
    Load one ROI log, automatically find its matching marker file,
    add question markers, and save the graph beside the log.
    """

    marker_path = find_marker_file(log_path)

    df = pd.read_csv(log_path)

    time = pd.to_numeric(
        df["time_sec"],
        errors="coerce"
    )

    markers_df = pd.read_csv(marker_path)

    question_markers = get_question_markers(
        markers_df
    )

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    for roi_name, column in ROIS.items():

        if column not in df.columns:
            print(
                f"WARNING: {column} not found in "
                f"{os.path.basename(log_path)}"
            )
            continue

        temperature = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        temperature_sma = temperature.rolling(
            window=SMA_WINDOW,
            min_periods=1
        ).mean()

        ax.plot(
            time,
            temperature_sma,
            linewidth=2,
            label=roi_name
        )

    # --------------------------------------------------------
    # Question markers
    #
    # Only the number is displayed:
    #
    #     1   2   3   4 ...
    #
    # It is placed just ABOVE the x-axis.
    # --------------------------------------------------------

    for marker in question_markers:

        marker_time = marker["video_time_sec"]

        question_number = (
            marker["question"].replace("Q", "")
        )

        ax.axvline(
            x=marker_time,
            linestyle="--",
            linewidth=1,
            alpha=0.55
        )

        ax.text(
            marker_time,
            0.02,
            question_number,
            transform=ax.get_xaxis_transform(),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9
        )

    ax.set_xlabel(
        "Video Time (seconds)"
    )

    ax.set_ylabel(
        "Mean ROI Temperature (°C)"
    )

    video_name = os.path.splitext(
        os.path.basename(log_path)
    )[0].replace(
        "_log_stats",
        ""
    )

    ax.set_title(
        f"{video_name} - Mean Temperature of Facial ROIs "
        f"({SMA_WINDOW}-Frame SMA)"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    graph_path = os.path.join(
        os.path.dirname(log_path),
        video_name + "_temperature_graph.png"
    )

    fig.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("GRAPH SAVED:")
    print(graph_path)
    print()
    print("MARKER FILE USED:")
    print(marker_path)

    return graph_path


# ============================================================
# BATCH SETTINGS
# ============================================================

ROOT_DATA_DIR = r"D:\000_ofc_thermalData\sorted_data"  

# True  -> show OpenCV windows while each video is processed.
# False -> process in the background without OpenCV windows.
SHOW_WINDOWS = False


# ============================================================
# MAIN BATCH PROCESSING
# ============================================================

def main():

    root = os.path.abspath(
        ROOT_DATA_DIR
    )

    if not os.path.isdir(root):
        raise FileNotFoundError(
            "Root data directory does not exist:\n"
            + root
        )

    subject_dirs = sorted(
        [
            p for p in os.scandir(root)
            if p.is_dir() and p.name.isdigit()
        ],
        key=lambda p: int(p.name)
    )

    if not subject_dirs:
        raise RuntimeError(
            "No subject folders were found in:\n"
            + root
        )

    print()
    print("=" * 80)
    print("BATCH THERMAL PROCESSING")
    print("=" * 80)
    print(f"Root directory: {root}")
    print(f"Subjects found: {len(subject_dirs)}")
    print("=" * 80)

    successful = []
    failed = []

    for subject_entry in subject_dirs:

        subject_dir = subject_entry.path
        subject = subject_entry.name

        print()
        print()
        print("#" * 80)
        print(f"SUBJECT {subject}")
        print("#" * 80)

        for condition in ["HDRS", "Passive"]:

            pattern = os.path.join(
                subject_dir,
                f"{subject}_{condition}_Thermal_30_40.mpg"
            )

            if not os.path.isfile(pattern):

                print()
                print(
                    f"{condition}: video not found -> SKIPPING"
                )
                print(pattern)
                continue

            video_path = pattern

            print()
            print("=" * 80)
            print(
                f"PROCESSING SUBJECT {subject} - {condition}"
            )
            print("=" * 80)
            print(f"Input video: {video_path}")

            try:

                # ------------------------------------------------
                # PROCESS VIDEO
                # ------------------------------------------------
                #
                # The existing FFmpeg conversion logic remains
                # inside process_video().
                #
                # The original MPG is never modified.
                # The temporary MP4 is deleted afterward.
                # ------------------------------------------------

                # Save current display preference globally so the
                # existing processing code can use it.
                globals()["SHOW_WINDOWS"] = SHOW_WINDOWS

                process_video(
                    video_path
                )

                log_path = os.path.join(
                    subject_dir,
                    os.path.splitext(
                        os.path.basename(video_path)
                    )[0]
                    + "_log_stats.csv"
                )

                # ------------------------------------------------
                # CREATE + SAVE GRAPH
                # ------------------------------------------------

                save_temperature_graph(
                    log_path
                )

                successful.append(
                    (subject, condition)
                )

            except Exception as e:

                failed.append(
                    (subject, condition, str(e))
                )

                print()
                print("!!! ERROR !!!")
                print(
                    f"Subject: {subject}"
                )
                print(
                    f"Condition: {condition}"
                )
                print(e)

                # Continue with the next video instead of
                # stopping the entire batch.
                continue

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()
    print()
    print("=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)

    print()
    print("SUCCESSFUL:")
    for subject, condition in successful:
        print(
            f"  Subject {subject} - {condition}"
        )

    print()
    print("FAILED:")
    if not failed:
        print("  None")
    else:
        for subject, condition, error in failed:
            print(
                f"  Subject {subject} - {condition}"
            )
            print(
                f"      {error}"
            )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
