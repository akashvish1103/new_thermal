import pandas as pd
import subprocess
import os

# ==========================================================
# USER INPUTS
# ==========================================================

MARKER_FILE =  r"D:\000_data_collection_hti\girish-demo-clippong-markers.csv"

VIDEO_FILE = r"D:\000_data_collection_hti\girish_thermal_clipping.wmv"   

OUTPUT_FOLDER =  r"D:\000_data_collection_hti"

# Time (in seconds) where Question 1 starts in the VIDEO
Q1_START_IN_VIDEO = 23.0

# ==========================================================


def compute_durations(marker_file):

    df = pd.read_csv(marker_file)

    start_time = pd.to_datetime(
        df["start_time"].dropna().iloc[0]
    )

    next_times = pd.to_datetime(
        df["next_question_time"]
    )

    durations = []

    # Q1 duration
    durations.append(
        (next_times.iloc[0] - start_time).total_seconds()
    )

    # Q2 onwards
    for i in range(1, len(next_times)):

        durations.append(
            (
                next_times.iloc[i]
                - next_times.iloc[i - 1]
            ).total_seconds()
        )

    return durations


def clip_video(
        video_file,
        output_folder,
        durations,
        q1_start):

    os.makedirs(output_folder, exist_ok=True)

    current_start = q1_start

    for i, duration in enumerate(durations, start=1):

        current_end = current_start + duration

        output_file = os.path.join(
            output_folder,
            f"question_0_{i}.mp4"
        )

        print(
            f"Q{i}: "
            f"{current_start:.3f} -> "
            f"{current_end:.3f}"
        )

        cmd = [
            "ffmpeg",
            "-y",

            "-i", video_file,

            "-ss", str(current_start),

            "-t", str(duration),

            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",

            output_file
        ]

        subprocess.run(cmd, check=True)

        current_start = current_end

    print("\nAll clips generated successfully.")


def main():

    durations = compute_durations(MARKER_FILE)

    print("\nQuestion Durations")

    for i, d in enumerate(durations, start=1):
        print(f"Q{i}: {d:.3f} sec")

    print("\nCreating clips...\n")

    clip_video(
        VIDEO_FILE,
        OUTPUT_FOLDER,
        durations,
        Q1_START_IN_VIDEO
    )


if __name__ == "__main__":
    main()