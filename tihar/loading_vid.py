# Clipping of RGB/Thermal Video based on the timestamps provided in the Markers CSV file.

import os
import subprocess
import pandas as pd   

# ==========================================================
# USER SETTINGS
# ==========================================================

#Input Video Path (RGB/Thermal)
VIDEO_PATH = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"

# Markes File Path (CSV)
TIMESTAMP_CSV = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_Passive_Markers.csv"

# Output Folder for the clipped videos
OUTPUT_FOLDER = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\Thermal_Clips"

FFMPEG = "ffmpeg"        # or full path to ffmpeg.exe

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# READ CSV
# ==========================================================
df = pd.read_csv(TIMESTAMP_CSV)

# Convert timestamps to datetime
df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
df["next_question_time"] = pd.to_datetime(df["next_question_time"], utc=True)

# ==========================================================
# VIDEO START = FIRST QUESTION START
# ==========================================================
video_start = df.loc[0, "start_time"]

# ==========================================================
# BUILD CLIP LIST
# ==========================================================
clip_starts = [video_start]
clip_ends = []

for t in df["next_question_time"]:
    clip_ends.append(t)
    clip_starts.append(t)

# Remove the last start (no matching end)
clip_starts = clip_starts[:-1]

print("=" * 60)
print("Total Clips:", len(clip_starts))
print("=" * 60)

# ==========================================================
# CROP EACH CLIP
# ==========================================================
for i, (start, end) in enumerate(zip(clip_starts, clip_ends), start=1):

    start_sec = (start - video_start).total_seconds()
    end_sec = (end - video_start).total_seconds()
    duration = end_sec - start_sec

    if duration <= 0:
        print(f"Skipping Clip {i:02d} (Invalid duration)")
        continue

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"Q{i:02d}.mp4"
    )

    print(
        f"Clip {i:02d} | "
        f"Start={start_sec:.3f}s "
        f"End={end_sec:.3f}s "
        f"Duration={duration:.3f}s"
    )

    cmd = [
        FFMPEG,
        "-y",

        # Accurate seeking
        "-ss", f"{start_sec:.3f}",

        "-i", VIDEO_PATH,

        "-t", f"{duration:.3f}",

        # Video Encoding
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",

        # Audio
        "-c:a", "aac",
        "-b:a", "192k",

        output_file
    ]

    subprocess.run(cmd, check=True)

print("\nAll clips created successfully.")