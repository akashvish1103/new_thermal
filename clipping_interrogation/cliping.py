import subprocess
import os

video_file = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\all_grey_manual_mp4\shivam_grey_manual.mp4"
output_folder = "cropped_clips"
os.makedirs(output_folder, exist_ok=True)

timestamps = [
    ("10", "14"),
    ("19", "23"),
    ("26", "35"),
    ("41", "47"),
    ("55", "1:36"),
    ("1:49", "2:33"),
    ("2:43", "3:09"),
    ("3:19", "3:29"),
    ("3:42", "4:00"),
    ("4:17", "4:22"),
    ("4:34", "4:48"),
    ("4:53", "4:59"),
    ("5:05", "5:24"),
    ("5:36", "5:44"),
    ("6:00", "6:25"),
    ("6:55", "7:02"),
    ("7:13", "7:37")
]

def time_to_seconds(t):
    if ":" in t:
        mins, secs = map(int, t.split(":"))
        return mins * 60 + secs
    return int(t)

for i, (start, end) in enumerate(timestamps, start=1):

    start_sec = time_to_seconds(start)
    end_sec = time_to_seconds(end)

    duration = end_sec - start_sec

    if duration <= 0:
        print(f"Skipping clip {i}: Invalid interval {start} -> {end}")
        continue

    output_file = os.path.join(output_folder, f"shivam_clip_{i}.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", video_file,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        output_file
    ]

    print(f"Creating {output_file}")
    subprocess.run(cmd)

print("All clips created.")

