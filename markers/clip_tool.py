# import pandas as pd
# import subprocess
# import os

# # ==========================================
# # CONFIGURATION
# # ==========================================
# VIDEO_FILE_PATH = r"D:\000_data_collection_hti\2026-06-11 10-52-36.mp4"        # Update this to your video file
# MARKER_FILE_PATH = r"D:\000_data_collection_hti\paradigm_timestamps (4).csv"
# OUTPUT_FOLDER = r"D:\000_data_collection_hti"
# # ==========================================

# def process_video():
#     # 1. Create output directory
#     if not os.path.exists(OUTPUT_FOLDER):
#         os.makedirs(OUTPUT_FOLDER)

#     # 2. Load markers
#     df = pd.read_csv(MARKER_FILE_PATH)
#     df['start_time'] = pd.to_datetime(df['start_time'])
#     df['next_question_time'] = pd.to_datetime(df['next_question_time'])

#     # 3. Calculate relative timing
#     # We use the VERY FIRST timestamp as the "Zero" point (0.0s)
#     base_time = df['start_time'].iloc[0]
    
#     # Logic: 
#     # Start of Q1 = 0s
#     # Start of Q2 = (Start of Q2 - Base)
#     starts = [0.0]
#     for i in range(len(df) - 1):
#         time_diff = (df['next_question_time'].iloc[i] - base_time).total_seconds()
#         starts.append(time_diff)
        
#     # Ends are simply the diff of next_question_time relative to base
#     ends = [(t - base_time).total_seconds() for t in df['next_question_time']]

#     # 4. Clipping
#     for i, (s, e) in enumerate(zip(starts, ends), 1):
#         output_path = os.path.join(OUTPUT_FOLDER, f"question_{i}.mp4")
#         print(f"Generating {output_path} ({s:.2f}s to {e:.2f}s)...")
        
#         cmd = [
#             'ffmpeg', '-i', VIDEO_FILE_PATH,
#             '-ss', str(s),
#             '-to', str(e),
#             '-c', 'copy', output_path
#         ]
#         subprocess.run(cmd, check=True)

# if __name__ == "__main__":
#     process_video()
#     print("Done! Check the 'Clips' folder.")

####################################################################

# """
# ╔══════════════════════════════════════════════════════════╗
# ║           INTERROGATION VIDEO CLIPPER                    ║
# ║  Set the 3 paths below and run:  python clip_video.py   ║
# ╚══════════════════════════════════════════════════════════╝
# """

# # ─────────────────────────────────────────────────────────
# #  ✏️  SET YOUR PATHS HERE
# # ─────────────────────────────────────────────────────────
# VIDEO_PATH   = r"D:\000_data_collection_hti\girish_clipping_vid.mp4"         # input video
# MARKERS_FILE = r"D:\000_data_collection_hti\girish-demo-clippong-markers.csv"            # marker Excel/CSV file
# OUTPUT_FOLDER = r"D:\000_data_collection_hti"            # output folder (created if missing)

# # Optional: also save each question as an individual clip (Q01.mp4 … Q10.mp4)
# SAVE_INDIVIDUAL_CLIPS = True
# # ─────────────────────────────────────────────────────────

# import json
# import os
# import subprocess
# import sys
# from datetime import datetime


# def parse_ts(ts_str: str) -> datetime:
#     return datetime.fromisoformat(str(ts_str).strip().replace("Z", "+00:00"))

# def fmt(dt: datetime) -> str:
#     return dt.strftime("%H:%M:%S.%f")[:-3] + "Z"

# def to_sec(ts: datetime, origin: datetime) -> float:
#     return (ts - origin).total_seconds()


# # ── 1. Load marker file (Excel or CSV) ───────────────────────────────────────
# def load_markers(path: str):
#     try:
#         import pandas as pd
#     except ImportError:
#         print("[ERROR] pandas not installed.  Run:  pip install pandas openpyxl")
#         sys.exit(1)

#     ext = os.path.splitext(path)[1].lower()
#     if ext in (".xlsx", ".xlsm", ".xls"):
#         df = pd.read_excel(path)
#     elif ext in (".csv", ".tsv"):
#         df = pd.read_csv(path)
#     else:
#         print(f"[ERROR] Unrecognised marker file extension: {ext}")
#         sys.exit(1)

#     # Normalise column names (strip whitespace, lowercase)
#     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

#     if "start_time" not in df.columns or "next_question_time" not in df.columns:
#         print(f"[ERROR] Expected columns 'start_time' and 'next_question_time'. Found: {list(df.columns)}")
#         sys.exit(1)

#     session_start = parse_ts(df["start_time"].dropna().iloc[0])
#     boundaries    = [parse_ts(t) for t in df["next_question_time"].dropna()]

#     if len(boundaries) == 0:
#         print("[ERROR] No values found in 'next_question_time' column.")
#         sys.exit(1)

#     starts   = [session_start] + boundaries[:-1]
#     segments = list(zip(starts, boundaries))
#     return session_start, boundaries, segments


# # ── 2. Extract video creation time from metadata ──────────────────────────────
# def get_video_start(video_path: str) -> datetime | None:
#     try:
#         cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
#                "-show_format", "-show_streams", video_path]
#         meta = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))
#         for src in [meta.get("format", {})] + meta.get("streams", []):
#             for key in ("creation_time", "com.apple.quicktime.creationdate"):
#                 val = src.get("tags", {}).get(key)
#                 if val:
#                     return parse_ts(val)
#     except Exception:
#         pass
#     return None


# # ── 3. Clip with ffmpeg ───────────────────────────────────────────────────────
# def clip(video_path: str, out_path: str, t_start: float, t_end: float):
#     dur = t_end - t_start
#     # Try fast stream-copy first (no re-encode)
#     cmd = ["ffmpeg", "-y", "-ss", f"{t_start:.6f}", "-i", video_path,
#            "-t", f"{dur:.6f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
#            out_path]
#     r = subprocess.run(cmd, capture_output=True, text=True)
#     if r.returncode != 0:
#         # Fallback: re-encode
#         cmd = ["ffmpeg", "-y", "-ss", f"{t_start:.6f}", "-i", video_path,
#                "-t", f"{dur:.6f}", "-c:v", "libx264", "-c:a", "aac",
#                "-avoid_negative_ts", "make_zero", out_path]
#         subprocess.run(cmd, check=True, capture_output=True)


# # ── MAIN ──────────────────────────────────────────────────────────────────────
# def main():
#     print("\n" + "="*62)
#     print("  INTERROGATION VIDEO CLIPPER")
#     print("="*62)

#     # Validate paths
#     for label, path in [("VIDEO_PATH", VIDEO_PATH), ("MARKERS_FILE", MARKERS_FILE)]:
#         if not os.path.isfile(path):
#             print(f"\n[ERROR] {label} not found:\n  {path}")
#             sys.exit(1)

#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#     # ── Load markers ──────────────────────────────────────────────────────────
#     session_start, boundaries, segments = load_markers(MARKERS_FILE)
#     n = len(segments)

#     print(f"\n  Marker file     : {MARKERS_FILE}")
#     print(f"  Session start   : {fmt(session_start)}")
#     print(f"  Session end     : {fmt(boundaries[-1])}")
#     total = (boundaries[-1] - session_start).total_seconds()
#     print(f"  Q&A duration    : {total:.1f}s  ({total/60:.1f} min)")
#     print(f"  Questions found : {n}")
#     print(f"\n  {'Q':>4}   {'Start':>16}   {'End':>16}   {'Dur':>7}")
#     print(f"  {'-'*52}")
#     for i, (s, e) in enumerate(segments, 1):
#         d = (e - s).total_seconds()
#         print(f"  Q{i:02d}   {fmt(s):>16}   {fmt(e):>16}   {d:6.2f}s")
#     print()

#     # ── Get video recording start time ────────────────────────────────────────
#     print(f"  Video file      : {VIDEO_PATH}")
#     video_start = get_video_start(VIDEO_PATH)

#     if video_start:
#         print(f"  Video start     : {fmt(video_start)}  (from file metadata ✓)")
#     else:
#         print("  [WARN] Could not read creation_time from video metadata.")
#         print(f"         Session Q1 begins at: {fmt(session_start)}")
#         raw = input("  Enter video recording start time (ISO-8601, e.g. 2026-06-11T06:58:10Z): ").strip()
#         video_start = parse_ts(raw)

#     offset = (session_start - video_start).total_seconds()
#     if offset < 0:
#         print(f"\n  [ERROR] video_start is AFTER session_start — check timestamps.")
#         sys.exit(1)
#     print(f"  Pre-Q1 margin   : {offset:.2f}s  ({offset/60:.1f} min)")

#     # ── Get video duration ────────────────────────────────────────────────────
#     cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", VIDEO_PATH]
#     meta = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))
#     vid_dur = float(meta["format"]["duration"])
#     print(f"  Video duration  : {vid_dur:.1f}s  ({vid_dur/60:.1f} min)")
#     print(f"  Output folder   : {OUTPUT_FOLDER}\n")

#     def clamp(t):
#         return max(0.0, min(vid_dur, t))

#     # ── Full session clip ─────────────────────────────────────────────────────
#     t0 = clamp(to_sec(session_start,   video_start))
#     t1 = clamp(to_sec(boundaries[-1],  video_start))

#     full_out = os.path.join(OUTPUT_FOLDER, "full_session.mp4")
#     print(f"  ► Clipping full session  [{t0:.2f}s → {t1:.2f}s]  ({t1-t0:.1f}s) ...")
#     clip(VIDEO_PATH, full_out, t0, t1)
#     print(f"    ✓ full_session.mp4\n")

#     # ── Individual question clips ─────────────────────────────────────────────
#     if SAVE_INDIVIDUAL_CLIPS:
#         print(f"  ► Clipping {n} individual question files ...")
#         for i, (s, e) in enumerate(segments, 1):
#             t_s = clamp(to_sec(s, video_start))
#             t_e = clamp(to_sec(e, video_start))
#             out = os.path.join(OUTPUT_FOLDER, f"Q{i:02d}.mp4")
#             print(f"    Q{i:02d}  [{t_s:.2f}s → {t_e:.2f}s]  ({t_e-t_s:.1f}s)", end="  ... ", flush=True)
#             clip(VIDEO_PATH, out, t_s, t_e)
#             print("✓")

#     print(f"\n{'='*62}")
#     print(f"  ✅  Done!  Clips saved to:\n  {os.path.abspath(OUTPUT_FOLDER)}")
#     print(f"{'='*62}\n")


# if __name__ == "__main__":
#     main()

############################################################


# import pandas as pd
# import subprocess
# import os

# # ==========================================
# # CONFIGURATION
# # ==========================================
# VIDEO_FILE_PATH =  r"D:\000_data_collection_hti\girish_thermal_clipping.wmv"      
# MARKER_FILE_PATH = r"D:\000_data_collection_hti\girish-demo-clippong-markers.csv"
# OUTPUT_FOLDER = r"D:\000_data_collection_hti"
# DURATION_LAG = 21.0  # Your verified start time
# # ==========================================

# def clip_video_by_duration():
#     if not os.path.exists(OUTPUT_FOLDER):
#         os.makedirs(OUTPUT_FOLDER)

#     df = pd.read_csv(MARKER_FILE_PATH)
    
#     # Convert to datetime to perform math, then extract total seconds
#     df['start_time'] = pd.to_datetime(df['start_time'])
#     df['next_question_time'] = pd.to_datetime(df['next_question_time'])
    
#     # Calculate durations for each question row
#     # Q1 = next_question_time[0] - start_time[0]
#     # Q2 = next_question_time[1] - next_question_time[0]
    
#     durations = []
#     # Q1 duration
#     durations.append((df['next_question_time'].iloc[0] - df['start_time'].iloc[0]).total_seconds())
#     # Q2-Q10 durations
#     for i in range(1, len(df)):
#         durations.append((df['next_question_time'].iloc[i] - df['next_question_time'].iloc[i-1]).total_seconds())
        
#     # Generate clips
#     current_time = DURATION_LAG
    
#     for i, duration in enumerate(durations, 1):
#         start = current_time
#         end = current_time + duration
#         output_path = os.path.join(OUTPUT_FOLDER, f"question_{i}.mp4")
        
#         print(f"Clipping Q{i} | Start: {start:.2f}s | End: {end:.2f}s | Duration: {duration:.2f}s")
        
#         # FFmpeg command (Re-encoding for frame accuracy)
#         cmd = [
#             'ffmpeg', '-y',
#             '-ss', str(start),
#             '-i', VIDEO_FILE_PATH,
#             '-to', str(end),
#             '-c:v', 'libx264', '-crf', '18',
#             '-c:a', 'aac',
#             output_path
#         ]
        
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
#         # Advance the clock
#         current_time = end

#     print(f"\nSuccess! All 10 clips saved in: {OUTPUT_FOLDER}")

# if __name__ == "__main__":
#     clip_video_by_duration()

##############################################


# import pandas as pd
# import subprocess
# import os


# import subprocess
# import os

# # ==========================================
# # CONFIGURATION
# # ==========================================

# VIDEO_FILE_PATH =  r"D:\000_data_collection_hti\girish_thermal_clipping.wmv"      
# MARKER_FILE_PATH = r"D:\000_data_collection_hti\girish-demo-clippong-markers.csv"
# OUTPUT_FOLDER = r"D:\000_data_collection_hti"
# DURATION_LAG = 21.0  # The confirmed start of Q1
# # ==========================================

# # LIST YOUR 10 EXACT DURATIONS HERE (in seconds)
# # You can edit these numbers if a clip is too long or too short
# DURATIONS = [3.9, 4.0, 3.5, 5.0, 4.5, 3.9, 4.5, 4.7, 3.0, 5.5] 
# # ==========================================

# def clip_video_hardcoded():
#     if not os.path.exists(OUTPUT_FOLDER):
#         os.makedirs(OUTPUT_FOLDER)

#     current_time = DURATION_LAG
    
#     for i, duration in enumerate(DURATIONS, 1):
#         start = current_time
#         end = current_time + duration
#         output_path = os.path.join(OUTPUT_FOLDER, f"question_{i}.mp4")
        
#         print(f"Clipping Q{i} | Start: {start:.2f}s | End: {end:.2f}s | Duration: {duration:.2f}s")
        
#         # FFmpeg command
#         cmd = [
#             'ffmpeg', '-y',
#             '-ss', str(start),
#             '-i', VIDEO_FILE_PATH,
#             '-to', str(end),
#             '-c:v', 'libx264', '-crf', '18',
#             '-c:a', 'aac',
#             output_path
#         ]
        
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
#         # Advance clock to the exact end of this clip
#         current_time = end

#     print(f"\nSuccess! All 10 clips saved in: {OUTPUT_FOLDER}")

# if __name__ == "__main__":
#     clip_video_hardcoded()

#################################

import pandas as pd
import subprocess
import os

# ==========================================================
# USER INPUTS (EDIT THESE ONLY)
# ==========================================================

# Path to interrogation video
VIDEO_FILE = r"D:\000_data_collection_hti\girish_thermal_clipping.wmv"     

# Path to marker CSV file
MARKER_FILE =  r"D:\000_data_collection_hti\girish-demo-clippong-markers.csv"

# Folder where clipped question videos will be saved
OUTPUT_FOLDER = r"D:\000_data_collection_hti"

# Time (in seconds) where Question 1 starts in the video
# Example:
# If Q1 begins at 13 sec in the video, put 13
# If Q1 begins at 2 min 15 sec, put 135
Q1_START_IN_VIDEO = 21

# ==========================================================


def compute_question_durations(df):
    """
    Computes duration of each question from marker file.
    """

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["next_question_time"] = pd.to_datetime(df["next_question_time"])

    durations = []

    # Q1 duration
    q1_duration = (
        df["next_question_time"].iloc[0]
        - df["start_time"].iloc[0]
    ).total_seconds()

    durations.append(q1_duration)

    # Remaining questions
    for i in range(1, len(df)):
        duration = (
            df["next_question_time"].iloc[i]
            - df["next_question_time"].iloc[i - 1]
        ).total_seconds()

        durations.append(duration)

    return durations


def format_duration(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def clip_video(video_path, output_folder, durations, q1_start):
    """
    Creates:
    question_1.mp4
    question_2.mp4
    ...
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_start = q1_start

    for i, duration in enumerate(durations, start=1):

        output_file = os.path.join(
            output_folder,
            f"0_question_{i}.mp4"
        )

        print("-" * 60)
        print(f"Question {i}")
        print(f"Start Time : {current_start:.3f} sec")
        print(f"Duration   : {duration:.3f} sec")
        print(f"Output     : {output_file}")

        cmd = [
    "ffmpeg",
    "-y",
    "-i", video_path,
    "-ss", str(current_start),
    "-t", str(duration),
    "-c:v", "libx264",
    "-crf", "18",
    output_file
]


        subprocess.run(cmd, check=True)

        current_start += duration

    print("\nAll clips generated successfully!")


def main():

    print("\nLoading marker file...")
    df = pd.read_csv(MARKER_FILE)

    print("Computing question durations...")
    durations = compute_question_durations(df)

    print("\nQuestion Durations")
    print("=" * 60)

    for i, d in enumerate(durations, start=1):
        print(f"Q{i:02d} : {format_duration(d)}")

    print("\nStarting clipping process...\n")

    clip_video(
        VIDEO_FILE,
        OUTPUT_FOLDER,
        durations,
        Q1_START_IN_VIDEO
    )


if __name__ == "__main__":
    main()