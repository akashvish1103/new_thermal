# ============================================================
# ROI Mean Temperature Analysis Across 17 Question Videos
# Baseline (Q1-Q4) vs Crime-Related (Q5-Q17)
#
# USAGE:
#   1. Set VIDEO_FOLDER to the folder containing your 17 videos.
#   2. Videos must be named so they sort in question order,
#      e.g. q1_krishna.mp4, q2_krishna.mp4 ... q17_krishna.mp4
#      OR  01_question.mp4, 02_question.mp4 ... 17_question.mp4
#   3. Set UTILITIES_PATH to the folder that contains your utilities.py
#   4. Run:  python roi_analysis_across_questions.py
#
# OUTPUT:
#   - roi_means_across_questions.png  (saved in same folder as script)
#   - Console printout of per-question ROI means
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import glob
import re

# ============================================================
# >>> USER CONFIGURATION  <<<
# ============================================================

# Folder that contains your 17 question videos
VIDEO_FOLDER = r"cropped_clips\jayesh"          # <-- CHANGE THIS

# Folder that contains utilities.py  (the one you uploaded)
UTILITIES_PATH = r"basics\forehead\utilities.py"         # <-- CHANGE THIS

# Number of baseline questions
NUM_BASELINE = 4       # Q1–Q4 are baseline, Q5–Q17 are crime-related

# Pixel-to-temperature mapping  (same as your existing code)
def get_temp_from_pixel(pixel_value):
    m = 0.05891454
    b = 30.07676744
    return m * pixel_value + b

# ============================================================
# Dynamic import of utilities.py from user-specified path
# ============================================================
sys.path.insert(0, UTILITIES_PATH)
import utilities as ut

# ============================================================
# MediaPipe setup (created once, reused for all videos)
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
# Helper: natural sort so q2 < q10 < q17
# ============================================================
def natural_sort_key(path):
    fname = os.path.basename(path)
    parts = re.split(r'(\d+)', fname)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

# ============================================================
# Helper: extract ROI crops from a single frame
# Returns dict of roi_name -> mean_pixel_value  (or None if not detected)
# ============================================================
def extract_roi_means_from_frame(frame, face_landmarks):
    h, w, _ = frame.shape
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    means = {}

    # ---- FOREHEAD ----
    try:
        lm_f, tm_f, rm_f, bm_f = ut.get_forehead_coordinates(frame.copy(), face_landmarks, False)
        crop = grey[tm_f:bm_f, lm_f:rm_f]
        if crop.size > 0:
            means['Forehead'] = np.mean(crop)
    except Exception:
        means['Forehead'] = None

    # ---- NOSE ----
    try:
        lm_n, tm_n, rm_n, bm_n = ut.get_nose_coordinates(frame.copy(), face_landmarks)
        crop = grey[tm_n:bm_n, lm_n:rm_n]
        if crop.size > 0:
            means['Nose'] = np.mean(crop)
    except Exception:
        means['Nose'] = None

    # ---- LEFT EYE INNER CORNER ----
    # ---- RIGHT EYE INNER CORNER ----
    try:
        tl, br, tr, bl = ut.get_eyes_coordinates(frame.copy(), grey, face_landmarks)
        # Left eye ROI
        crop_l = grey[tl[1]:br[1], tl[0]:br[0]]
        if crop_l.size > 0:
            means['Left Eye'] = np.mean(crop_l)
        # Right eye ROI
        crop_r = grey[tr[1]:bl[1], bl[0]:tr[0]]
        if crop_r.size > 0:
            means['Right Eye'] = np.mean(crop_r)
    except Exception:
        means['Left Eye'] = None
        means['Right Eye'] = None

    # ---- LEFT CHEEK + RIGHT CHEEK (polygon masked mean) ----
    try:
        points_left, points_right = ut.get_cheeks_coordinates(frame.copy(), face_landmarks, [], [])

        for side, pts, key in [('left', points_left, 'Left Cheek'),
                                ('right', points_right, 'Right Cheek')]:
            if len(pts) >= 3:
                polygon = np.array(pts, dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                pixels = grey[mask == 255]
                if pixels.size > 0:
                    means[key] = np.mean(pixels)
                else:
                    means[key] = None
            else:
                means[key] = None
    except Exception:
        means['Left Cheek'] = None
        means['Right Cheek'] = None

    return means

# ============================================================
# Process a single video → returns dict of roi → mean over all frames
# ============================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS={fps:.1f}  Frames={total_frames}")

    roi_accumulator = {
        'Forehead': [], 'Nose': [],
        'Left Eye': [], 'Right Eye': [],
        'Left Cheek': [], 'Right Cheek': []
    }

    frame_count = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = ut.get_transformed_image(grey)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            detected_frames += 1
            face_lm = results.multi_face_landmarks[0]
            means = extract_roi_means_from_frame(frame, face_lm)

            for roi, val in means.items():
                if val is not None:
                    roi_accumulator[roi].append(val)

    cap.release()

    detection_rate = detected_frames / frame_count * 100 if frame_count > 0 else 0
    print(f"  Detection rate: {detection_rate:.1f}%  ({detected_frames}/{frame_count} frames)")

    # Average across all detected frames, convert to temperature
    video_means = {}
    for roi, values in roi_accumulator.items():
        if values:
            video_means[roi] = get_temp_from_pixel(np.mean(values))
        else:
            video_means[roi] = np.nan

    return video_means

# ============================================================
# MAIN
# ============================================================
def main():
    # Collect and sort videos
    extensions = ['*.mp4', '*.wmv', '*.avi', '*.mpg', '*.mpeg', '*.mov']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(VIDEO_FOLDER, ext)))
        video_files.extend(glob.glob(os.path.join(VIDEO_FOLDER, ext.upper())))

    video_files = sorted(set(video_files), key=natural_sort_key)

    if len(video_files) == 0:
        print(f"[ERROR] No video files found in: {VIDEO_FOLDER}")
        return

    print(f"\nFound {len(video_files)} video(s) in: {VIDEO_FOLDER}")
    for i, vf in enumerate(video_files):
        print(f"  Q{i+1}: {os.path.basename(vf)}")

    if len(video_files) != 17:
        print(f"\n[WARNING] Expected 17 videos, found {len(video_files)}. Proceeding anyway.")

    # ---- Process each video ----
    all_results = []   # list of dicts, one per question video
    question_labels = []

    for i, vf in enumerate(video_files):
        q_num = i + 1
        label = f"Q{q_num}"
        question_labels.append(label)
        print(f"\n[{label}] Processing: {os.path.basename(vf)}")
        result = process_video(vf)
        if result is None:
            result = {roi: np.nan for roi in ['Forehead', 'Nose', 'Left Eye', 'Right Eye', 'Left Cheek', 'Right Cheek']}
        all_results.append(result)
        print(f"  Means (°C): { {k: f'{v:.2f}' for k, v in result.items()} }")

    # ---- Organise data for plotting ----
    roi_names = ['Forehead', 'Nose', 'Left Eye', 'Right Eye', 'Left Cheek', 'Right Cheek']
    colors     = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653']
    markers    = ['o', 's', '^', 'D', 'v', 'P']

    x = np.arange(1, len(video_files) + 1)

    roi_data = {roi: [all_results[i].get(roi, np.nan) for i in range(len(all_results))]
                for roi in roi_names}

    # ---- Plot ----
    fig, axes = plt.subplots(len(roi_names), 1,
                             figsize=(14, 3.5 * len(roi_names)),
                             sharex=True)
    fig.suptitle("ROI Mean Temperature Across Questions\nBaseline (Q1–Q4)  vs  Crime-Related (Q5–Q17)",
                 fontsize=14, fontweight='bold', y=1.01)

    for ax, roi, color, marker in zip(axes, roi_names, colors, markers):
        values = roi_data[roi]

        # Shade baseline region
        ax.axvspan(0.5, NUM_BASELINE + 0.5,
                   alpha=0.12, color='green', label='Baseline (Q1–Q4)')
        # Shade crime region
        ax.axvspan(NUM_BASELINE + 0.5, len(video_files) + 0.5,
                   alpha=0.08, color='red', label='Crime-related')

        # Vertical divider line
        ax.axvline(x=NUM_BASELINE + 0.5, color='gray', linestyle='--', linewidth=1)

        # Mean lines for each phase
        baseline_vals = [v for v in values[:NUM_BASELINE] if not np.isnan(v)]
        crime_vals    = [v for v in values[NUM_BASELINE:] if not np.isnan(v)]
        if baseline_vals:
            ax.axhline(np.mean(baseline_vals), xmin=0,
                       xmax=NUM_BASELINE / len(video_files),
                       color=color, linestyle=':', linewidth=1.5, alpha=0.7)
        if crime_vals:
            ax.axhline(np.mean(crime_vals),
                       xmin=NUM_BASELINE / len(video_files), xmax=1,
                       color=color, linestyle=':', linewidth=1.5, alpha=0.7)

        # Plot signal
        ax.plot(x, values, color=color, marker=marker,
                linewidth=1.8, markersize=7, label=roi)

        # Annotate each point with its value
        for xi, yi in zip(x, values):
            if not np.isnan(yi):
                ax.annotate(f'{yi:.1f}',
                            xy=(xi, yi),
                            xytext=(0, 7),
                            textcoords='offset points',
                            ha='center', fontsize=7, color=color)

        ax.set_ylabel("Temp (°C)", fontsize=9)
        ax.set_title(roi, fontsize=10, fontweight='bold', loc='left')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(0.5, len(video_files) + 0.5)

        # Custom legend
        baseline_patch = mpatches.Patch(color='green', alpha=0.3, label='Baseline')
        crime_patch    = mpatches.Patch(color='red',   alpha=0.2, label='Crime-related')
        ax.legend(handles=[baseline_patch, crime_patch],
                  loc='upper right', fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(question_labels, fontsize=9)
    axes[-1].set_xlabel("Question", fontsize=10)

    plt.tight_layout()

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "roi_means_across_questions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Plot saved to: {output_path}")
    plt.show()

    # ---- Print summary table ----
    print("\n" + "="*70)
    print(f"{'Q':<5}" + "".join(f"{roi[:10]:<13}" for roi in roi_names))
    print("-"*70)
    for i, label in enumerate(question_labels):
        row = f"{label:<5}"
        for roi in roi_names:
            v = roi_data[roi][i]
            row += f"{v:<13.2f}" if not np.isnan(v) else f"{'N/A':<13}"
        print(row)

    # ---- Delta summary: crime mean - baseline mean ----
    print("\n--- Delta (Crime Mean - Baseline Mean) ---")
    for roi in roi_names:
        b_vals = [roi_data[roi][i] for i in range(NUM_BASELINE) if not np.isnan(roi_data[roi][i])]
        c_vals = [roi_data[roi][i] for i in range(NUM_BASELINE, len(video_files)) if not np.isnan(roi_data[roi][i])]
        if b_vals and c_vals:
            delta = np.mean(c_vals) - np.mean(b_vals)
            print(f"  {roi:<15}: {delta:+.3f} °C  (baseline={np.mean(b_vals):.2f}, crime={np.mean(c_vals):.2f})")


if __name__ == "__main__":
    main()