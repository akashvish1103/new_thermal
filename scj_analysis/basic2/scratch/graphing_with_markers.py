import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD LOG CSV
# ============================================================

LOG_PATH = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\44_2026-07-07\01_Passive_Profiling\44_Passive_Thermal_25_40_roi_log.csv"

df = pd.read_csv(LOG_PATH)

# ============================================================
# LOAD QUESTION MARKERS CSV
# ============================================================
# File format:
#   start_time              -> absolute UTC timestamp of Q1 start (video recording start)
#   next_question_time      -> absolute UTC timestamp each subsequent question started
#
# We convert these absolute timestamps into "seconds since video start" so they
# line up with the ROI log's `time_sec` column, then use them as Q1, Q2, ... labels.

MARKERS_PATH = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\44_2026-07-07\01_Passive_Profiling\44_Passive_Markers.csv"

markers_df = pd.read_csv(MARKERS_PATH)

# start_time only has a value on the first row -- that IS the video's recording start
video_start_utc = pd.to_datetime(markers_df["start_time"].dropna().iloc[0], utc=True)

next_question_utc = pd.to_datetime(markers_df["next_question_time"], utc=True)

# Q1 starts at elapsed 0.0 (the recording start itself);
# Q2, Q3, ... start at each next_question_time, converted to elapsed seconds
question_elapsed_sec = [0.0] + [
    (t - video_start_utc).total_seconds() for t in next_question_utc
]
question_labels = [f"Q{i+1}" for i in range(len(question_elapsed_sec))]

# ============================================================
# SETTINGS
# ============================================================

SMA_WINDOW = 50                         # taking 50 frames for smoothing

ROIS = {
    # "Breathing": "breathing_mean_temp",
    "Forehead": "forehead_mean_temp",
    "Left Cheek": "cheek_L_mean_temp",
    "Right Cheek": "cheek_R_mean_temp",
    # "Left Eye": "eye_L_mean_temp",
    # "Right Eye": "eye_R_mean_temp",
    "Nose": "nose_mean_temp"
}

# ============================================================
# TIME
# ============================================================

time = pd.to_numeric(df["time_sec"], errors="coerce")


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(16, 7))

for roi_name, column in ROIS.items():

    # Mean ROI temperature in °C
    temperature = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 50-frame Simple Moving Average
    # --------------------------------------------------------
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

# ============================================================
# QUESTION MARKERS -- tick marks (no lines) on a secondary top axis,
# so the bottom axis stays pure seconds and the top axis shows Q1, Q2, ...
# ============================================================

# only keep markers that actually fall within the plotted time range,
# in case the log covers less of the session than the full markers file
in_range = [
    (t, lbl) for t, lbl in zip(question_elapsed_sec, question_labels)
    if time.min() <= t <= time.max()
]

if in_range:
    marker_times, marker_labels = zip(*in_range)

    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(marker_times)
    ax_top.set_xticklabels(marker_labels, rotation=90, fontsize=7)
    ax_top.tick_params(axis="x", length=6)
else:
    print("WARNING: no question markers fall within this log's time range "
          "-- check that LOG_PATH and MARKERS_PATH are from the same session/video.")

# ============================================================
# GRAPH FORMATTING
# ============================================================

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Mean ROI Temperature (°C)")

ax.set_title(
    f"Mean Temperature of Facial ROIs "
    f"({SMA_WINDOW}-Frame SMA) — with Question Markers"
)

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()