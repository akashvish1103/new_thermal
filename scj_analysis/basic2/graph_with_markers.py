import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD LOG CSV
# ============================================================

# LOG_PATH = (
#     r"D:\000_ofc_thermalData\sorted_data\48"
#     r"\48_Passive_Thermal_30_40_log_stats.csv"
# )

LOG_PATH = r"D:\000_ofc_thermalData\sorted_data\48\48_Passive_Thermal_30_40_log_stats.csv"

# ============================================================
# LOAD INTERROGATION MARKER CSV
#
# IMPORTANT:
#
# Q1 starts exactly when the thermal video starts recording.
# Therefore:
#
#     Q1 = video time 0
#
# The marker file contains:
#
#     start_time
#     next_question_time
#
# The first row's start_time = Q1 start.
# Then each next_question_time gives the start of the
# following question.
#
# The final next_question_time is treated as the end of the
# final question, NOT as another question marker.
# ============================================================

# MARKER_PATH = (
#     r"D:\000_ofc_thermalData\sorted_data\48"
#     r"\48_Passive_Markers.csv"
# )

MARKER_PATH = r"D:\000_ofc_thermalData\sorted_data\48\48_Passive_Markers.csv"
# ============================================================
# SETTINGS
# ============================================================

SMA_WINDOW = 50

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
# LOAD LOG
# ============================================================

df = pd.read_csv(LOG_PATH)


# ============================================================
# TIME FROM VIDEO LOG
# ============================================================

time = pd.to_numeric(
    df["time_sec"],
    errors="coerce"
)


# ============================================================
# LOAD QUESTION MARKERS
# ============================================================

markers_df = pd.read_csv(
    MARKER_PATH
)


# ------------------------------------------------------------
# Convert marker timestamps to datetime.
#
# utc=True handles timestamps such as:
#
# 2026-07-08T05:32:29.517Z
#
# and makes the subtraction safe.
# ------------------------------------------------------------

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


# ============================================================
# QUESTION START TIMES
# ============================================================

question_markers = []


# ------------------------------------------------------------
# Q1:
#
# The first start_time is exactly the instant the thermal
# video started recording.
#
# Therefore Q1 is video time 0.
# ------------------------------------------------------------

video_start_time = start_times.iloc[0]


if pd.isna(video_start_time):

    raise ValueError(
        "The first start_time in the marker file is empty "
        "or could not be parsed."
    )


question_markers.append(
    {
        "question": "Q1",
        "absolute_time": video_start_time,
        "video_time_sec": 0.0
    }
)


# ------------------------------------------------------------
# Q2, Q3, Q4, ...
#
# Every next_question_time starts the following question.
#
# IMPORTANT:
#
# We stop one row before the final marker because the final
# next_question_time represents the end of the last question,
# not the start of another question.
# ------------------------------------------------------------

for i in range(len(next_question_times) - 1):

    next_time = next_question_times.iloc[i]

    if pd.isna(next_time):
        continue

    relative_seconds = (
        next_time - video_start_time
    ).total_seconds()

    question_number = i + 2

    question_markers.append(
        {
            "question": f"Q{question_number}",
            "absolute_time": next_time,
            "video_time_sec": relative_seconds
        }
    )


# ============================================================
# PRINT QUESTION MARKERS
# ============================================================

print()
print("=" * 70)
print("QUESTION MARKERS")
print("=" * 70)

for marker in question_markers:

    seconds = marker["video_time_sec"]

    minutes = int(seconds // 60)

    remaining_seconds = seconds % 60

    formatted_time = (
        f"{minutes:02d}:"
        f"{remaining_seconds:05.2f}"
    )

    print(
        f"{marker['question']:>4}  "
        f"Video Time: {formatted_time}"
    )

print("=" * 70)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 8)
)


# ============================================================
# PLOT TEMPERATURE SIGNALS
# ============================================================

for roi_name, column in ROIS.items():

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
# QUESTION MARKERS
# ============================================================

# ------------------------------------------------------------
# Determine top of graph so question labels can be placed
# above the temperature curves.
# ------------------------------------------------------------

all_temperature_values = []

for _, column in ROIS.items():

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    all_temperature_values.extend(
        values.dropna().tolist()
    )


if all_temperature_values:

    y_min = min(all_temperature_values)
    y_max = max(all_temperature_values)

    y_range = y_max - y_min

    if y_range == 0:
        y_range = 1.0

    label_y = (
        y_max
        + 0.06 * y_range
    )

    ax.set_ylim(
        y_min - 0.05 * y_range,
        y_max + 0.15 * y_range
    )

else:

    label_y = 1.0


# ------------------------------------------------------------
# Draw one vertical line for every question start.
# ------------------------------------------------------------

for marker in question_markers:

    q = marker["question"]

    marker_time = marker["video_time_sec"]

    ax.axvline(
        x=marker_time,
        linestyle="--",
        linewidth=1,
        alpha=0.55
    )

    # --------------------------------------------------------
    # Put Q1, Q2, Q3... at the top of each marker.
    # --------------------------------------------------------

    ax.text(
        marker_time,
        label_y,
        q,
        rotation=90,
        verticalalignment="bottom",
        horizontalalignment="center",
        fontsize=8
    )


# ============================================================
# GRAPH FORMATTING
# ============================================================

ax.set_xlabel(
    "Video Time (seconds)"
)

ax.set_ylabel(
    "Mean ROI Temperature (°C)"
)

ax.set_title(
    f"Mean Temperature of Facial ROIs "
    f"({SMA_WINDOW}-Frame SMA) "
    f"with Question Start Markers"
)

ax.legend()

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

plt.show()