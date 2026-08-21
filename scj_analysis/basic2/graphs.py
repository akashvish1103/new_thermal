# Use this for both Passive and HDRS (auto adjust according to the LOG file name)
# Final Graphing Code by AKash V.


import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD LOG CSV
# ============================================================

# Change ONLY this line when you want to plot another log.
#
# If this is HDRS:
#   48_HDRS_Thermal_30_40_log_stats.csv
#       -> automatically uses 48_HDRS_Markers.csv
#
# If this is Passive:
#   48_Passive_Thermal_30_40_log_stats.csv
#       -> automatically uses 48_Passive_Markers.csv

LOG_PATH = r"D:\000_ofc_thermalData\sorted_data\48\48_HDRS_Thermal_30_40_roi_log.csv"
LOG_PATH = r"D:\000_ofc_thermalData\sorted_data\48\48_Passive_Thermal_30_40_log_stats.csv"


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
# AUTOMATIC MARKER-FILE SELECTION
# ============================================================
#
# The marker file is selected from the LOG filename.
#
# 48_Passive_Thermal_30_40_log_stats.csv
#          ↓
# 48_Passive_Markers.csv
#
# 48_HDRS_Thermal_30_40_log_stats.csv
#          ↓
# 48_HDRS_Markers.csv
#
# Therefore you only need to change LOG_PATH.
# ============================================================

log_filename = os.path.basename(LOG_PATH)

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
        "Could not determine whether the log is Passive or HDRS.\n"
        f"Log filename: {log_filename}"
    )


MARKER_PATH = os.path.join(
    os.path.dirname(LOG_PATH),
    marker_filename
)


# ============================================================
# PRINT FILES BEING USED
# ============================================================

print()
print("=" * 70)
print("FILES")
print("=" * 70)
print(f"Log file:    {LOG_PATH}")
print(f"Marker file: {MARKER_PATH}")
print("=" * 70)


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
# LOAD MARKER CSV
# ============================================================

markers_df = pd.read_csv(
    MARKER_PATH
)


# ============================================================
# FUNCTION: GET QUESTION MARKERS
# ============================================================

def get_question_markers(markers_df):
    """
    Supports BOTH marker-file formats.

    PASSIVE format:
        start_time
        next_question_time

    HDRS format:
        marker_type
        question_id
        question_order
        timestamp_iso
        timestamp_ms

    Returns a common list containing:

        question
        absolute_time
        video_time_sec

    Q1 is always video time 0 because the thermal video
    starts recording at the beginning of Q1.
    """

    # ========================================================
    # HDRS MARKER FORMAT
    # ========================================================

    if "question_order" in markers_df.columns:

        # ----------------------------------------------------
        # Find video start
        # ----------------------------------------------------

        start_rows = markers_df[
            markers_df["marker_type"]
            .astype(str)
            .str.lower()
            == "start"
        ]

        if start_rows.empty:

            raise ValueError(
                "HDRS marker file does not contain a "
                "'start' marker."
            )

        video_start_time = pd.to_datetime(
            start_rows.iloc[0]["timestamp_iso"],
            errors="coerce",
            utc=True
        )

        if pd.isna(video_start_time):

            raise ValueError(
                "Could not parse the HDRS video start time."
            )

        # ----------------------------------------------------
        # HDRS question starts are the LOCK rows.
        # ----------------------------------------------------

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

        # Use question_order for Q1, Q2, Q3...
        lock_rows = lock_rows.sort_values(
            "question_order"
        )

        question_markers = []

        for _, row in lock_rows.iterrows():

            question_number = int(
                row["question_order"]
            )

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

            question_markers.append(
                {
                    "question": f"Q{question_number}",   
                    "absolute_time": question_time,
                    "video_time_sec": relative_seconds
                }
            )

        # ----------------------------------------------------
        # User confirmed Q1 = video start.
        #
        # Force Q1 to exactly 0 seconds.
        # ----------------------------------------------------

        for marker in question_markers:

            if marker["question"] == "Q1":

                marker["video_time_sec"] = 0.0
                break

        # ----------------------------------------------------
        # Check for timestamp inconsistencies.
        #
        # We DO NOT silently correct/reorder the timestamps.
        # We use the timestamps exactly as stored.
        # ----------------------------------------------------

        for previous, current in zip(
            question_markers,
            question_markers[1:]
        ):

            if (
                current["video_time_sec"]
                <
                previous["video_time_sec"]
            ):

                print()
                print("WARNING:")
                print(
                    "HDRS marker timestamps are not "
                    "chronological according to question_order."
                )

                print(
                    f"{previous['question']} = "
                    f"{previous['video_time_sec']:.3f} sec"
                )

                print(
                    f"{current['question']} = "
                    f"{current['video_time_sec']:.3f} sec"
                )

                print(
                    "The graph will use the timestamps "
                    "exactly as stored in the marker file."
                )

                print()

        return question_markers


    # ========================================================
    # PASSIVE MARKER FORMAT
    # ========================================================

    elif (
        "start_time" in markers_df.columns
        and
        "next_question_time" in markers_df.columns
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

        # ----------------------------------------------------
        # Q1 = first start_time = video start
        # ----------------------------------------------------

        valid_start_times = start_times.dropna()

        if valid_start_times.empty:

            raise ValueError(
                "Passive marker file has no valid start_time."
            )

        video_start_time = (
            valid_start_times.iloc[0]
        )

        question_markers = [
            {
                "question": "Q1",
                "absolute_time": video_start_time,
                "video_time_sec": 0.0
            }
        ]

        # ----------------------------------------------------
        # Q2 onward:
        #
        # Each next_question_time starts the next question.
        #
        # The FINAL next_question_time is the end of the
        # final question, so it is NOT used as a question
        # start.
        # ----------------------------------------------------

        valid_next_times = (
            next_question_times.dropna()
        )

        for i, next_time in enumerate(
            valid_next_times.iloc[:-1],
            start=2
        ):

            relative_seconds = (
                next_time - video_start_time
            ).total_seconds()

            question_markers.append(
                {
                    "question": f"Q{i}",
                    "absolute_time": next_time,
                    "video_time_sec": relative_seconds
                }
            )

        return question_markers


    # ========================================================
    # UNKNOWN FORMAT
    # ========================================================

    else:

        raise ValueError(
            "Unknown marker-file format.\n\n"
            f"Columns found:\n{list(markers_df.columns)}\n\n"
            "Expected either Passive or HDRS marker format."
        )


# ============================================================
# CREATE QUESTION MARKERS
# ============================================================

question_markers = get_question_markers(
    markers_df
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

    minutes = int(
        seconds // 60
    )

    remaining_seconds = (
        seconds % 60
    )

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
# CREATE PLOT
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
    # Simple Moving Average
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
# FIND Y RANGE FOR QUESTION LABELS
# ============================================================

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

    y_min = min(
        all_temperature_values
    )

    y_max = max(
        all_temperature_values
    )

    y_range = (
        y_max - y_min
    )

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


# ============================================================
# DRAW QUESTION MARKERS
# ============================================================

# for marker in question_markers:

#     # question = marker["question"]                              # Changeed this for Q1 ---> 1
#     question = marker["question"].replace("Q", "")

#     marker_time = (
#         marker["video_time_sec"]
#     )

#     # --------------------------------------------------------
#     # Vertical marker
#     # --------------------------------------------------------

#     ax.axvline(
#         x=marker_time,
#         linestyle="--",
#         linewidth=1,
#         alpha=0.55
#     )

    # --------------------------------------------------------
    # Question label
    # --------------------------------------------------------

    # ax.text(
    #     marker_time,
    #     label_y,
    #     question,
    #     rotation=90,
    #     verticalalignment="bottom",
    #     horizontalalignment="center",
    #     fontsize=8
    # )
    
#     ax.text(
#     marker_time,
#     0.02,
#     question,
#     transform=ax.get_xaxis_transform(),
#     horizontalalignment="center",
#     verticalalignment="bottom",
#     fontsize=9
# )

for marker in question_markers:

    question = marker["question"].replace("Q", "")

    marker_time = marker["video_time_sec"]

    ax.axvline(
        x=marker_time,
        linestyle="--",
        linewidth=1,
        alpha=0.55
    )

    ax.text(
        marker_time,
        0.02,
        question,
        transform=ax.get_xaxis_transform(),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9
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