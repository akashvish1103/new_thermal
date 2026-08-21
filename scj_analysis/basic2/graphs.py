import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD LOG CSV
# ============================================================

LOG_PATH = r"d:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40_roi_log.csv"
LOG_PATH = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal_wmv_30_40_roi_log.csv"

df = pd.read_csv(LOG_PATH)

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

plt.figure(figsize=(14, 7))

for roi_name, column in ROIS.items():

    # Mean ROI temperature in °C
    temperature = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 5-frame Simple Moving Average
    # --------------------------------------------------------
    temperature_sma = temperature.rolling(
        window=SMA_WINDOW,
        min_periods=1
    ).mean()

    plt.plot(
        time,
        temperature_sma,
        linewidth=2,
        label=roi_name
    )


# ============================================================
# GRAPH FORMATTING
# ============================================================

plt.xlabel("Time (seconds)")
plt.ylabel("Mean ROI Temperature (°C)")

plt.title(
    f"Mean Temperature of Facial ROIs "
    f"(5-Frame SMA)"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()
plt.show()