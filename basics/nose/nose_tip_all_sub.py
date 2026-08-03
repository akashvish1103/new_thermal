import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

import utilities as ut


# ==================================================
# SETTINGS
# ==================================================

VIDEO_FOLDER = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\all_grey_manual_mp4"

NORMALIZED_LENGTH = 1000

SMA_WINDOW = 30
EWMA_BETA = 0.95

TEMP_A = 0.05891454
TEMP_B = 30.07676744


# ==================================================
# MEDIAPIPE
# ==================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def pixel_to_temperature(pixel):
    return TEMP_A * pixel + TEMP_B


def sma(signal, window=30):

    signal = np.array(signal)

    kernel = np.ones(window) / window

    return np.convolve(
        signal,
        kernel,
        mode='same'
    )


def ewma(signal, beta=0.95):

    signal = np.array(signal)

    ewma_signal = np.zeros_like(signal)

    ewma_signal[0] = signal[0]

    for i in range(1, len(signal)):
        ewma_signal[i] = (
            beta * ewma_signal[i-1]
            +
            (1-beta) * signal[i]
        )

    return ewma_signal


def normalize_signal(signal, target_length=1000):

    old_x = np.linspace(
        0,
        1,
        len(signal)
    )

    new_x = np.linspace(
        0,
        1,
        target_length
    )

    return np.interp(
        new_x,
        old_x,
        signal
    )


# ==================================================
# PROCESS SINGLE VIDEO
# ==================================================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    nose_temperature = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        grey = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        enhanced = ut.get_transformed_image(grey)

        rgb = cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR
        )

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            left, top, right, bottom = \
                ut.get_nose_coordinates(
                    frame,
                    face_landmarks
                )

            h, w = grey.shape

            left = max(0, left)
            top = max(0, top)

            right = min(w, right)
            bottom = min(h, bottom)

            if right > left and bottom > top:

                roi = grey[
                    top:bottom,
                    left:right
                ]

                mean_pixel = np.mean(roi)

                temp = pixel_to_temperature(
                    mean_pixel
                )

                nose_temperature.append(temp)

    cap.release()

    return np.array(nose_temperature)


# ==================================================
# GET ALL VIDEOS
# ==================================================

video_files = []

for file in os.listdir(VIDEO_FOLDER):

    if file.lower().endswith(
        (
            ".mp4",
            ".avi",
            ".wmv",
            ".mpg",
            ".mpeg"
        )
    ):
        video_files.append(
            os.path.join(
                VIDEO_FOLDER,
                file
            )
        )

video_files.sort()

print(
    f"Found {len(video_files)} videos"
)


# ==================================================
# PROCESS ALL SUBJECTS
# ==================================================

all_subjects_sma = []
all_subjects_ewma = []
subject_names = []

for video in video_files:

    print(
        "Processing:",
        os.path.basename(video)
    )

    signal = process_video(video)

    if len(signal) < 100:
        continue

    signal = normalize_signal(
        signal,
        NORMALIZED_LENGTH
    )

    sma_signal = sma(
        signal,
        SMA_WINDOW
    )

    ewma_signal = ewma(
        signal,
        EWMA_BETA
    )

    all_subjects_sma.append(
        sma_signal
    )

    all_subjects_ewma.append(
        ewma_signal
    )

    subject_names.append(
        os.path.splitext(
            os.path.basename(video)
        )[0]
    )


# ==================================================
# GRAPH 1 : SMA
# ==================================================

plt.figure(figsize=(16,8))

for sig, name in zip(
        all_subjects_sma,
        subject_names):

    plt.plot(
        sig,
        label=name,
        linewidth=1.5
    )

plt.title(
    "Nose Tip Temperature (SMA)"
)

plt.xlabel(
    "Interrogation Progress (%)"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.grid(True)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.show()


# ==================================================
# GRAPH 2 : EWMA
# ==================================================

plt.figure(figsize=(16,8))

for sig, name in zip(
        all_subjects_ewma,
        subject_names):

    plt.plot(
        sig,
        label=name,
        linewidth=1.5
    )

plt.title(
    "Nose Tip Temperature (EWMA)"
)

plt.xlabel(
    "Interrogation Progress (%)"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.grid(True)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.show()