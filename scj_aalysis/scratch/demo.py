# import cv2
# import numpy as np

# # ============================================================
# # CLAHE (Create Once)
# # ============================================================

# clahe = cv2.createCLAHE(
#     clipLimit=2.0,
#     tileGridSize=(8, 8)
# )

# # ============================================================
# # GRAYSCALE ENHANCEMENT
# # ============================================================

# def transform_gray(gray):

#     # Contrast Stretch
#     stretched = cv2.normalize(
#         gray,
#         None,
#         0,
#         255,
#         cv2.NORM_MINMAX
#     )

#     # CLAHE
#     enhanced = clahe.apply(stretched)

#     # Gamma
#     gamma = (
#         np.power(enhanced / 255.0, 0.6) * 255
#     ).astype(np.uint8)

#     # Blur
#     blurred = cv2.GaussianBlur(
#         gamma,
#         (0, 0),
#         3
#     )

#     # Sharpen
#     sharpened = cv2.addWeighted(
#         gamma,
#         2,
#         blurred,
#         -1.5,
#         0
#     )

#     return sharpened


# # ============================================================
# # RGB ENHANCEMENT
# # (Apply enhancement only on L channel)
# # ============================================================

# def transform_rgb(rgb):

#     lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)

#     l, a, b = cv2.split(lab)

#     l = transform_gray(l)

#     lab = cv2.merge((l, a, b))

#     rgb_enhanced = cv2.cvtColor(
#         lab,
#         cv2.COLOR_LAB2BGR
#     )

#     return rgb_enhanced


# # ============================================================
# # VIDEO
# # ============================================================

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"

# cap = cv2.VideoCapture(video_path)

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     # ---------------------------------------
#     # Original RGB
#     # ---------------------------------------
#     original_rgb = frame.copy()

#     # ---------------------------------------
#     # Original Gray
#     # ---------------------------------------
#     gray = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2GRAY
#     )

#     # ---------------------------------------
#     # Enhanced Gray
#     # ---------------------------------------
#     gray_transformed = transform_gray(gray)

#     # ---------------------------------------
#     # Enhanced RGB
#     # ---------------------------------------
#     rgb_transformed = transform_rgb(frame)

#     # ---------------------------------------
#     # Show
#     # ---------------------------------------

#     cv2.imshow("1. Original RGB", original_rgb)

#     cv2.imshow("2. Original Gray", gray)

#     cv2.imshow(
#         "3. Enhanced Gray",
#         gray_transformed
#     )

#     cv2.imshow(
#         "4. Enhanced RGB",
#         rgb_transformed
#     )

#     key = cv2.waitKey(25)

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# ============================================================
# CLAHE
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# ============================================================
# Dummy function for Trackbars
# ============================================================

def nothing(x):
    pass


# ============================================================
# Create Control Window
# ============================================================

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)

# alpha = value / 10
cv2.createTrackbar(
    "Contrast x10",
    "Controls",
    6,          # default = 1.2
    30,          # max = 3.0
    nothing
)

# beta
cv2.createTrackbar(
    "Brightness",
    "Controls",
    29,          # default = 0
    100,
    nothing
)


# ============================================================
# GRAYSCALE ENHANCEMENT
# ============================================================

def transform_gray(gray, alpha=1.2, beta=0):

    # --------------------------------------------------------
    # Brightness + Contrast
    # --------------------------------------------------------

    adjusted = cv2.convertScaleAbs(
        gray,
        alpha=alpha,
        beta=beta
    )

    # --------------------------------------------------------
    # Contrast Stretch
    # --------------------------------------------------------

    stretched = cv2.normalize(
        adjusted,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    enhanced = clahe.apply(stretched)

    # --------------------------------------------------------
    # Gamma Correction
    # --------------------------------------------------------

    gamma = (
        np.power(enhanced / 255.0, 0.6) * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gamma,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        gamma,
        2,
        blurred,
        -1.5,
        0
    )

    return sharpened


# ============================================================
# RGB Enhancement
# ============================================================

def transform_rgb(rgb, alpha=1.2, beta=0):

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    l = transform_gray(
        l,
        alpha,
        beta
    )

    lab = cv2.merge((l, a, b))

    rgb_enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return rgb_enhanced


# ============================================================
# VIDEO
# ============================================================

# video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\71_2026-07-15\02_Psychometric_Tests\71_HDRS_Thermal.mpg"
video_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg"

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Read Trackbar Values
    # --------------------------------------------------------

    alpha = cv2.getTrackbarPos(
        "Contrast x10",
        "Controls"
    ) / 10.0

    beta = cv2.getTrackbarPos(
        "Brightness",
        "Controls"
    ) - 50

    # Prevent alpha = 0
    alpha = max(alpha, 0.1)

    # --------------------------------------------------------
    # Original Images
    # --------------------------------------------------------

    original_rgb = frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Enhanced Images
    # --------------------------------------------------------

    gray_transformed = transform_gray(
        gray,
        alpha,
        beta
    )

    rgb_transformed = transform_rgb(
        frame,
        alpha,
        beta
    )

    # --------------------------------------------------------
    # Display Parameters
    # --------------------------------------------------------

    info = f"Alpha={alpha:.1f}  Beta={beta}"

    cv2.putText(
        gray_transformed,
        info,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2
    )

    cv2.putText(
        rgb_transformed,
        info,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Show Images
    # --------------------------------------------------------

    cv2.imshow(
        "1. Original RGB",
        original_rgb
    )

    cv2.imshow(
        "2. Original Gray",
        gray
    )

    cv2.imshow(
        "3. Enhanced Gray",
        gray_transformed
    )

    cv2.imshow(
        "4. Enhanced RGB",
        rgb_transformed
    )

    key = cv2.waitKey(25)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()