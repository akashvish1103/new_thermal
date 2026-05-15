# this custom module will have all the intermediate procesing stuffs required.


import cv2
import numpy as np
# ============================================================
# CLAHE (Create Once)
# ============================================================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

def get_transformed_image(grey_frame):
    """
    Applies contrast enhancement pipeline to the input grayscale frame and returns the enhanced greyscaled frame.
    """

    # ========================================================
    # CONTRAST ENHANCEMENT PIPELINE
    # ========================================================

    # Stretch contrast
    stretched = cv2.normalize(
        grey_frame,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # CLAHE
    enhanced = clahe.apply(stretched)

    # Gamma correction
    gamma_corrected = (
        np.power(enhanced / 255.0, 0.6) * 255
    ).astype(np.uint8)

    # Unsharp mask
    blurred = cv2.GaussianBlur(
        gamma_corrected,
        (0, 0),
        3
    )

    # sharpened = cv2.addWeighted(
    #     gamma_corrected,
    #     1.5,
    #     blurred,
    #     -0.5,
    #     0
    # )

    sharpened_grey_frame = cv2.addWeighted(                        # sharpened the edges
    gamma_corrected,
    2,
    blurred,
    -1.5,
    0
   )
    
    return sharpened_grey_frame                                  # returning the enhanced greyscaled frame