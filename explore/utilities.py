"""
forehead/utilities.py
=====================
All ROI detection helpers used by main_live_graph.py.
Each function draws on the BGR `frame` in-place AND returns
the bounding coordinates so the main loop can crop `grey`
for intensity extraction.
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  CLAHE (created once, reused every frame)
# ─────────────────────────────────────────────

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ═════════════════════════════════════════════
#  IMAGE ENHANCEMENT
# ═════════════════════════════════════════════

def get_transformed_image(grey_frame: np.ndarray) -> np.ndarray:
    """
    Contrast-enhancement pipeline for grayscale thermal frames.
    Returns an enhanced single-channel uint8 image.
    """
    # 1. Stretch contrast to full [0, 255]
    stretched = cv2.normalize(grey_frame, None, 0, 255, cv2.NORM_MINMAX)

    # 2. CLAHE for local contrast
    enhanced = clahe.apply(stretched)

    # 3. Gamma correction (γ = 0.6 → brightens midtones)
    gamma_corrected = (np.power(enhanced / 255.0, 0.6) * 255).astype(np.uint8)

    # 4. Unsharp mask — aggressive edge sharpening
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), 3)
    sharpened = cv2.addWeighted(gamma_corrected, 2.0, blurred, -1.5, 0)

    return sharpened


# ═════════════════════════════════════════════
#  ROI 1 & 2 — INNER EYE CORNERS
# ═════════════════════════════════════════════

LEFT_INNER_EYE  = 133   # MediaPipe landmark index
RIGHT_INNER_EYE = 362


def get_eyes_coordinates(frame: np.ndarray, grey: np.ndarray, face_landmarks):
    """
    Draws small ROI boxes around both inner eye corners on `frame`.

    Returns
    -------
    (top_left, bottom_right, top_right, bottom_left)
        Pixel coordinate tuples for LEFT and RIGHT eye boxes.
        LEFT  box  : top_left     → bottom_right
        RIGHT box  : top_right    → bottom_left   (note reversed x-order)
    """
    h, w, _ = frame.shape

    # Left inner corner
    lp = face_landmarks.landmark[LEFT_INNER_EYE]
    lx, ly = int(lp.x * w), int(lp.y * h)

    # Right inner corner
    rp = face_landmarks.landmark[RIGHT_INNER_EYE]
    rx, ry = int(rp.x * w), int(rp.y * h)

    # Box corners
    top_left     = (lx,      ly - 10)
    bottom_right = (lx + 20, ly + 10)
    top_right    = (rx,      ry - 10)
    bottom_left  = (rx - 20, ry + 10)

    # Draw
    for pt in (top_left, bottom_right, top_right, bottom_left):
        cv2.circle(frame, pt, 2, (0, 255, 0), -5)

    cv2.rectangle(frame, top_left,  bottom_right, (255,   0,   0), 2)
    cv2.rectangle(frame, top_right, bottom_left,  (255,   0,   0), 2)

    return top_left, bottom_right, top_right, bottom_left


# ═════════════════════════════════════════════
#  ROI 3 — FOREHEAD
# ═════════════════════════════════════════════

FOREHEAD_POINTS = [67, 297, 105, 334]


def get_forehead_coordinates(frame: np.ndarray, face_landmarks, flag: bool):
    """
    Draws a bounding rectangle for the forehead ROI on `frame`.

    Returns
    -------
    (left_margin, top_margin, right_margin, adjusted_bottom)
    """
    h, w, _ = frame.shape
    coords = []

    for idx in FOREHEAD_POINTS:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        coords.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(frame, str(idx), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if len(coords) != 4:
        flag = True

    left_margin   = max(coords[0][0], coords[2][0])
    right_margin  = min(coords[1][0], coords[3][0])
    top_margin    = max(coords[0][1], coords[1][1])
    bottom_margin = min(coords[2][1], coords[3][1])

    # Clamp to image bounds
    left_margin   = max(0, left_margin)
    top_margin    = max(0, top_margin)
    right_margin  = min(w, right_margin)
    bottom_margin = min(h, bottom_margin)

    if left_margin >= right_margin or top_margin >= bottom_margin:
        flag = True

    adjusted_bottom = bottom_margin - 2   # exclude eyebrow fringe

    cv2.rectangle(frame,
                  (left_margin, top_margin),
                  (right_margin, adjusted_bottom),
                  (255, 255, 255), 2)

    return left_margin, top_margin, right_margin, adjusted_bottom


# ═════════════════════════════════════════════
#  ROI 4 — NOSE TIP
# ═════════════════════════════════════════════

NOSE_LANDMARKS = [19]


def get_nose_coordinates(frame: np.ndarray, face_landmarks):
    """
    Draws a bounding rectangle around the nose-bridge/tip area on `frame`.

    Returns
    -------
    (left_margin, top_margin, right_margin, bottom_margin)
    """
    h, w, _ = frame.shape

    for idx in NOSE_LANDMARKS:
        nose_tip = face_landmarks.landmark[idx]
        nx = int(nose_tip.x * w)
        ny = int(nose_tip.y * h)

        left_margin   = nx - 10
        right_margin  = nx + 10
        top_margin    = ny - 20
        bottom_margin = ny - 3

        cv2.rectangle(frame,
                      (left_margin, top_margin),
                      (right_margin, bottom_margin),
                      (100, 255, 150), 2)
        cv2.circle(frame, (nx, ny), 3, (0, 0, 255), -1)

        return left_margin, top_margin, right_margin, bottom_margin


# ═════════════════════════════════════════════
#  ROI 5 & 6 — CHEEKS (POLYGON)
# ═════════════════════════════════════════════

LEFT_CHEEK  = [214, 216, 206, 120, 101,  50, 187]
RIGHT_CHEEK = [432, 436, 426, 349, 330, 280, 411]
ALL_CHEEKS  = LEFT_CHEEK + RIGHT_CHEEK


def get_cheeks_coordinates(frame: np.ndarray, face_landmarks,
                            points_left: list, points_right: list):
    """
    Draws cheek polygons on `frame` and populates the point lists.

    Returns
    -------
    (points_left, points_right)
        Each is a list of (x, y) tuples — the polygon vertices.
    """
    h, w, _ = frame.shape

    for idx in ALL_CHEEKS:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)

        if idx in LEFT_CHEEK:
            points_left.append((x, y))
        else:
            points_right.append((x, y))

        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(frame, str(idx), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    if len(points_left) >= 3:
        polygon_left = np.array(points_left, dtype=np.int32)
        cv2.polylines(frame, [polygon_left], isClosed=True,
                      color=(0, 255, 0), thickness=1)

    if len(points_right) >= 3:
        polygon_right = np.array(points_right, dtype=np.int32)
        cv2.polylines(frame, [polygon_right], isClosed=True,
                      color=(0, 255, 0), thickness=1)

    return points_left, points_right