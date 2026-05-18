import cv2
import numpy as np

# -----------------------------
# Read grayscale image
# -----------------------------
img_path = r"C:\Users\Akash Vishwakarma\Pictures\Screenshots\demo_grey_themal_frame.png"

img = cv2.imread(img_path)
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -----------------------------
# Define polygon
# -----------------------------
polygon = np.array([
    [100, 100],
    [300, 120],
    [350, 300],
    [200, 400],
    [80, 250]
], dtype=np.int32)

# -----------------------------
# Create empty mask
# -----------------------------
mask = np.zeros(image.shape, dtype=np.uint8)

# Fill polygon region with white
cv2.fillPoly(mask, [polygon], 255)

# -----------------------------
# Keep only polygon pixels
# -----------------------------
result = cv2.bitwise_and(image, mask)

# -----------------------------
# Display
# -----------------------------
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Polygon Extracted", result)

cv2.waitKey(0)
cv2.destroyAllWindows()