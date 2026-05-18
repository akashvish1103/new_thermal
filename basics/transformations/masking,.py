import cv2
import numpy as np


img_path = r"C:\Users\Akash Vishwakarma\Downloads\gian-gomez-rYB1r1MoOXc-unsplash.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img.shape)
# img = img.reshape(600,400)
img = cv2.resize(img, (400, 600))
print(img.shape)


points = np.array([
    [100, 100],
    [300, 120],
    [350, 300],
    [200, 400],
    [140, 250]
], dtype=np.int32)


cv2.polylines(img, [points], isClosed=True, color=(0,0,255), thickness=2)

mask = np.zeros(img.shape, dtype=np.uint8)
print("shape of the MASK:", mask.shape)
print(mask)
cv2.fillPoly(mask, [points], 255)
print("after the fillPoly:", mask)
result = cv2.bitwise_and(img, mask)

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Polygon Extracted", result)
cv2.waitKey(0)
cv2.destroyAllWindows()



