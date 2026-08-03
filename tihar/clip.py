import cv2
import numpy as np

cap = cv2.VideoCapture(r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg")
 
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

left_frame = frame[:, 640: 800]  

onlyleft_frame = left_frame[33:446,26:74]

print("Frame shape:", frame.shape)
print("DataType of frame:", type(frame))
print("DataType of left_frame",  left_frame.shape)
print("DataType of onlyleft_frame",  onlyleft_frame.shape)

cv2.circle(onlyleft_frame, (47, 412), 1, (255, 255, 255), -1)


cv2.imshow("Frame", frame)
cv2.imshow("Left Frame", left_frame)
cv2.imshow("Only Left Frame", onlyleft_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

