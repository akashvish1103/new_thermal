import cv2
import numpy as np

cap = cv2.VideoCapture(r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg")
 
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# legend_frame = cv2.imread(r"D:\akashvProfile-TESTO-recorded-InCDAC-Lab\thermal-data\cropped_and_saved\legend\frame_000000_legend.png")
# legend_frame = cv2.cvtColor(legend_frame, cv2.COLOR_BGR2GRAY)
# print(legend_frame.shape)


person_frame = frame[:, :640]       # we want to exclude 640 that why we wrote :640 and not :639
right_frame = frame[:, 640: 800]   # Legend
onlyright_frame = right_frame[33:446,26:74]      # color pallete box in legend  (calcualted excat pixel cordinates in MS)

print("Frame shape:", frame.shape)

print("shape of right_frame",  right_frame.shape)
# print("shape of onlyright_frame",  onlyright_frame.shape)

cv2.circle(onlyright_frame, (47, 412), 1, (255, 255, 255), -1)
 

cv2.imshow("Frame", frame)
# cv2.imshow("Legend Frame", legend_frame)    
cv2.imshow("Person Frame", person_frame)
cv2.imshow("Only Right Frame", onlyright_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

