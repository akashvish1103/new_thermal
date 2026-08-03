import cv2
import numpy as np

# cap = cv2.VideoCapture(r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\61_2026-07-13\01_Passive_Profiling\61_passive_thermal.mpg")
# cap = cv2.VideoCapture(r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\62_2026-07-13\02_Psychometric_Tests\62_HDRS_grayscaled_Thermal.mpg")
cap = cv2.VideoCapture(r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\62_2026-07-13\02_Psychometric_Tests\62_HDRS_Thermal_30_40.mpg")

print("Opened:", cap.isOpened())

ret, frame = cap.read()

print("ret:", ret)
print("frame:", frame)

if not ret:
    print("ERROR: Couldn't read first frame.")
    exit()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ret, frame = cap.read()
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# frame = cv2.imread(r"D:\akashvProfile-TESTO-recorded-InCDAC-Lab\thermal-data\cropped_and_saved\legend\frame_000000_legend.png")
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# print(legend_frame.shape)


person_frame = frame[:, :640]       # we want to exclude 640 that why we wrote :640 and not :639
right_frame = frame[:, 640: 800]   # Legend
# right_frame = cv2.imread(r"D:\akashvProfile-TESTO-recorded-InCDAC-Lab\thermal-data\cropped_and_saved\legend\frame_000000_legend.png")
# right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
onlyright_frame = right_frame[33:446,26:74]   # color pallete box in legend(calcualted excat pixel cordinates in MS Paint)

print("Frame shape:", right_frame.shape)

print("shape of right_frame",  right_frame.shape)
print("shape of onlyright_frame",  onlyright_frame.shape)
one_pixel_column = onlyright_frame[:, 24:25]   # Extracting a single pixel column from the color palette box
one_pixel_column = onlyright_frame[:, 24]   # Extracting a single pixel column from the color palette box, using 24th index as SINGLE pixel column
print("shape of one_pixel_column",  one_pixel_column.shape) 
print("number of dimensions:", one_pixel_column.ndim)

# print(one_pixel_column)
lst = one_pixel_column.tolist()  # Convert the single pixel column to a list
print("one_pixel_column as list:")
print(lst)
 


cv2.imshow("Only Right Frame", onlyright_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

