# Extracting the min and max value of LEGEND

import os

file_path = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data\62_2026-07-13\02_Psychometric_Tests\62_HDRS_Thermal_30_40.mpg"

file_name = os.path.basename(file_path)
file_name = os.path.splitext(file_name)[0]

max = file_name[-2:] 
min = file_name[-5:-3] 
 
print("min value", min)
print("max value", max)
print(type(file_name))
print(file_name)


