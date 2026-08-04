# This file contians all the Utilities that will be required to map pixel values to Temperature (in Celsius).
# contains 3 functions as of now, will more functionalities as needed....


import cv2

# Function 1
def get_one_pixel_column(path):
    """
    This function takes the path of a video file as input, reads the first frame of the video,
    converts it to grayscale, and extracts a single pixel column from the color palette box 
    in the legend. The extracted pixel values are returned as a list.
    """

    cap = cv2.VideoCapture(path)
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
    # one_pixel_column = onlyright_frame[:, 24:25]   # Extracting a single pixel column from the color palette box (getting 2d array, so using single index to get 1d array)
    one_pixel_column = onlyright_frame[:, 24]   # Extracting a single pixel column from the color palette box, using 24th index as SINGLE pixel column
    print("shape of one_pixel_column",  one_pixel_column.shape) 
    print("number of dimensions:", one_pixel_column.ndim)

    # print(one_pixel_column)
    lst = one_pixel_column.tolist()  # Convert the single pixel column to a list
    print("one_pixel_column as list:")
    print(lst)
 


    # cv2.imshow("Only Right Frame", onlyright_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return lst                                              # returning the list of pixel values of the single pixel column extracted from the color palette box in the legend


#Function 2
def get_first_last_element_from_OnePixelColumn_AND_min_max_from_filename(lst, path):
    """
    this function takes the list of pixel values and the path of the video file as input,extracts the min and max temperature values 
    from the filename, and returns them along with the first and last elements of the pixel column array.
    """

    import os

    array_first_element = lst[0]  # First element of the pixel column array
    array_last_element = lst[-1]  # Last element of the pixel column array

    file_name = os.path.basename(path)
    file_name = os.path.splitext(file_name)[0]

    max_value = int(file_name[-2:]) 
    min_value = int(file_name[-5:-3]) 
 
    print("min value", min_value)
    print("max value", max_value)
    print(type(file_name))
    print(file_name)

    return min_value, max_value, array_first_element, array_last_element         # returning the min and max values, and the first and last elements of the one_pixel_column array


# ==========================================================
# Function 3: Map pixel intensity to temperature 
# ==========================================================

def map_pixel_to_temperature(pixel_value,
                             min_value,
                             max_value,
                             array_first_element,
                             array_last_element):
    """
    Maps a grayscale pixel intensity to a temperature using
    linear interpolation.

    Calibration Points
    ------------------
    P1 = (array_last_element,  min_value)
    P2 = (array_first_element, max_value)
    """

    if array_first_element == array_last_element:
        raise ValueError("Calibration pixel(i.e min and max pixles of legend) values cannot be equal.")

    temperature = (
        ((max_value - min_value) /
         (array_first_element - array_last_element))
        * (pixel_value - array_last_element)
        + min_value
    )

    return temperature


# Function-4, Driver Function
