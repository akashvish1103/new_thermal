# READ ME for uitlity inside SCRATCH folder
# ROI Extraction Utilities using MediaPipe FaceMesh

## Overview

This project contains a set of utility functions for extracting different **Regions of Interest (ROIs)** from a human face using **MediaPipe FaceMesh**. These utilities are primarily developed for thermal image/video processing, where different facial regions are required for physiological analysis such as breathing pattern, forehead temperature, cheek temperature, and eye temperature.

The project is divided into two parts:

* **Driver File** – Reads the video, performs face detection using MediaPipe, and calls the utility functions.
* **utilities.py** – Contains reusable functions for extracting different facial ROIs.

---

# Workflow

```
Video
   │
   ▼
Read Frame
   │
   ▼
Convert to Grayscale
   │
   ▼
Enhance Image
(get_transformed_image)
   │
   ▼
Convert to RGB
   │
   ▼
MediaPipe FaceMesh
   │
   ▼
face_landmarks
   │
   ├── Eyes ROI
   ├── Forehead ROI
   ├── Nose Tip ROI
   ├── Breathing ROI
   └── Cheek ROIs
   │
   ▼
Modified Frame
   │
   ▼
Display using cv2.imshow()
```

---

# Requirements

* Python 3.x
* OpenCV
* NumPy
* MediaPipe

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

---

# Driver File

The driver file performs the following tasks:

1. Reads the thermal video.
2. Converts each frame to grayscale.
3. Enhances the grayscale image.
4. Runs MediaPipe FaceMesh.
5. Calls utility functions to extract different ROIs.
6. Displays the processed frame.

---

# Utility Functions

## 1. `get_transformed_image(grey_frame)`

### Description

Enhances the grayscale image using:

* Contrast Stretching
* CLAHE
* Gamma Correction
* Sharpening

### Input

```python
grey_frame
```

**Type**

```
numpy.ndarray
```

### Returns

```python
enhanced_grey_frame
```

---

## 2. `get_eyes_coordinates(grey_frame, face_landmarks)`

### Description

Computes rectangular ROIs around both inner eye corners.

### Inputs

```python
grey_frame
face_landmarks
```

### Returns

```python
top_left_coords
bottom_right_coords
top_right_coords
bottom_left_coords
grey_frame
```

Example:

```
Left Eye ROI

Top Left        : (220,145)
Bottom Right    : (240,165)

Right Eye ROI

Top Right       : (395,145)
Bottom Left     : (375,165)
```

---

## 3. `get_nose_tip_coordinates(grey_frame, face_landmarks)`

### Description

Creates a rectangular ROI around the nose tip.

### Inputs

```python
grey_frame
face_landmarks
```

### Returns

```python
top_left_coords
bottom_right_coords
grey_frame
```

---

## 4. `get_breathing_roi_cords(grey_frame, face_landmarks)`

### Description

Creates a rectangular ROI around the nostril region. This ROI can be used for breathing pattern extraction.

### Inputs

```python
grey_frame
face_landmarks
```

### Returns

```python
top_left_coords
bottom_right_coords
grey_frame
```

---

## 5. `get_cheeks_coordinates(grey_frame, face_landmarks, points_left, points_right)`

### Description

Extracts landmark coordinates of both cheeks and creates cheek polygons.

### Inputs

```python
grey_frame
face_landmarks
points_left
points_right
```

### Returns

```python
points_left
points_right
grey_frame
```

Example:

```
Left Cheek

[(210,205),
 (220,212),
 (235,220),
 ...]

Right Cheek

[(430,205),
 (420,212),
 (405,220),
 ...]
```

---

## 6. `get_forhead_poly_coords(grey_frame, face_landmarks)`

### Description

Creates a polygon over the forehead using predefined MediaPipe landmarks and computes the average pixel intensity inside the polygon.

### Inputs

```python
grey_frame
face_landmarks
```

### Returns

```python
polygon_points
mean_pixel
grey_frame
```

Example:

```
polygon_points

[
 [245,98],
 [260,95],
 [280,92],
 [315,94],
 [340,99],
 [335,115],
 [310,120],
 [275,118],
 [250,112]
]

mean_pixel

132.54
```

---

# MediaPipe FaceMesh Hierarchy

Each utility function receives the following object:

```python
for face_landmarks in results.multi_face_landmarks:
```

Hierarchy:

```
results
│
└── multi_face_landmarks
      │
      └── face_landmarks
             │
             ├── landmark[0]
             ├── landmark[1]
             ├── ...
             └── landmark[477]
```

Each landmark contains:

```
landmark.x
landmark.y
landmark.z
```

The coordinates are normalized and converted to pixel coordinates as:

```python
x = int(landmark.x * image_width)
y = int(landmark.y * image_height)
```

---

# Notes

* All utility functions **modify the input image** by drawing circles, rectangles, or polygons.
* None of the utility functions display the image.
* The modified image is returned so that the driver file can display all ROIs together using `cv2.imshow()`.
* The extracted coordinates can be used later for ROI cropping, temperature analysis, or physiological signal extraction.

---

# Author

Developed for thermal facial ROI extraction using **MediaPipe FaceMesh** and **OpenCV**.
