# This code is for understanding the basics of Mediapipe Landmarks, how the mp.solutions hierarchy works.

import mediapipe as mp
import  cv2
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True)

IMG_PTH = r"C:\Users\Akash Vishwakarma\Pictures\Screenshots\Screenshot 2025-06-30 123138.png"
img = cv2.imread(IMG_PTH)

print(img.shape)
print(type(img))

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = face_mesh.process(rgb)


print(type(results))
# print(len(results))

# print(results.multi_face_landmarks)
print(type(results.multi_face_landmarks))
print(len(results.multi_face_landmarks))

# print(results.multi_face_landmarks[0])
print(type(results.multi_face_landmarks[0]))

try:
    print(len(results.multi_face_landmarks[0]))
except Exception as e:
    print("Some ERROR :", e)


# print(results.multi_face_landmarks[0])

print(results.multi_face_landmarks[0].landmark[0])   # First face in results.multi_face_landmarks having a nromalisedList , we are tapping out 0th Index Landmark.
print(results.multi_face_landmarks[0].landmark[0].x)

