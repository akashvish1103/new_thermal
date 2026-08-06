import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh 

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )


image_path = r"C:\Users\Akash Vishwakarma\Pictures\Screenshots\Screenshot 2025-06-30 125137.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(image)
print(type(results))

h,w, _ = image.shape
for face_landmarks in results.multi_face_landmarks:
    print("face_landmarks:", face_landmarks)
    landmark = face_landmarks.landmark[362]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    print(x, y)

    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


cv2.imshow("Face Mesh", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
