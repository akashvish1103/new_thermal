import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't access camera")
        break

    cv2.imshow("my-window", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Q pressed, quitting...")
        break

    if key == 27:  # ESC key
        break


# Release resources
cap.release()
cv2.destroyAllWindows()