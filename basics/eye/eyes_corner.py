import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


           # how much hottest pixels to keep from eye ROI in the analysis (10% in this case)
# -----------------------------
# MediaPipe Face Mesh Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


left_mean_values = []
right_mean_values = []

# Inner eye corner landmarks
LEFT_INNER_EYE = 133                                      # mediapipe landmark for left eye inner corner
RIGHT_INNER_EYE = 362                                     # mediapipe landmark for right eye inner corner
PERCENTAGE_PIXEL_TO_KEEP = 0.80

# -----------------------------
# Input Video Path
# -----------------------------
# video_path = r"C:\Users\Akash Vishwakarma\Downloads\krishna_grey_manual1.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\pratham_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub2_rahul\output_rahul_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub3_shivam\output_shivam_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\aditi_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\priyank_grey_manual.wmv"
video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\grey_manual_mp4\jayesh_grey_manual.mp4"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sneha_grey_manual.wmv"
# video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\prsg_akash_grey_manual_mp4_output_trimmed_for_sync_new.mp4"
# video_path = r"D:\Lie Detection Data HTI\Yetnak\2026-02-25 14-29-38.mp4"
# video_path = r"D:\Lie Detection Data HTI\Akash\akash_manual_grey.mpg"
# video_path = video_path = r"D:\Lie Detection Data HTI\Lie_detection_ex2\Thermal_lie_detection_ex2\grey_manual\sub6_pooja\output_pooja_grey_manual.mp4"

cap = cv2.VideoCapture(video_path)

# -----------------------------
# Video Writer (optional)
# -----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# -----------------------------
# Process Video
# -----------------------------
while True:
    ret, frame = cap.read()
    

    if not ret:
        break
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                # converted the 3 channel bgr grame to a single channel grey frame.

    h, w, _ = frame.shape

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Mesh Detection
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # LEFT INNER EYE   --->   lx is the x-coordinate of inner corner for left eye and ly is the y coordinate of inner corner of left eye
            left_point = face_landmarks.landmark[LEFT_INNER_EYE]                     
            lx = int(left_point.x * w)
            ly = int(left_point.y * h)

            # RIGHT INNER EYE  --->  rx is the x-coordinate of inner corner for right eye and ry is the y coordinate of inner corner of right eye
            right_point = face_landmarks.landmark[RIGHT_INNER_EYE]
            rx = int(right_point.x * w)
            ry = int(right_point.y * h)
 
            
            
            # displaying the top left and bottom right corner of the box around inner eye corners LEFT EYE
            cv2.circle(frame, (lx, ly-10), 2, (0, 255, 0), -5)    #dot above left inner eye corner
            cv2.circle(frame, (lx+20, ly+10), 2, (0, 255, 0), -5)    #dot right to left inner eye corner

            # displaying the top left and bottom right corner of the box around inner eye corners RIGHT EYE
            cv2.circle(frame, (rx, ry-10), 2, (0, 255, 0), -5)    #dot above left inner eye corner
            cv2.circle(frame, (rx-20, ry+10), 2, (0, 255, 0), -5)    #dot right to left inner eye corner


            cv2.rectangle(frame, (lx, ly-10), (lx+20, ly+10), (255, 0, 0), 2)    #rectangle around left inner eye corner
            cv2.rectangle(frame, (rx, ry-10), (rx-20, ry+10), (255, 0, 0), 2)    #rectangle around right inner eye corner

            # Draw dots on the inner eye corners for visualization
            # cv2.circle(frame, (lx, ly), 2, (255, 255, 255), -3)  
            # cv2.circle(frame, (rx, ry), 2, (255, 255, 255), -3)

            # Extracting the ROI box around inner eye corners (20x20 box) 
            box_left = grey[ly-10:ly+10, lx:lx+20]
            box_right = grey[ry-10:ry+10, rx-20:rx]

            # Converting 2D box to 1D array for easier analysis
            left_flatten = box_left.flatten()
            right_flatten = box_right.flatten()

            # print(len(left_flatten), len(right_flatten))

            # Sorting pixel values in descending order to get hottest pixels at the beginning
            left_sorted_pixels = np.sort(left_flatten)[::-1]
            right_sorted_pixels = np.sort(right_flatten)[::-1]

            # Getting hottest 10% pixels
            left_sorted_pixels = left_sorted_pixels[0:int((len(left_sorted_pixels))*PERCENTAGE_PIXEL_TO_KEEP)]
            right_sorted_pixels = right_sorted_pixels[0:int((len(right_sorted_pixels))*PERCENTAGE_PIXEL_TO_KEEP)]
 
            # mean of hottest pixels for left and right eye
            left_mean = left_sorted_pixels.mean() 
            right_mean = right_sorted_pixels.mean()

            left_mean_values.append(left_mean)
            right_mean_values.append(right_mean)

            filt_frame = grey.copy()
            filt_frame = np.zeros_like(grey)
            cv2.rectangle(filt_frame, (lx, ly-10), (lx+20, ly+10), (255, 0, 0), 1)    #rectangle around left inner eye corner
            cv2.rectangle(filt_frame, (rx, ry-10), (rx-20, ry+10), (255, 0, 0), 1)    #rectangle around right inner eye corner
            # Extracting the ROI box around inner eye corners (20x20 box) 
            # box_left_filt = filt_frame[ly-10:ly+10, lx:lx+20]
            # box_right_filt = filt_frame[ry-10:ry+10, rx-20:rx]
            # -----------------------------
            # LEFT EYE
            # -----------------------------
            percentile_value = 100 - (PERCENTAGE_PIXEL_TO_KEEP * 100)

            # threshold corresponding to hottest 10%
            left_threshold = np.percentile(box_left, percentile_value)

            # mask for hottest pixels
            left_hot_mask = box_left >= left_threshold

            # put ONLY hottest pixels into filt_frame
            filt_frame[ly-10:ly+10, lx:lx+20][left_hot_mask] = \
                box_left[left_hot_mask]


            # -----------------------------
            # RIGHT EYE
            # -----------------------------

            right_threshold = np.percentile(box_right, percentile_value)

            right_hot_mask = box_right >= right_threshold

            filt_frame[ry-10:ry+10, rx-20:rx][right_hot_mask] = \
                box_right[right_hot_mask]



            # print(box_left.shape, box_right.shape)
            # # print(box_left.mean(), box_right.mean())
            print(left_sorted_pixels.mean(), right_sorted_pixels.mean())

    # Show frame
    cv2.imshow("Video Landmark Detection", frame)
    colored = cv2.applyColorMap(filt_frame, cv2.COLORMAP_JET)

    cv2.imshow("Filtered Frame", colored)
    # cv2.imshow("Filtered Frame", filt_frame)


    # ESC key to exitc
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# Release Resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(20,3))

plt.plot(left_mean_values, label="Left Eye")
plt.plot(right_mean_values, label="Right Eye")

plt.xlabel("Frame Number")
plt.ylabel("Mean Intensity")

plt.title("Peri-Orbital Thermal Signal")

plt.legend()

plt.show()