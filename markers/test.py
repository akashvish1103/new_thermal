import os
import cv2

# Your exact local file path (using 'r' to handle Windows backslashes)
file_path = r"D:\akashvProfile-TESTO-recorded-InCDAC-Lab\thermal-data\jayesh_QA203.vmt"

print(f"--- Checking File: {os.path.basename(file_path)} ---")

# 1. VALIDATE FILE SIZE (Always works, regardless of format)
try:
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")
    
    if size_mb == 0:
        print("WARNING: File is 0 MB. The recording failed or the file is corrupt.")
except FileNotFoundError:
    print(f"ERROR: File not found at path: {file_path}")

# 2. VALIDATE VIDEO METADATA (Depends on OpenCV compatibility)
print("\n--- Checking Video Properties ---")
cap = cv2.VideoCapture(file_path)

if cap.isOpened():
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps > 0:
        duration = frames / fps
        print(f"Total Frames: {frames}")
        print(f"FPS: {fps}")
        print(f"Duration: {duration:.2f} seconds")
    else:
        print("WARNING: OpenCV opened the file, but FPS returned as 0.")
        print("This usually means it's a locked proprietary container.")
else:
    print("WARNING: OpenCV could not read the .vmt file.")
    print("The radiometric thermal data is locked behind the manufacturer's encoding.")

cap.release()