# homography_helper.py
import cv2
import numpy as np
import json
import os

VIDEO_PATH = r"C:\DroneProject\phase4\inputs\test_video.mp4"
OUT_H = r"C:\DroneProject\phase4\outputs\homography.npy"
OUT_JSON = r"C:\DroneProject\phase4\outputs\homography_points.json"

# Load first frame for picking points
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    raise SystemExit("Cannot read video: " + VIDEO_PATH)

img = frame.copy()
h, w = img.shape[:2]
pts = []

def mouse_cb(event, x, y, flags, param):
    global pts, img
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])
        cv2.circle(img, (x, y), 6, (0,255,0), -1)
        cv2.putText(img, f"{len(pts)}", (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Pick 4 image points (clockwise or any order)", img)

cv2.imshow("Pick 4 image points (clockwise or any order)", img)
cv2.setMouseCallback("Pick 4 image points (clockwise or any order)", mouse_cb)

print("Click exactly 4 points on the IMAGE that correspond to known real-world points.")
print("After picking 4 points, press ENTER (or any key) in the image window to continue.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(pts) != 4:
    raise SystemExit("You must select exactly 4 image points. Selected: " + str(len(pts)))

# Ask user for corresponding real-world coordinates (units: meters)
print("\nNow enter the corresponding REAL-WORLD coordinates for each point in same order.")
print("Format for each point: x,y  (example: 0.0,0.0  or  12.5,3.2)")
world_pts = []
for i in range(4):
    s = input(f"Real-world coords for point {i+1} (image pt {pts[i]}): ")
    try:
        x,y = [float(v.strip()) for v in s.split(",")]
    except:
        raise SystemExit("Invalid input. Use format x,y (numbers).")
    world_pts.append([x,y])

img_pts = np.array(pts, dtype=np.float32)
world_pts = np.array(world_pts, dtype=np.float32)

# Compute homography from image -> world (2D)
H = cv2.getPerspectiveTransform(img_pts, world_pts)  # shape (3,3)
np.save(OUT_H, H)
with open(OUT_JSON, "w") as f:
    json.dump({"image_points": pts, "world_points": world_pts.tolist()}, f, indent=2)

print(f"Saved homography matrix to {OUT_H}")
print(f"Saved point pairs to {OUT_JSON}")