import cv2
import pandas as pd
import numpy as np

VIDEO_PATH = r"C:\DroneProject\phase4\inputs\test_video.mp4"
TRACKS_PATH = r"C:\DroneProject\phase4\outputs\tracks.csv"
OUTPUT_VIDEO = r"C:\DroneProject\phase4\outputs\tracked_speed_video.mp4"

# Assumed road lane width (meters)
LANE_WIDTH_METERS = 3.5

tracks = pd.read_csv(TRACKS_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Estimate lane width in pixels from frame
ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

lane_pixels = 120  # fallback

if lines is not None:
    lengths = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        lengths.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
    lane_pixels = np.median(lengths)

meters_per_pixel = LANE_WIDTH_METERS / lane_pixels

print("Meters per pixel:",meters_per_pixel)

track_history = {}

frame_id = 0

cap.set(cv2.CAP_PROP_POS_FRAMES,0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_tracks = tracks[tracks["frame"]==frame_id]

    for _,row in frame_tracks.iterrows():

        x1,y1,x2,y2 = int(row["x1"]),int(row["y1"]),int(row["x2"]),int(row["y2"])
        track_id = int(row["track_id"])

        cx = (x1+x2)/2
        cy = (y1+y2)/2

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx,cy))

        speed = 0

        if len(track_history[track_id])>=2:

            x_prev,y_prev = track_history[track_id][-2]
            dist_pixels = np.sqrt((cx-x_prev)**2+(cy-y_prev)**2)

            dist_meters = dist_pixels * meters_per_pixel

            speed = (dist_meters * fps) * 3.6  # km/h

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        label = f"ID {track_id} | {speed:.1f} km/h"

        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    out.write(frame)

    frame_id +=1

cap.release()
out.release()

print("Speed video saved:",OUTPUT_VIDEO)