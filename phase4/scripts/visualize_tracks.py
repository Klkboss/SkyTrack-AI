import cv2
import pandas as pd

VIDEO_PATH = r"C:\DroneProject\phase4\inputs\test_video.mp4"
TRACKS_CSV = r"C:\DroneProject\phase4\outputs\tracks.csv"
OUTPUT_VIDEO = r"C:\DroneProject\phase4\outputs\tracked_video.mp4"

tracks = pd.read_csv(TRACKS_CSV)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_tracks = tracks[tracks["frame"] == frame_id]

    for _, row in frame_tracks.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        track_id = int(row["track_id"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)
    frame_id += 1

cap.release()
out.release()

print("Tracked video saved:", OUTPUT_VIDEO)