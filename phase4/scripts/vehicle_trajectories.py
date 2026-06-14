import cv2
import pandas as pd

VIDEO = r"C:\DroneProject\phase4\inputs\test_video.mp4"
TRACKS = r"C:\DroneProject\phase4\outputs\tracks_filtered.csv"

tracks = pd.read_csv(TRACKS)

cap = cv2.VideoCapture(VIDEO)

trajectories = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    frame_tracks = tracks[tracks["frame"] == frame_id]

    for _,row in frame_tracks.iterrows():

        x1,y1,x2,y2 = int(row["x1"]),int(row["y1"]),int(row["x2"]),int(row["y2"])
        track_id = int(row["track_id"])

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        if track_id not in trajectories:
            trajectories[track_id] = []

        trajectories[track_id].append((cx,cy))

        for i in range(1,len(trajectories[track_id])):
            cv2.line(frame,
                     trajectories[track_id][i-1],
                     trajectories[track_id][i],
                     (255,0,0),
                     2)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("Trajectories",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()