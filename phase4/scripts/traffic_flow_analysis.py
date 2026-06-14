import cv2
import pandas as pd

VIDEO_PATH = r"C:\DroneProject\phase4\inputs\test_video.mp4"
TRACKS_PATH = r"C:\DroneProject\phase4\outputs\tracks_filtered.csv"

tracks = pd.read_csv(TRACKS_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# counting line
line_y = height // 2

counted_ids = set()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    frame_tracks = tracks[tracks["frame"] == frame_id]

    for _,row in frame_tracks.iterrows():

        x1,y1,x2,y2 = int(row["x1"]),int(row["y1"]),int(row["x2"]),int(row["y2"])
        track_id = int(row["track_id"])

        cy = int((y1+y2)/2)

        if cy > line_y and track_id not in counted_ids:
            counted_ids.add(track_id)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.line(frame,(0,line_y),(width,line_y),(0,0,255),3)

    cv2.putText(frame,
                f"Vehicles Passed: {len(counted_ids)}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3)

    cv2.imshow("Traffic Flow",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

print("Vehicles passed:",len(counted_ids))