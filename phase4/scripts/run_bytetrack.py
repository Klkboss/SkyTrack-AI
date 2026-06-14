import pandas as pd
import numpy as np
import sys

sys.path.append(r"C:\DroneProject\gitrepos\ByteTrack")

from yolox.tracker.byte_tracker import BYTETracker

INPUT = r"C:\DroneProject\phase4\outputs\detections.csv"
OUTPUT = r"C:\DroneProject\phase4\outputs\tracks.csv"

df = pd.read_csv(INPUT)

class Args:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    frame_rate = 30
    mot20 = False

args = Args()

tracker = BYTETracker(args)

tracks_output = []

frames = sorted(df["frame"].unique())

for frame in frames:

    detections = df[df["frame"] == frame]

    dets = []

    for _, row in detections.iterrows():

        dets.append([
            row["x1"],
            row["y1"],
            row["x2"],
            row["y2"],
            row["score"]
        ])

    dets = np.array(dets)

    online_targets = tracker.update(dets, (1080,1920), (1080,1920))

    for t in online_targets:

        tlbr = t.tlbr
        track_id = t.track_id

        tracks_output.append([
            frame,
            int(tlbr[0]),
            int(tlbr[1]),
            int(tlbr[2]),
            int(tlbr[3]),
            track_id
        ])

out_df = pd.DataFrame(
    tracks_output,
    columns=["frame","x1","y1","x2","y2","track_id"]
)

out_df.to_csv(OUTPUT,index=False)

print("Real ByteTrack tracking completed")