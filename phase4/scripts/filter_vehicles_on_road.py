import cv2
import pandas as pd
import torch
import numpy as np
import segmentation_models_pytorch as smp

VIDEO_PATH = r"C:\DroneProject\phase4\inputs\test_video.mp4"
TRACKS_PATH = r"C:\DroneProject\phase4\outputs\tracks.csv"
MODEL_PATH = r"C:\DroneProject\road_segmentation_model.pth"

OUTPUT_PATH = r"C:\DroneProject\phase4\outputs\tracks_filtered.csv"

tracks = pd.read_csv(TRACKS_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)

filtered_rows = []

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_tracks = tracks[tracks["frame"] == frame_id]

    img = cv2.resize(frame,(256,256))
    img = img / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        mask = model(img)
        mask = torch.sigmoid(mask)
        mask = mask.cpu().numpy()[0,0]

    mask = cv2.resize(mask,(frame.shape[1],frame.shape[0]))

    for _,row in frame_tracks.iterrows():

        x1,y1,x2,y2 = int(row["x1"]),int(row["y1"]),int(row["x2"]),int(row["y2"])
        track_id = row["track_id"]

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        if mask[cy,cx] > 0.5:
            filtered_rows.append(row)

    frame_id +=1

cap.release()

filtered_df = pd.DataFrame(filtered_rows)

filtered_df.to_csv(OUTPUT_PATH,index=False)

print("Filtered tracks saved:",OUTPUT_PATH)