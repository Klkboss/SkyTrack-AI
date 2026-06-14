import pandas as pd
import json

TRACKS = r"C:\DroneProject\phase4\outputs\tracks_filtered.csv"

tracks = pd.read_csv(TRACKS)

unique_vehicles = tracks["track_id"].nunique()
total_detections = len(tracks)

vehicle_frames = tracks.groupby("track_id").size()
avg_track_length = vehicle_frames.mean()

# traffic density estimation
if unique_vehicles < 5:
    density = "Low"
elif unique_vehicles < 15:
    density = "Medium"
else:
    density = "High"

report = {
    "unique_vehicles": int(unique_vehicles),
    "total_detections": int(total_detections),
    "average_track_length_frames": float(avg_track_length),
    "traffic_density": density
}

print("\nTRAFFIC REPORT")
print("----------------------")
for k,v in report.items():
    print(k,":",v)

with open("phase4/outputs/traffic_report.json","w") as f:
    json.dump(report,f,indent=4)