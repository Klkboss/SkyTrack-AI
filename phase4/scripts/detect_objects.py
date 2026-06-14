import cv2
import csv
from ultralytics import YOLO

VIDEO_PATH = r"C:\DroneProject\phase4\inputs\test_video.mp4"
MODEL_PATH = r"C:\DroneProject\runs\drone_training\yolov8n_Trained2\weights\best.pt"
OUTPUT_CSV = r"C:\DroneProject\phase4\outputs\detections.csv"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame","x1","y1","x2","y2","score","class"])

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls = int(box.cls[0])

                writer.writerow([
                    frame_id,
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    score,
                    cls
                ])

        frame_id += 1

cap.release()

print("Detection CSV saved")