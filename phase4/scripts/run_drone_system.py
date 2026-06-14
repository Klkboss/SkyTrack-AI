# run_drone_system_fixed_v2.py
# Drone Traffic Analysis Pipeline (optimized)
# YOLOv8 Detection + ByteTrack Tracking + Semantic Segmentation + Analytics

import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
from math import hypot
from collections import defaultdict
from ultralytics import YOLO
import random

# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\DroneProject\phase4\inputs\test_video.mp4"
OUT_DIR = r"C:\DroneProject\phase4\outputs"

YOLO_WEIGHTS = r"C:\DroneProject\runs\drone_training\yolov8n_Trained2\weights\best.pt"
BYTETRACK_PATH = r"C:\DroneProject\gitrepos\ByteTrack"

# Optional segmentation model (single-channel road mask)
SEG_MODEL_PATH = r"C:\DroneProject\road_segmentation_model.pth"

# Optional homography (world mapping)
HOMOGRAPHY_NPY = os.path.join(OUT_DIR, "homography.npy")  # optional

OUT_VIDEO = os.path.join(OUT_DIR, "run_master_tracked.avi")
OUT_JSON = os.path.join(OUT_DIR, "run_master_report.json")
OUT_TRACKS = os.path.join(OUT_DIR, "tracks_master.csv")
HEATMAP_IMG = os.path.join(OUT_DIR, "traffic_heatmap.png")

# Parameters (tuneable)
CONF = 0.4           # detection confidence threshold
IOU = 0.5            # detection IOU when predicting
IMG_SZ = 1280
COUNT_LINE_POS = 0.5
SEG_EVERY_N_FRAMES = 5
SEG_THRESHOLD = 0.3
LANE_WIDTH_M = 3.5    # fallback lane width (meters)
HEATMAP_BLUR = (51, 51)
TRAJECTORY_LEN = 40   # how many points back to draw
MIN_BOX_AREA = 800    # filter small boxes (tune: 500-1500)
MIN_BOX_CONF = 0.4    # extra safety filter on box conf
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- helpers ----------------
def clamp_int(v, lo, hi):
    return int(min(max(int(round(v)), lo), hi))

def safe_mean(arr):
    return float(np.mean(arr)) if len(arr) else 0.0

# ---------------- weather detection ----------------
def detect_weather_simple(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 60:
        return "Night"
    if brightness < 100:
        return "Dark/Rainy"
    return "Clear"

# ---------------- load YOLO ----------------
print("Loading YOLO model...", YOLO_WEIGHTS)
yolo = YOLO(YOLO_WEIGHTS)

# ---------------- load ByteTrack ----------------
sys.path.append(BYTETRACK_PATH)
try:
    from yolox.tracker.byte_tracker import BYTETracker
except Exception as e:
    print("Error importing BYTETracker - check BYTETRACK_PATH:", BYTETRACK_PATH)
    raise

class _BTArgs:
    # tighter thresholds for more stable tracking
    track_thresh = 0.6
    track_buffer = 40
    match_thresh = 0.85
    frame_rate = 30
    mot20 = False

tracker = BYTETracker(_BTArgs())

# ---------------- segmentation (optional) ----------------
use_seg = os.path.exists(SEG_MODEL_PATH)
seg_model = None
if use_seg:
    try:
        import segmentation_models_pytorch as smp
        print("Loading segmentation model on device", DEVICE)
        seg_model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
        seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
        seg_model.to(DEVICE).eval()
        print("Segmentation model loaded.")
    except Exception as e:
        print("Failed to load segmentation model - disabling segmentation. Error:", e)
        use_seg = False
        seg_model = None

# ---------------- open video ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit("Cannot open video: " + VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(f"Video: {width}x{height} @ {fps} FPS, approx {frame_count} frames")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

count_line_y = int(height * COUNT_LINE_POS)

# ---------------- meters-per-pixel fallback (estimate from first frame) ----------------
ret0, first_frame = cap.read()
if not ret0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_frame = None

lane_pixels = 120.0
if first_frame is not None:
    try:
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            lengths = [np.hypot(x2-x1, y2-y1) for [[x1,y1,x2,y2]] in lines]
            if lengths:
                lane_pixels = float(np.median(lengths))
    except Exception:
        pass

meters_per_pixel = LANE_WIDTH_M / max(lane_pixels, 1.0)
print("Estimated meters_per_pixel (fallback):", meters_per_pixel)

# ---------------- optional homography ----------------
use_h = os.path.exists(HOMOGRAPHY_NPY)
H = None
if use_h:
    try:
        H = np.load(HOMOGRAPHY_NPY)
        print("Loaded homography:", HOMOGRAPHY_NPY)
    except Exception as e:
        print("Could not load homography - continuing without it:", e)
        H = None
        use_h = False

# reset to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------- data structures ----------------
track_hist = defaultdict(list)   # tid -> list of (frame, cx, cy, wx, wy)
counted_ids = set()
heatmap = np.zeros((height, width), dtype=np.float32)
tracks_csv = []
frame_idx = 0
last_mask = None
weather = "Unknown"

# additional structures for improvements
prev_positions = {}             # tid -> previous cy (pixel)
speed_history = defaultdict(list)  # tid -> recent speeds for smoothing
track_colors = {}               # tid -> color for drawing

def get_track_color(tid):
    if tid not in track_colors:
        random.seed(int(tid))
        track_colors[tid] = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
    return track_colors[tid]

# segmentation helper
def get_road_mask(frame):
    if not use_seg or seg_model is None:
        return None
    img = cv2.resize(frame, (512,512)).astype("float32") / 255.0
    t = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        m = seg_model(t)
        m = torch.sigmoid(m)[0,0].cpu().numpy()
    mask = cv2.resize(m, (width, height))
    return mask

print("Starting main loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # set weather based on first frame only
    if frame_idx == 0:
        weather = detect_weather_simple(frame)

    # update segmentation mask occasionally
    if use_seg and (frame_idx % SEG_EVERY_N_FRAMES == 0):
        try:
            last_mask = get_road_mask(frame)
        except Exception as e:
            print("Segmentation error:", e)
            last_mask = None
    mask = last_mask

    # ---------- YOLO detection ----------
    try:
        results = yolo.predict(source=frame, imgsz=IMG_SZ, conf=CONF, iou=IOU, verbose=False)
    except Exception as e:
        print("YOLO inference error on frame", frame_idx, " — skipping frame. Error:", e)
        frame_idx += 1
        continue

    # gather and filter detections (x1,y1,x2,y2,conf)
    dets = []
    if len(results) > 0:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), sc in zip(xyxy, confs):
                area = (x2 - x1) * (y2 - y1)
                # area filter (remove tiny boxes / noise)
                if area < MIN_BOX_AREA:
                    continue
                # confidence safety
                if sc < MIN_BOX_CONF:
                    continue
                dets.append([float(x1), float(y1), float(x2), float(y2), float(sc)])

    dets_arr = np.array(dets) if dets else np.zeros((0, 5), dtype=float)
    print(f"Frame {frame_idx} detections: {len(dets_arr)}")

    # ---------- ByteTrack update ----------
    try:
        # NOTE: ByteTrack expects detections as [[x1,y1,x2,y2,score], ...] and image size as (H,W)
        online_targets = tracker.update(dets_arr, (height, width), (height, width))
    except Exception as e:
        print("Tracker update error:", e)
        online_targets = []

    # ---------- process targets ----------
    for t in online_targets:
        # get bbox robustly
        if hasattr(t, "tlbr"):
            tlbr = t.tlbr
            x1, y1, x2, y2 = [int(round(v)) for v in tlbr]
        elif hasattr(t, "tlwh"):
            tlwh = t.tlwh
            x1 = int(round(tlwh[0])); y1 = int(round(tlwh[1]))
            x2 = int(round(x1 + tlwh[2])); y2 = int(round(y1 + tlwh[3]))
        else:
            continue

        tid = int(getattr(t, "track_id", getattr(t, "track_id_", 0)))
        score = float(getattr(t, "score", 0.0))

        # skip low-quality tracks
        if score < _BTArgs.track_thresh:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # road filter (if segmentation mask available)
        if mask is not None:
            ix = clamp_int(cx, 0, width-1)
            iy = clamp_int(cy, 0, height-1)
            try:
                if mask[iy, ix] < SEG_THRESHOLD:
                    continue
            except Exception:
                pass

        # pixel -> world conversion (use homography if available)
        if H is not None:
            try:
                pts = np.array([[[cx, cy]]], dtype=np.float32)
                wxwy = cv2.perspectiveTransform(pts, H)[0,0]
                wx, wy = float(wxwy[0]), float(wxwy[1])
            except Exception:
                wx = float(cx * meters_per_pixel)
                wy = float(cy * meters_per_pixel)
        else:
            wx = float(cx * meters_per_pixel)
            wy = float(cy * meters_per_pixel)

        # append track history
        track_hist[tid].append((frame_idx, cx, cy, wx, wy))
        tracks_csv.append({"frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "track_id": tid, "score": score})

        # compute smoothed speed (km/h)
        speed_kmh = 0.0
        if len(track_hist[tid]) >= 2:
            f0, cx0, cy0, wx0, wy0 = track_hist[tid][-2]
            dt = (frame_idx - f0) / fps if fps > 0 else 0.0
            if dt > 0:
                dist_m = hypot(wx - wx0, wy - wy0)
                inst_speed = (dist_m / dt) * 3.6

                # smooth speeds with short history
                speed_history[tid].append(inst_speed)
                if len(speed_history[tid]) > 5:
                    speed_history[tid].pop(0)
                speed_kmh = sum(speed_history[tid]) / len(speed_history[tid])

        # CROSSING logic: count on crossing the line from top -> bottom
        prev_y = prev_positions.get(tid, None)
        if prev_y is not None:
            if prev_y < count_line_y and cy >= count_line_y:
                counted_ids.add(tid)
        prev_positions[tid] = cy

        # draw bbox and label
        color = get_track_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid}"
        if speed_kmh > 0.5:
            label += f" {speed_kmh:.0f} km/h"
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # trajectory drawing (last TRAJECTORY_LEN points)
        pts = track_hist[tid][-TRAJECTORY_LEN:]
        for i in range(1, len(pts)):
            _, x_prev, y_prev, _, _ = pts[i-1]
            _, x_curr, y_curr, _, _ = pts[i]
            cv2.line(frame, (int(x_prev), int(y_prev)), (int(x_curr), int(y_curr)), color, 2)

        # accumulate heatmap (speed-weighted)
        hx = clamp_int(cx, 0, width-1)
        hy = clamp_int(cy, 0, height-1)
        heatmap[hy, hx] += speed_kmh

    # ---------- analytics (realtime) ----------
    unique_vehicles = len(track_hist.keys())
    passed_vehicles = len(counted_ids)

    # lane flow (simple center split)
    left_lane = 0
    right_lane = 0
    for tid, hist in track_hist.items():
        if not hist: continue
        _, cxn, cyn, _, _ = hist[-1]
        if cxn < width/2.0:
            left_lane += 1
        else:
            right_lane += 1

    # realtime speed list from last steps
    current_speeds = []
    for tid, hist in track_hist.items():
        if len(hist) < 2: continue
        f0, cx0, cy0, wx0, wy0 = hist[-2]
        f1, cx1, cy1, wx1, wy1 = hist[-1]
        dt = (f1 - f0) / fps if fps > 0 else 0.0
        if dt <= 0: continue
        dist_m = hypot(wx1 - wx0, wy1 - wy0)
        current_speeds.append((dist_m / dt) * 3.6)
    realtime_avg_speed = safe_mean(current_speeds)

    # traffic density label
    if unique_vehicles < 5:
        density = "Low"
    elif unique_vehicles < 15:
        density = "Medium"
    else:
        density = "High"

    # realtime congestion heuristic
    if density == "Low":
        congestion = "Low"
    elif density == "Medium":
        if realtime_avg_speed > 50:
            congestion = "Low"
        elif realtime_avg_speed > 30:
            congestion = "Moderate"
        else:
            congestion = "High"
    else:
        congestion = "High"

    # flow per hour (based on vehicles that passed count line and elapsed time)
    elapsed_seconds = frame_idx / fps if fps > 0 else 0.0
    if elapsed_seconds > 0:
        flow_per_hour = (len(counted_ids) / elapsed_seconds) * 3600.0
    else:
        flow_per_hour = 0.0

    # overlay everything (draw a background panel)
    overlay_lines = [
        f"Vehicles (unique): {unique_vehicles}",
        f"Total Detections: {len(tracks_csv)}",
        f"Passed line: {passed_vehicles}",
        f"Left lane: {left_lane}   Right lane: {right_lane}",
        f"Realtime Avg Speed: {realtime_avg_speed:.1f} km/h",
        f"Flow (veh/hr): {flow_per_hour:.1f}",
        f"Density: {density}",
        f"Congestion: {congestion}",
        f"Weather: {weather}"
    ]

    panel_w = 460
    panel_h = 28 + len(overlay_lines) * 24
    cv2.rectangle(frame, (6,6), (6 + panel_w, 6 + panel_h), (0,0,0), -1)
    for i, txt in enumerate(overlay_lines):
        cv2.putText(frame, txt, (12, 30 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # draw count line
    cv2.line(frame, (0, count_line_y), (width, count_line_y), (0,0,255), 2)

    # write frame to output video
    out.write(frame)

    # increment frame index exactly once per frame
    frame_idx += 1

# ---------------- end loop ----------------
cap.release()
out.release()

# ---------------- save CSV ----------------
df_tracks = pd.DataFrame(tracks_csv)
if not df_tracks.empty:
    df_tracks.to_csv(OUT_TRACKS, index=False)
    total_detections = len(df_tracks)
else:
    total_detections = 0

# ---------------- compute final metrics ----------------
unique_vehicles = len(track_hist.keys())

# per-track median speeds
median_speeds = []
for tid, hist in track_hist.items():
    if len(hist) < 2: continue
    speeds = []
    for i in range(1, len(hist)):
        f0, cx0, cy0, wx0, wy0 = hist[i-1]
        f1, cx1, cy1, wx1, wy1 = hist[i]
        dt = (f1 - f0) / fps if fps > 0 else 0.0
        if dt <= 0: continue
        speeds.append(hypot(wx1 - wx0, wy1 - wy0) / dt * 3.6)
    if speeds:
        median_speeds.append(float(np.median(speeds)))
avg_median_speed = float(np.mean(median_speeds)) if median_speeds else 0.0

# flow per hour final
duration_seconds = (frame_count / fps) if (fps > 0 and frame_count > 0) else (frame_idx / fps if fps>0 else 1.0)
flow_per_hour = (len(counted_ids) / duration_seconds) * 3600.0 if duration_seconds > 0 else 0.0

# final left/right tally
left_lane = 0; right_lane = 0
for tid, hist in track_hist.items():
    if not hist: continue
    _, cxn, cyn, _, _ = hist[-1]
    if cxn < width / 2.0:
        left_lane += 1
    else:
        right_lane += 1

# density (final)
vehicle_count = unique_vehicles
if vehicle_count < 5:
    density = "Low"
elif vehicle_count < 15:
    density = "Medium"
else:
    density = "High"

# congestion final
if density == "Low":
    congestion = "Low"
elif density == "Medium":
    if avg_median_speed > 50:
        congestion = "Low"
    elif avg_median_speed > 30:
        congestion = "Moderate"
    else:
        congestion = "High"
else:
    congestion = "High"

report = {
    "unique_vehicles": int(unique_vehicles),
    "total_detections": int(total_detections),
    "vehicles_passed_line": int(len(counted_ids)),
    "left_lane_vehicles": int(left_lane),
    "right_lane_vehicles": int(right_lane),
    "traffic_flow_vehicles_per_hour": float(round(flow_per_hour, 1)),
    "average_median_speed_kmh": float(round(avg_median_speed, 1)),
    "traffic_density": density,
    "traffic_congestion": congestion,
    "weather": weather
}

with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)

print("\nFINAL TRAFFIC REPORT")
for k, v in report.items():
    print(f"{k}: {v}")

# heatmap
if heatmap.sum() > 0:
    heat = cv2.GaussianBlur(heatmap, HEATMAP_BLUR, 0)
    norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(HEATMAP_IMG, cmap)
    print("Saved heatmap image:", HEATMAP_IMG)

print("Annotated video saved to:", OUT_VIDEO)
print("Tracks CSV saved to:", OUT_TRACKS)
print("JSON report saved to:", OUT_JSON)
print("Done.")