# SkyTrack AI
## Drone-Based Vehicle Detection and Tracking System

SkyTrack AI is a drone-video analytics pipeline for detecting, tracking, and analyzing vehicles from aerial footage. The project combines modern object detection, multi-object tracking, and post-processing analytics to turn raw drone video into usable traffic intelligence.

## Project Summary

The system is designed to:

- detect vehicles in drone footage,
- track them across frames,
- filter detections using road segmentation,
- estimate speed,
- count traffic flow,
- analyze trajectories,
- detect congestion,
- generate speed heatmaps,
- and produce a final traffic report.

The project is organized in phases, with the later phase integrating the full end-to-end pipeline through a script such as `phase4/scripts/run_drone_system.py`.

---
## Team Collaboration

SkyTrack AI was developed as a collaborative team project focused on drone-based vehicle detection, tracking, and traffic analytics.

### Team Members

* **Kushagra Chauhan**
* **Utsav Majumdar**
* **Siddhi Vinayak Dash**

### Acknowledgement

This repository is a personal portfolio copy of the original team project. Full credit goes to all team members for their contributions throughout the design, development, testing, and documentation phases of the project.


## Core Pipeline

```text
Drone Video
   ↓
Vehicle Detection (YOLOv8)
   ↓
Multi-Object Tracking (ByteTrack)
   ↓
Road Segmentation Filter
   ↓
Speed Estimation
   ↓
Traffic Flow Counting
   ↓
Weather Detection
   ↓
Trajectory Analysis
   ↓
Congestion Detection
   ↓
Speed Heatmap
   ↓
Traffic Report
```

---

## Main Features

- **Vehicle detection** from drone footage using YOLO-based inference.
- **Multi-object tracking** with ByteTrack to maintain identities across frames.
- **Road-area filtering** to reduce false positives outside drivable regions.
- **Speed estimation** for tracked vehicles.
- **Line-crossing / traffic counting** for traffic flow analysis.
- **Trajectory analysis** for path visualization and motion understanding.
- **Congestion detection** based on density and movement behavior.
- **Heatmap generation** to highlight frequent vehicle movement zones.
- **Weather-aware analysis** for better interpretation of scene conditions.
- **Traffic summary report** generated from the processed video.

---

## Datasets

This project uses drone / aerial vehicle datasets for training and validation. Based on the project workflow, the main dataset references are:

### 1) UAVDT
A drone-based vehicle detection and tracking dataset commonly used for aerial traffic analysis. It includes vehicle instances captured from drone viewpoints and is well-suited for detection, tracking, and motion-based analytics.

### 2) VisDrone
A widely used benchmark for object detection and tracking in drone imagery. It contains diverse aerial scenes, challenging object scales, occlusions, and different traffic conditions.

### 3) SkyScapes / SkyScapes reference data
The project discussion also references SkyScapes-style aerial data for drone scene understanding. If your final training run used a specific SkyScapes split or derivative subset, keep that exact name here.

### Dataset Usage in the Project
Typical dataset workflow:

- download the dataset,
- extract images and annotations,
- convert annotations if required by the detector,
- split into train / validation / test sets,
- train the detector,
- evaluate detection and tracking quality.

### Annotation Formats
Depending on the dataset version, annotations may be in formats such as:

- YOLO text labels,
- COCO JSON,
- XML / custom aerial annotation format.

---

## Model Stack

### Detector
**YOLOv8** is used as the object detector for vehicle localization.

Why it fits this project:

- strong real-time performance,
- good accuracy on small objects,
- easy training and export support,
- suitable for drone video where vehicles are often tiny and densely packed.

### Tracker
**ByteTrack** is used for multi-object tracking.

Why it is used:

- keeps track IDs stable across frames,
- works well with dense traffic scenes,
- is effective even when detections are intermittent.

### Post-Processing Modules
The pipeline includes additional analytics modules for:

- road segmentation,
- speed estimation,
- traffic counting,
- congestion scoring,
- trajectory drawing,
- heatmap creation,
- and report generation.

---

## Project Structure

A typical structure for the project looks like this:

```text
SkyTrack-AI/
├── phase1/
├── phase2/
├── phase3/
├── phase4/
│   └── scripts/
│       └── run_drone_system.py
├── datasets/
├── models/
├── outputs/
├── reports/
├── requirements.txt
└── README.md
```

Your exact folder names may differ slightly, but the project is organized around a phased workflow and a final execution script.

---

## Installation

### 1) Create a Python environment

Using Conda:

```bash
conda create --prefix ./Drone python=3.10
conda activate ./Drone
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If the project uses ByteTrack or custom compiled dependencies, install them after the base requirements.

---

## Requirements

Common packages used in this project include:

- `python 3.10`
- `numpy`
- `opencv-python`
- `torch`
- `torchvision`
- `ultralytics`
- `pandas`
- `matplotlib`
- `scipy`
- `filterpy`
- `lap`
- `pyyaml`

If your repository already contains a `requirements.txt`, that file should remain the source of truth for exact versions.

---

## Training

### Detection Training
Train the detector with your prepared dataset and YAML config.

Example:

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

### What to Record From Training
For the README, keep the following training metrics from your final run:

- precision,
- recall,
- mAP@0.5,
- mAP@0.5:0.95,
- training loss,
- validation loss,
- inference FPS.

### Tracking Evaluation
For the tracking stage, include metrics such as:

- ID switches,
- MOTA,
- MOTP,
- IDF1,
- tracking FPS.

---

## Training Results

Replace the values below with the exact numbers from your final training logs.

| Metric | Value |
|---|---:|
| Precision | `<add value>` |
| Recall | `<add value>` |
| mAP@0.5 | `<add value>` |
| mAP@0.5:0.95 | `<add value>` |
| Tracking IDF1 | `<add value>` |
| Tracking FPS | `<add value>` |

### Sample Project Output
From the existing demo output discussed in the project:

- **Unique vehicles:** 14
- **Total detections:** 862

If you have the full JSON report, add the remaining fields here as well, especially the line-crossing and congestion values.

---

## Running the Project

The final pipeline can be launched from the main runtime script.

Example:

```bash
python phase4/scripts/run_drone_system.py
```

If your script accepts arguments, you can run it with custom input paths:

```bash
python phase4/scripts/run_drone_system.py --source path/to/input_video.mp4 --output path/to/output_folder
```

---

## Output Files

The pipeline may generate the following outputs:

- annotated detection video,
- tracking overlay video,
- JSON traffic statistics,
- CSV summary files,
- heatmaps,
- trajectory plots,
- congestion reports.

---

## How the System Works

### 1. Video Input
The user provides a drone video or aerial traffic video.

### 2. Detection
YOLOv8 detects vehicles frame by frame.

### 3. Tracking
ByteTrack assigns and maintains object IDs over time.

### 4. Filtering
Road segmentation removes objects outside the road region.

### 5. Speed Estimation
Vehicle movement between frames is used to estimate speed.

### 6. Traffic Counting
Vehicles crossing a reference line are counted for traffic flow analysis.

### 7. Analytics
The system computes:

- trajectory patterns,
- congestion levels,
- heatmap density,
- and a final traffic summary.

### 8. Report Generation
The final output is stored as a report for further analysis or presentation.

---

## Configuration

The most common things you may need to change are:

- dataset paths,
- model weights path,
- input video path,
- output folder path,
- confidence threshold,
- tracking threshold,
- line position for counting,
- speed calibration parameters,
- road mask / segmentation settings.

---

## Troubleshooting

### 1) Module import errors
Make sure the correct environment is activated and all packages from `requirements.txt` are installed.

### 2) ByteTrack / tracking issues
Check that the tracker dependencies are installed and that your detector confidence threshold is not too high.

### 3) Wrong counting results
Verify the line position, frame size, and coordinate calibration.

### 4) Speed estimation looks incorrect
Confirm the frame rate, pixel-to-meter conversion, and calibration settings.

### 5) Output video is lagging
Use a smaller input resolution or a lighter detector weight such as a nano/small model.

---

## Future Improvements

Possible next upgrades for SkyTrack AI:

- stronger detector backbones,
- better small-object handling,
- lane-wise vehicle analytics,
- real-time dashboard integration,
- camera calibration for more accurate speed,
- weather-specific model adaptation,
- per-class traffic statistics,
- deployment as a Flask or FastAPI application.

---

## Conclusion

SkyTrack AI brings together drone-based detection, tracking, and analytics into a single traffic intelligence pipeline. It can be used for vehicle counting, congestion monitoring, speed estimation, and aerial traffic analysis.

---

## Maintainer Notes

If you want this README to reflect your exact final experiment, update these items before publishing:

- dataset names and split sizes,
- final detection metrics,
- tracking metrics,
- model checkpoint name,
- exact run command used in the final phase,
- and any custom thresholds or calibration constants.
