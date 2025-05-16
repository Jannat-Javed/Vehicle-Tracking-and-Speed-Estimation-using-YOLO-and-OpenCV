## Vehicle Detection, Counting, and Speed Estimation with YOLO and OpenCV
This repository contains two powerful Python-based computer vision pipelines:

**Vehicle Detection & Line-Crossing Counting**

**Vehicle Speed Estimation using Line-Based Timing**

Built using YOLO (You Only Look Once) models via the Ultralytics library and OpenCV, this system is capable of detecting and tracking vehicles in video streams, counting them as they cross a virtual line, and estimating their speed based on movement between defined lines.

### Features
**Vehicle Detection & Counting**
Uses YOLOv12 for real-time object detection and tracking.
Draws a horizontal red line across the frame.
Counts vehicles when they cross the line.
Displays per-class vehicle counts on the screen.

**Vehicle Speed Estimation**
Two horizontal lines (red and blue) define the detection zone.
Calculates vehicle speed in km/h based on the time taken to cross the distance between the lines.
Distinguishes upward and downward traffic.
Displays individual vehicle speed on bounding boxes.
Saves every processed frame to detected_frames/.
Outputs a processed video file output.avi.

### Models
You need YOLO model weights:

yolo12l.pt for object tracking and counting.

yolo12s.pt for detection + speed estimation.
