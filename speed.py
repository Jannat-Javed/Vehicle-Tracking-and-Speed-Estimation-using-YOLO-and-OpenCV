import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker  

# Load the YOLO model
model = YOLO('yolo12s.pt')

# Class names (limited to first 30 from COCO)
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
              'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase']

# Vehicle classes to detect
vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}

# Initialize tracker
tracker = Tracker()

# Load video
cap = cv2.VideoCapture('videos/highway.mp4')

# Speed detection lines
red_line_y = 198
blue_line_y = 268
offset = 6  # Margin

# Trackers
down = {}
up = {}
counter_down = []
counter_up = []

# Create output directory
os.makedirs('detected_frames', exist_ok=True)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (1020, 500))

    # Predict
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() else []
    df = pd.DataFrame(detections).astype(float)

    vehicle_list = []
    for _, row in df.iterrows():
        x1, y1, x2, y2, conf, class_id = row
        class_id = int(class_id)
        if class_id < len(class_list):
            class_name = class_list[class_id]
            if class_name in vehicle_classes:
                vehicle_list.append([int(x1), int(y1), int(x2), int(y2)])

    # Track vehicles
    tracked_objects = tracker.update(vehicle_list)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Going Down (Red to Blue)
        if red_line_y - offset < cy < red_line_y + offset:
            down[obj_id] = time.time()
        if obj_id in down and blue_line_y - offset < cy < blue_line_y + offset:
            elapsed = time.time() - down[obj_id]
            if obj_id not in counter_down:
                counter_down.append(obj_id)
                distance_m = 10
                speed_kmph = (distance_m / elapsed) * 3.6
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{int(speed_kmph)} Km/h', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Going Up (Blue to Red)
        if blue_line_y - offset < cy < blue_line_y + offset:
            up[obj_id] = time.time()
        if obj_id in up and red_line_y - offset < cy < red_line_y + offset:
            elapsed = time.time() - up[obj_id]
            if obj_id not in counter_up:
                counter_up.append(obj_id)
                distance_m = 10
                speed_kmph = (distance_m / elapsed) * 3.6
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{int(speed_kmph)} Km/h', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw detection lines
    cv2.line(frame, (50, red_line_y), (970, red_line_y), (0, 0, 255), 2)
    cv2.putText(frame, 'Red Line (Up)', (55, red_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.line(frame, (50, blue_line_y), (970, blue_line_y), (255, 0, 0), 2)
    cv2.putText(frame, 'Blue Line (Down)', (55, blue_line_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw counters (UI: Up = Top Right, Down = Bottom Right)
    cv2.rectangle(frame, (800, 10), (1010, 80), (255, 255, 255), -1)
    cv2.putText(frame, f'Upward: {len(counter_up)}', (810, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f'Downward: {len(counter_down)}', (810, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Save output
    cv2.imwrite(f'detected_frames/frame_{frame_count}.jpg', frame)
    out.write(frame)
    cv2.imshow("Vehicle Detection & Speed", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()



