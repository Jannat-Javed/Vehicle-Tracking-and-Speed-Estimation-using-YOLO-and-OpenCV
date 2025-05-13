import cv2
from ultralytics import YOLO
from collections import defaultdict


# Load the YOLO model
model = YOLO('yolo12l.pt')

class_list = model.names 

# Open the video file
cap = cv2.VideoCapture('videos/4.mp4')

line_y = 510  # Used for counting logic only

# Dictionary to store object counts by class
class_counts = defaultdict(int)

# Dictionary to keep track of object IDs that have crossed the line
crossed_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])

    # Ensure results are not empty
    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

         # Show the invisible (now visible) counting line
        # Draw a long, visible red counting line across the frame
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)


        # Loop through each detected object
        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cy = (y1 + y2) // 2

            class_name = class_list[class_idx]

            # Draw only the bounding box and class name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Check if the object has crossed the line
            if cy > line_y and track_id not in crossed_ids:
                crossed_ids.add(track_id)
                class_counts[class_name] += 1

        # Display the counts
        y_offset = 30
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

    # Show the frame
    cv2.imshow("Vehicle Detection & Counting", frame)

    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for ESC key
     break


# Release resources
cap.release()
cv2.destroyAllWindows()


