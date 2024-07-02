import cv2
import torch
from ultralytics import YOLO

model = YOLO('yolov8s.pt') 

video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', frame_width, frame_height)

alert_threshold = 7 
fps = cap.get(cv2.CAP_PROP_FPS)
alert_frame_threshold = int(alert_threshold * fps)

cut_in_detected = False
cut_in_start_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    car_detected = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if model.names[cls] == 'car':
                car_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if car_detected:
        if not cut_in_detected:
            cut_in_detected = True
            cut_in_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    else:
        cut_in_detected = False

    if cut_in_detected:
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_frame - cut_in_start_frame >= alert_frame_threshold:
            print("ALERT: Vehicle cut-in detected for 7 seconds!")
            cut_in_detected = False  

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
