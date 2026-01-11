# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO

def process_person_phone(img, yolo_model):
    """
    Process a single frame for person and phone detection using YOLOv11n.
    Returns the processed image and a list of alerts.
    """
    alerts = []
    
    # Run inference
    # verbose=False to clean up logs, conf=0.5 for confidence threshold
    results = yolo_model(img, verbose=False, conf=0.5, save=False) 
    
    person_count = 0
    phone_detected = False
    
    # Iterate through detections
    # YOLOv11 (COCO) classes:
    # 0: person
    # 67: cell phone
    
    for r in results:
        # We manually draw to control what is shown
        # r.plot() draws everything, which annoys the user if 1 person is present.
        
        boxes = r.boxes
        
        # First count people
        person_boxes = []
        phone_boxes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0: # Person
                person_boxes.append(box)
            elif cls_id == 67: # Cell Phone
                phone_boxes.append(box)
        
        person_count = len(person_boxes)
        if len(phone_boxes) > 0:
            phone_detected = True
            
        # Draw Phones (Always)
        for box in phone_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Draw Persons (Only if > 1, to show intruders)
        # If count == 1, we assume it's the candidate and show nothing (clean feed)
        if person_count > 1:
            for box in person_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for person
                cv2.putText(img, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
    if phone_detected:
        alerts.append('Mobile Phone detected')
        
    if person_count == 0:
        alerts.append('No person detected')
    elif person_count > 1:
        alerts.append('More than one person detected')
    
    return img, alerts

def detect_phone_and_person(video_path):
    # Standalone testing function
    yolo_model = YOLO("yolo11n.pt") 
    cap = cv2.VideoCapture(video_path if video_path else 0)
    
    if not cap.isOpened():
        return

    while(True):
        ret, image = cap.read()
        if not ret:
            break
            
        image, alerts = process_person_phone(image, yolo_model)
        for alert in alerts:
            print(alert)

        cv2.imshow('Prediction', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


