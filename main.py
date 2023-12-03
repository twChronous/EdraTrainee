from ultralytics import YOLO
import cv2
import cvzone
import math
import os

model_path = os.path.join('.', 'train', 'runs', 'detect', 'train', 'weights', 'last.pt')
cap = cv2.VideoCapture(1)
model = YOLO(model_path) 

classNames = ["base"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),3)

            w, h, = x2 - x1, y2 - y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            # Confidence
            conf = math.ceil((box.conf[0] *100))

            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}%', (max(0, x1), max(35, y1)),scale=0.9,thickness=2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Image",img)
    cv2.waitKey(1)