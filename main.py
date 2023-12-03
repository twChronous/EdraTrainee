import os
import cv2
from ultralytics import YOLO
model_path = os.path.join('.', 'train', 'runs', 'detect', 'train2', 'weights', 'last.pt')


#model = YOLO(model_path)  

def exibir_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar o frame.")
            break

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


print(model_path)
exibir_webcam()



