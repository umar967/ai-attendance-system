from ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, model_path):
        self.model = YOLO("yolov8n.pt")


    def detect_faces(self, frame):
        results = self.model(frame)
        boxes = []

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

        return boxes
