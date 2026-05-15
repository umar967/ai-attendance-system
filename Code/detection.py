from ultralytics import YOLO

from config import YOLO_MODEL_PATH


class FaceDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model(frame, conf=0.5, verbose=False)
        boxes = []

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

        return boxes
