from ultralytics import YOLO

from config import YOLO_MODEL_PATH


_face_yolo = None


def get_face_yolo():
    """Load the YOLOv8 face detector once and reuse it."""
    global _face_yolo
    if _face_yolo is None:
        _face_yolo = YOLO(YOLO_MODEL_PATH)
    return _face_yolo


def clamp_xyxy(box, image_width, image_height):
    """Clamp a YOLO xyxy box to image bounds."""
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    return x1, y1, x2, y2


def xyxy_to_css(box):
    """Convert x1, y1, x2, y2 into top, right, bottom, left."""
    x1, y1, x2, y2 = box
    return y1, x2, y2, x1


def full_image_css_box(image):
    """Return one face_recognition-style box covering the whole image."""
    height, width = image.shape[:2]
    return 0, width - 1, height - 1, 0


def yolo_face_locations(image, conf=0.5, max_faces=None):
    """Detect faces with yolov8n-face.pt and return CSS boxes."""
    if image is None or image.size == 0:
        return []

    image_height, image_width = image.shape[:2]
    detections = []
    results = get_face_yolo()(image, conf=conf, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = clamp_xyxy(
                box.xyxy[0],
                image_width,
                image_height,
            )
            if x2 <= x1 or y2 <= y1:
                continue

            confidence = float(box.conf[0]) if box.conf is not None else 0
            area = (x2 - x1) * (y2 - y1)
            detections.append((confidence, area, xyxy_to_css((x1, y1, x2, y2))))

    detections.sort(key=lambda detection: (detection[0], detection[1]), reverse=True)
    boxes = [box for _, _, box in detections]
    if max_faces is not None:
        return boxes[:max_faces]
    return boxes
