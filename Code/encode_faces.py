import os
import cv2
import face_recognition
import numpy as np
from PIL import Image
from config import DATASET_DIR
from database import save_student_encoding
from yolo_utils import yolo_face_locations


def prepare_rgb_image(image):
    """Convert an OpenCV image into contiguous RGB format for face_recognition."""
    if image is None or image.size == 0:
        return None

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return None
    else:
        return None

    pil_image = Image.fromarray(rgb_image).convert("RGB")
    return np.ascontiguousarray(np.array(pil_image), dtype=np.uint8)


def load_rgb_image_file(img_path):
    """Load a dataset image as an 8-bit RGB numpy array that dlib accepts."""
    try:
        image = face_recognition.load_image_file(img_path, mode="RGB")
        return np.ascontiguousarray(image, dtype=np.uint8)
    except Exception:
        image = cv2.imread(img_path)
        return prepare_rgb_image(image)


def read_sap_id(person_path):
    """Read the SAP-ID stored with a student's captured dataset images."""
    info_file = os.path.join(person_path, "info.txt")
    if not os.path.exists(info_file):
        return None

    with open(info_file, "r") as f:
        return f.readline().strip()


def encode_all_faces():
    """Encode all dataset faces and save the resulting numpy bytes to MySQL."""
    total_encoded_faces = 0
    total_saved_students = 0

    print("Encoding faces...")

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset folder not found: {DATASET_DIR}")
        return

    # Loop through each student folder
    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_path):
            continue

        sap_id = read_sap_id(person_path)
        if not sap_id:
            print(f"  [SKIP] Missing SAP-ID info.txt for: {person_name}")
            continue

        print(f"Processing : {person_name}")
        person_encodings = []

        # Loop through each image
        for img_name in sorted(os.listdir(person_path)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(person_path, img_name)

            # Read and convert image
            rgb_image = load_rgb_image_file(img_path)
            if rgb_image is None:
                print(f"  [SKIP] Unsupported image format: {img_name}")
                continue

            # Find face with YOLOv8 face model and create encoding
            try:
                boxes = yolo_face_locations(rgb_image, conf=0.45, max_faces=1)
                encodings = face_recognition.face_encodings(rgb_image, boxes)
            except Exception as error:
                print(f"  [SKIP] {img_name}: {error}")
                continue

            if len(encodings) > 0:
                # Validate face size (minimum 100x100 pixels for quality)
                box = boxes[0]
                y1, x2, y2, x1 = box
                face_width = x2 - x1
                face_height = y2 - y1

                if face_width < 100 or face_height < 100:
                    print(f"  [SKIP] {img_name}: Face too small ({face_width}x{face_height}px)")
                    continue

                person_encodings.append(encodings[0])
                total_encoded_faces += 1
            else:
                print(f"  [SKIP] No YOLO face found: {img_name}")

        if person_encodings:
            saved = save_student_encoding(
                sap_id,
                person_name,
                np.asarray(person_encodings, dtype=np.float64),
            )
            if saved:
                total_saved_students += 1
                print(f"  [OK] Saved {len(person_encodings)} encodings to MySQL")

    if total_encoded_faces > 0:
        print(
            f"Successfully encoded {total_encoded_faces} faces "
            f"for {total_saved_students} students"
        )
    else:
        print("[ERROR] No faces found")

if __name__ == "__main__":
    encode_all_faces()
