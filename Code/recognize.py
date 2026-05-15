import os
import time

import cv2
import face_recognition
import numpy as np
from PIL import Image

from config import (
    BLINK_BASELINE_FRAMES,
    BLINK_CLOSE_RATIO,
    BLINK_CONSEC_FRAMES,
    BLINK_MIN_EAR_DROP,
    BLINK_OPEN_EAR_THRESHOLD,
    BLINK_REOPEN_RATIO,
    DLIB_LANDMARK_MODEL,
    EAR_THRESHOLD,
    FACE_MATCH_MARGIN,
    FACE_MATCH_TOLERANCE,
    PROFILE_FACE_MATCH_TOLERANCE,
)
from database import load_face_encodings
from yolo_utils import full_image_css_box, yolo_face_locations

try:
    import dlib
except ImportError:
    dlib = None

try:
    import face_recognition_models
except ImportError:
    face_recognition_models = None


class FaceRecognizer:
    # Initialize the recognizer with known encodings
    def __init__(self, encodings_file=None):
        """Load known face encodings from MySQL and prepare dlib liveness checks."""
        self.known_encodings = []
        self.known_names = []
        self.known_sap_ids = []
        self.blink_states = {}
        self.blink_verified = set()
        self.last_rejection_log_time = 0
        self.landmark_predictor = self.load_landmark_predictor()
        self.reload_encodings()

    def reload_encodings(self):
        """Refresh face encodings from MySQL for recognition."""
        try:
            (
                self.known_encodings,
                self.known_names,
                self.known_sap_ids,
            ) = load_face_encodings()
            print(f"Loaded {len(self.known_encodings)} face encodings from MySQL")
        except Exception as error:
            print(f"Could not load face encodings from MySQL: {error}")
            self.known_encodings = []
            self.known_names = []
            self.known_sap_ids = []

    def load_landmark_predictor(self):
        """Load the dlib 68-point facial landmark model used for blink detection."""
        if dlib is None:
            print("[WARNING] dlib is not installed. Blink verification is unavailable.")
            return None

        model_path = DLIB_LANDMARK_MODEL
        if not os.path.exists(model_path) and face_recognition_models is not None:
            model_path = face_recognition_models.pose_predictor_model_location()

        if not model_path or not os.path.exists(model_path):
            print("[WARNING] dlib landmark model not found. Blink verification is unavailable.")
            return None

        try:
            return dlib.shape_predictor(model_path)
        except Exception as error:
            print(f"[WARNING] Could not load dlib landmark predictor: {error}")
            return None

    @staticmethod
    def prepare_rgb_face(face_image):
        """Convert a cropped face image into contiguous RGB format."""
        if face_image is None or face_image.size == 0:
            return None

        if face_image.dtype != np.uint8:
            face_image = np.clip(face_image, 0, 255).astype(np.uint8)

        if len(face_image.shape) == 2:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif len(face_image.shape) == 3:
            if face_image.shape[2] == 4:
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
            elif face_image.shape[2] == 3:
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                return None
        else:
            return None

        pil_face = Image.fromarray(rgb_face).convert("RGB")
        return np.ascontiguousarray(np.array(pil_face), dtype=np.uint8)

    @staticmethod
    def calculate_ear(eye):
        """Calculate eye aspect ratio from six dlib eye landmark points."""
        vertical_a = np.linalg.norm(eye[1] - eye[5])
        vertical_b = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        if horizontal == 0:
            return 0
        return (vertical_a + vertical_b) / (2.0 * horizontal)

    def calculate_blink_ear(self, rgb_face):
        """Detect facial landmarks and return the average EAR for both eyes."""
        if self.landmark_predictor is None:
            return None

        gray_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
        height, width = gray_face.shape[:2]
        if width < 20 or height < 20:
            return None

        rectangle = dlib.rectangle(0, 0, width - 1, height - 1)
        try:
            shape = self.landmark_predictor(gray_face, rectangle)
        except Exception:
            return None

        points = np.array(
            [(shape.part(index).x, shape.part(index).y) for index in range(68)],
            dtype=np.float64,
        )
        left_eye = points[36:42]
        right_eye = points[42:48]
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        return (left_ear + right_ear) / 2.0

    def update_blink_state(self, sap_id, rgb_face):
        """Confirm one open-close-open blink before allowing attendance marking."""
        if sap_id in self.blink_verified:
            return True, None

        ear = self.calculate_blink_ear(rgb_face)
        if ear is None:
            return False, None

        state = self.blink_states.setdefault(
            sap_id,
            {
                "open_ear": ear,
                "open_frames": 0,
                "closed_frames": 0,
                "waiting_for_reopen": False,
            },
        )
        open_ear = max(float(state.get("open_ear") or ear), 0.01)
        close_threshold = min(EAR_THRESHOLD, open_ear * BLINK_CLOSE_RATIO)
        reopen_threshold = max(BLINK_OPEN_EAR_THRESHOLD, open_ear * BLINK_REOPEN_RATIO)
        enough_drop = (open_ear - ear) >= BLINK_MIN_EAR_DROP
        eyes_closed = (
            state["open_frames"] >= BLINK_BASELINE_FRAMES
            and ear <= close_threshold
            and enough_drop
        )

        if state["waiting_for_reopen"]:
            if ear >= reopen_threshold:
                self.blink_verified.add(sap_id)
                state["waiting_for_reopen"] = False
                state["closed_frames"] = 0
            elif ear <= close_threshold:
                state["closed_frames"] += 1
            return sap_id in self.blink_verified, ear

        if eyes_closed:
            state["closed_frames"] += 1
            if state["closed_frames"] >= BLINK_CONSEC_FRAMES:
                state["waiting_for_reopen"] = True
            return False, ear

        if ear > open_ear:
            state["open_ear"] = (open_ear * 0.7) + (ear * 0.3)
        elif ear >= open_ear * BLINK_REOPEN_RATIO:
            state["open_ear"] = (open_ear * 0.95) + (ear * 0.05)

        state["open_frames"] = min(
            int(state["open_frames"]) + 1,
            BLINK_BASELINE_FRAMES,
        )
        state["closed_frames"] = 0

        return sap_id in self.blink_verified, ear

    def get_face_encodings(self, rgb_face):
        """Create face encodings from YOLO face boxes, with full-crop fallback."""
        try:
            boxes = yolo_face_locations(rgb_face, conf=0.35, max_faces=1)
            encodings = face_recognition.face_encodings(rgb_face, boxes)
        except Exception:
            boxes = []
            encodings = []

        if encodings:
            return encodings, False

        try:
            return (
                face_recognition.face_encodings(
                    rgb_face,
                    [full_image_css_box(rgb_face)],
                    model="large",
                ),
                True,
            )
        except Exception:
            return [], True

    def get_best_student_match(self, face_distances):
        """Return the closest student after grouping multiple images by SAP-ID."""
        student_matches = {}

        for index, distance in enumerate(face_distances):
            if index >= len(self.known_sap_ids) or index >= len(self.known_names):
                continue

            sap_id = self.known_sap_ids[index]
            if not sap_id:
                continue

            distance = float(distance)
            current_match = student_matches.get(sap_id)
            if current_match is None or distance < current_match["distance"]:
                student_matches[sap_id] = {
                    "index": index,
                    "sap_id": sap_id,
                    "name": self.known_names[index],
                    "distance": distance,
                }

        if not student_matches:
            return None

        ranked_matches = sorted(
            student_matches.values(),
            key=lambda match: match["distance"],
        )
        best_match = ranked_matches[0].copy()

        if len(ranked_matches) > 1:
            second_match = ranked_matches[1]
            best_match["second_distance"] = second_match["distance"]
            best_match["second_name"] = second_match["name"]
            best_match["second_sap_id"] = second_match["sap_id"]
        else:
            best_match["second_distance"] = None
            best_match["second_name"] = None
            best_match["second_sap_id"] = None

        return best_match

    def log_rejected_match(self, recognition):
        """Throttle console diagnostics for faces that nearly matched a student."""
        if recognition.get("best_distance") is None:
            return

        now = time.monotonic()
        if now - self.last_rejection_log_time < 3:
            return

        self.last_rejection_log_time = now
        print(
            "[DEBUG] Face rejected "
            f"({recognition.get('reason')}): "
            f"closest={recognition.get('best_name')} "
            f"distance={recognition.get('best_distance'):.3f} "
            f"tolerance={recognition.get('tolerance'):.2f}"
        )

    def recognize_with_details(self, face_image, check_liveness=True):
        """Recognize a face and include SAP-ID plus blink liveness status."""
        unknown_result = {
            "name": "Unknown",
            "sap_id": None,
            "live": False,
            "ear": None,
            "best_name": None,
            "best_sap_id": None,
            "best_distance": None,
            "tolerance": None,
            "used_fallback": False,
            "blink_waiting": False,
            "reason": "unknown",
        }

        if not self.known_encodings:
            return unknown_result

        rgb_face = self.prepare_rgb_face(face_image)
        if rgb_face is None:
            return unknown_result

        encodings, used_fallback = self.get_face_encodings(rgb_face)
        if len(encodings) == 0:
            return unknown_result

        face_encoding = encodings[0]
        face_distances = face_recognition.face_distance(
            self.known_encodings,
            face_encoding,
        )
        if len(face_distances) == 0:
            return unknown_result

        best_match = self.get_best_student_match(face_distances)
        if best_match is None:
            return unknown_result

        best_match_index = best_match["index"]
        best_distance = best_match["distance"]
        tolerance = (
            PROFILE_FACE_MATCH_TOLERANCE if used_fallback else FACE_MATCH_TOLERANCE
        )
        unknown_result.update(
            {
                "best_name": best_match["name"],
                "best_sap_id": best_match["sap_id"],
                "best_distance": best_distance,
                "tolerance": tolerance,
                "used_fallback": used_fallback,
            }
        )
        if best_distance > tolerance:
            unknown_result["reason"] = "distance"
            self.log_rejected_match(unknown_result)
            return unknown_result

        second_best_distance = best_match["second_distance"]
        if (
            second_best_distance is not None
            and (second_best_distance - best_distance) < FACE_MATCH_MARGIN
        ):
            unknown_result["reason"] = "ambiguous"
            self.log_rejected_match(unknown_result)
            return unknown_result

        sap_id = self.known_sap_ids[best_match_index]
        if check_liveness:
            live, ear = self.update_blink_state(sap_id, rgb_face)
        else:
            live, ear = True, None
        blink_state = self.blink_states.get(sap_id, {})

        return {
            "name": self.known_names[best_match_index],
            "sap_id": sap_id,
            "live": live,
            "ear": ear,
            "best_name": self.known_names[best_match_index],
            "best_sap_id": sap_id,
            "best_distance": best_distance,
            "tolerance": tolerance,
            "used_fallback": used_fallback,
            "blink_waiting": blink_state.get("waiting_for_reopen", False),
            "reason": "matched",
        }

    # Recognize a face from an image
    def recognize(self, face_image):
        """Return only the recognized name for compatibility with older callers."""
        return self.recognize_with_details(face_image)["name"]


if __name__ == "__main__":
    # Test the recognizer
    recognizer = FaceRecognizer()
    print(f"Ready to recognize {len(recognizer.known_names)} students")
