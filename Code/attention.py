import cv2
import numpy as np

from config import (
    HEAD_POSE_PITCH_THRESHOLD,
    HEAD_POSE_YAW_THRESHOLD,
    MEDIAPIPE_EAR_THRESHOLD,
)

try:
    import mediapipe as mp
except ImportError:
    mp = None


class AttentionTracker:
    def __init__(self):
        """Create a MediaPipe Face Mesh tracker for drowsiness and head-pose checks."""
        self.available = mp is not None
        self.face_mesh = None

        if not self.available:
            print("[WARNING] mediapipe is not installed. Attention tracking is unavailable.")
            return

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def prepare_rgb_image(image):
        """Convert an OpenCV image into RGB format for MediaPipe processing."""
        if image is None or image.size == 0:
            return None

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None

    @staticmethod
    def point_from_landmark(landmark, width, height):
        """Convert one normalized MediaPipe landmark into an image-space point."""
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float64)

    def eye_aspect_ratio(self, landmarks, indices, width, height):
        """Calculate eye aspect ratio from six MediaPipe eye landmarks."""
        eye = [self.point_from_landmark(landmarks[index], width, height) for index in indices]
        vertical_a = np.linalg.norm(eye[1] - eye[5])
        vertical_b = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        if horizontal == 0:
            return 0
        return (vertical_a + vertical_b) / (2.0 * horizontal)

    def estimate_head_pose(self, landmarks, width, height):
        """Estimate pitch and yaw angles from key Face Mesh landmarks with solvePnP."""
        image_points = np.array(
            [
                self.point_from_landmark(landmarks[1], width, height),
                self.point_from_landmark(landmarks[152], width, height),
                self.point_from_landmark(landmarks[33], width, height),
                self.point_from_landmark(landmarks[263], width, height),
                self.point_from_landmark(landmarks[61], width, height),
                self.point_from_landmark(landmarks[291], width, height),
            ],
            dtype=np.float64,
        )
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float64,
        )
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0, 0, 0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            projection_matrix
        )
        pitch = float(euler_angles[0])
        yaw = float(euler_angles[1])
        roll = float(euler_angles[2])
        return pitch, yaw, roll

    def analyze(self, face_image):
        """Return attention status, drowsiness, and distraction flags for one face."""
        result = {
            "status": "Attentive",
            "drowsy": False,
            "distracted": False,
            "ear": None,
            "pitch": 0,
            "yaw": 0,
        }

        if not self.available or self.face_mesh is None:
            result["status"] = "Attention unavailable"
            return result

        rgb_image = self.prepare_rgb_image(face_image)
        if rgb_image is None:
            return result

        height, width = rgb_image.shape[:2]
        if width < 20 or height < 20:
            return result

        rgb_image.flags.writeable = False
        mesh_results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True

        if not mesh_results.multi_face_landmarks:
            return result

        landmarks = mesh_results.multi_face_landmarks[0].landmark
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        left_ear = self.eye_aspect_ratio(landmarks, left_eye, width, height)
        right_ear = self.eye_aspect_ratio(landmarks, right_eye, width, height)
        ear = (left_ear + right_ear) / 2.0
        pitch, yaw, _ = self.estimate_head_pose(landmarks, width, height)

        result["ear"] = round(ear, 3)
        result["pitch"] = round(pitch, 2)
        result["yaw"] = round(yaw, 2)
        result["drowsy"] = ear < MEDIAPIPE_EAR_THRESHOLD
        result["distracted"] = (
            abs(yaw) > HEAD_POSE_YAW_THRESHOLD
            or abs(pitch) > HEAD_POSE_PITCH_THRESHOLD
        )

        if result["drowsy"]:
            result["status"] = "Drowsy"
        elif result["distracted"]:
            result["status"] = "Distracted"
        else:
            result["status"] = "Attentive"

        return result
