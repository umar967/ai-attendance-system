import numpy as np
from PIL import Image

import dlib
import face_recognition_models


_pose_predictor_68 = dlib.shape_predictor(
    face_recognition_models.pose_predictor_model_location()
)
_pose_predictor_5 = dlib.shape_predictor(
    face_recognition_models.pose_predictor_five_point_model_location()
)
_face_encoder = dlib.face_recognition_model_v1(
    face_recognition_models.face_recognition_model_location()
)


def load_image_file(file, mode="RGB"):
    """Load an image file into an 8-bit RGB numpy array."""
    image = Image.open(file)
    if mode:
        image = image.convert(mode)
    return np.array(image)


def _css_to_rect(css):
    """Convert top, right, bottom, left coordinates to a dlib rectangle."""
    top, right, bottom, left = css
    return dlib.rectangle(left, top, right, bottom)


def _trim_css_to_bounds(css, image_shape):
    """Keep a CSS face box inside image bounds."""
    top, right, bottom, left = css
    height, width = image_shape[:2]
    return (
        max(top, 0),
        min(right, width),
        min(bottom, height),
        max(left, 0),
    )


def face_locations(img, number_of_times_to_upsample=1, model="yolo"):
    """Return YOLO face boxes as top, right, bottom, left tuples."""
    del number_of_times_to_upsample
    if model not in {"yolo", "yolov8", "yolov8n-face"}:
        raise ValueError("Only YOLOv8 face detection is available in this project.")

    from yolo_utils import yolo_face_locations

    return [
        _trim_css_to_bounds(face_box, img.shape)
        for face_box in yolo_face_locations(img)
    ]


def _raw_face_landmarks(face_image, face_locations_value=None, model="small"):
    """Return dlib facial landmarks for detected or supplied face boxes."""
    if face_locations_value is None:
        face_locations_value = face_locations(face_image)

    pose_predictor = _pose_predictor_5 if model == "small" else _pose_predictor_68
    return [
        pose_predictor(face_image, _css_to_rect(face_location))
        for face_location in face_locations_value
    ]


def face_encodings(
    face_image,
    known_face_locations=None,
    num_jitters=1,
    model="small",
):
    """Return 128-dimensional face encodings for each supplied face location."""
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    return [
        np.array(
            _face_encoder.compute_face_descriptor(
                face_image,
                raw_landmark_set,
                num_jitters,
            )
        )
        for raw_landmark_set in raw_landmarks
    ]


def face_distance(face_encodings_value, face_to_compare):
    """Return Euclidean distances between known encodings and one face encoding."""
    if len(face_encodings_value) == 0:
        return np.empty((0,))

    return np.linalg.norm(
        np.asarray(face_encodings_value) - np.asarray(face_to_compare),
        axis=1,
    )
