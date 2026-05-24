DATASET_DIR = "dataset"
YOLO_MODEL_PATH = "yolov8n-face.pt"
COCO_MODEL_PATH = "yolov8n.pt"
FACE_MATCH_TOLERANCE = 0.46
PROFILE_FACE_MATCH_TOLERANCE = 0.50
FACE_MATCH_MARGIN = 0.04
FACE_CROP_PADDING = 0.25

# MySQL database credentials.
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "12345"
DB_NAME = "attendai"

# Liveness detection settings.
DLIB_LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.22
BLINK_OPEN_EAR_THRESHOLD = 0.12
BLINK_CONSEC_FRAMES = 2
BLINK_BASELINE_FRAMES = 3
BLINK_CLOSE_RATIO = 0.72
BLINK_REOPEN_RATIO = 0.86
BLINK_MIN_EAR_DROP = 0.035

# Class monitor settings.
PHONE_CLASS_ID = 67
PHONE_CLASS_LABEL = "cell phone"
MEDIAPIPE_EAR_THRESHOLD = 0.22
HEAD_POSE_YAW_THRESHOLD = 35
HEAD_POSE_PITCH_THRESHOLD = 30
ATTENTION_BASELINE_FRAMES = 15
ATTENTION_CONSEC_FRAMES = 10
AGGRESSIVE_MOVEMENT_THRESHOLD = 220
AGGRESSIVE_POSE_DELTA_THRESHOLD = 45
AGGRESSIVE_CONSEC_FRAMES = 8
PHONE_MISSING_RESET_FRAMES = 15
MONITOR_CROPS_DIR = "monitor_crops"

# Ollama report generation settings.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
REPORTS_DIR = "reports"

EMAIL_SENDER = "uawais967@gmail.com"
EMAIL_PASSWORD = "kduf psuc wicz mbbk"

# Theme persistence configuration
import json, os
THEME_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "theme_config.json")

def load_theme():
    """Load saved theme name from config file. Returns 'light' if not set or error."""
    try:
        with open(THEME_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("theme", "light")
    except Exception:
        return "light"

def save_theme(theme_name: str):
    """Save the given theme name to config file."""
    try:
        with open(THEME_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"theme": theme_name}, f)
    except Exception as e:
        print(f"[WARNING] Could not save theme config: {e}")
