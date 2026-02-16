import os

# --- Model paths ---
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/yolov11n-face.pt")
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")

# --- Detection ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "3"))

# --- Quality ---
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100"))
BRIGHTNESS_LOW = float(os.getenv("BRIGHTNESS_LOW", "40"))
BRIGHTNESS_HIGH = float(os.getenv("BRIGHTNESS_HIGH", "220"))
MIN_FACE_WIDTH = int(os.getenv("MIN_FACE_WIDTH", "80"))
MIN_FACE_HEIGHT = int(os.getenv("MIN_FACE_HEIGHT", "80"))
MAX_POSE_ANGLE = float(os.getenv("MAX_POSE_ANGLE", "30"))

# --- Anti-Spoofing ---
SPOOF_THRESHOLD = float(os.getenv("SPOOF_THRESHOLD", "0.5"))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.20"))
MAX_BUFFER_FRAMES = int(os.getenv("MAX_BUFFER_FRAMES", "60"))

# --- Recognition ---
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.5"))
IDENTIFY_THRESHOLD = float(os.getenv("IDENTIFY_THRESHOLD", "0.4"))
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "5"))

# --- Storage ---
FACE_DB_PATH = os.getenv("FACE_DB_PATH", "data/face_db.json")
FACES_DIR = os.getenv("FACES_DIR", "data/faces")  # folder-based registration

# --- Kafka ---
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_FACE_EVENTS = os.getenv("KAFKA_TOPIC_FACE_EVENTS", "face-events")
KAFKA_TOPIC_ALERTS = os.getenv("KAFKA_TOPIC_ALERTS", "face-alerts")
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"

# --- Server ---
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")  # "cpu" or "cuda"
