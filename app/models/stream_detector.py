import time
import cv2
from ultralytics import YOLO
from app.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, IMAGE_SIZE, DETECT_EVERY_N_FRAMES


class StreamDetector:
    """Real-time face detector with frame skipping for performance."""

    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.detect_every = DETECT_EVERY_N_FRAMES
        self.last_faces = []
        self.frame_count = 0
        self.fps = 0.0
        self._fps_start = time.time()
        self._fps_frames = 0

    def process_frame(self, frame):
        """Process a BGR numpy frame. Returns cached or fresh detections."""
        self.frame_count += 1
        self._update_fps()

        if self.frame_count % self.detect_every == 0:
            results = self.model.predict(
                source=frame, save=False, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE
            )[0]
            self.last_faces = self._parse(results)

        return self.last_faces

    def _parse(self, results):
        faces = []
        for i in range(len(results.boxes)):
            bbox = results.boxes.xyxy[i].tolist()
            conf = results.boxes.conf[i].item()
            landmarks = None
            if results.keypoints is not None:
                landmarks = results.keypoints.xy[i].tolist()
            faces.append({
                "bbox": [round(v, 2) for v in bbox],
                "confidence": round(conf, 4),
                "landmarks": landmarks,
            })
        return faces

    def _update_fps(self):
        self._fps_frames += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self.fps = round(self._fps_frames / elapsed, 1)
            self._fps_frames = 0
            self._fps_start = time.time()

    def reset(self):
        self.last_faces = []
        self.frame_count = 0
        self.fps = 0.0
        self._fps_start = time.time()
        self._fps_frames = 0
