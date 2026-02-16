from ultralytics import YOLO
from app.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, IMAGE_SIZE


model = YOLO(YOLO_MODEL_PATH)


def detect_faces(image):
    """Detect faces in a PIL Image. Returns list of face dicts."""
    results = model.predict(
        source=image, save=False, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE
    )[0]

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
