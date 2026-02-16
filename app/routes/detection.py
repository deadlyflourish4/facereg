"""Detection routes: /detect, /quality-check, /anti-spoof"""
import io
import time
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File

from app.models.detector import detect_faces
from app.models.quality import check_quality
from app.models.anti_spoof import check_liveness_image
from app.services.kafka_producer import publish_detection_event, publish_spoof_alert

router = APIRouter(tags=["Detection"])


def _read_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Detect faces in an uploaded image."""
    file_bytes = await file.read()
    image = _read_image(file_bytes)

    start = time.time()
    faces = detect_faces(image)
    processing_time = round(time.time() - start, 4)

    # Kafka event
    publish_detection_event(len(faces), processing_time)

    return {
        "total_faces": len(faces),
        "processing_time": processing_time,
        "faces": faces,
    }


@router.post("/quality-check")
async def quality_check(file: UploadFile = File(...)):
    """Detect faces and run quality assessment."""
    file_bytes = await file.read()
    image = _read_image(file_bytes)

    faces = detect_faces(image)
    if not faces:
        return {"total_faces": 0, "message": "No face detected", "results": []}

    results = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        face_crop = image.crop((x1, y1, x2, y2))
        quality = check_quality(face_crop, face["landmarks"])
        results.append({
            "bbox": face["bbox"],
            "confidence": face["confidence"],
            "quality": quality,
        })

    return {"total_faces": len(faces), "results": results}


@router.post("/anti-spoof")
async def anti_spoof(file: UploadFile = File(...)):
    """Check if the face in an image is real or spoofed."""
    file_bytes = await file.read()
    image = _read_image(file_bytes)

    faces = detect_faces(image)
    if not faces:
        return {"error": "no_face_detected"}

    face = faces[0]
    x1, y1, x2, y2 = face["bbox"]
    face_crop = np.array(image.crop((x1, y1, x2, y2)))
    face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

    result = check_liveness_image(face_bgr)

    # Alert if spoofed
    if not result["is_real"]:
        publish_spoof_alert(result["score"], result["details"])

    return result


@router.post("/anti-spoof/video")
async def anti_spoof_video(file: UploadFile = File(...)):
    """Analyze a video file for liveness (multi-frame)."""
    from app.models.anti_spoof import check_liveness_stream

    video_bytes = await file.read()
    temp_path = "temp_video_spoof.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_path)
    frames = []
    while cap.isOpened() and len(frames) < 60:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    import os
    os.remove(temp_path)

    if len(frames) < 5:
        return {"error": "video_too_short", "frames_read": len(frames)}

    result = check_liveness_stream(frames)
    return result
