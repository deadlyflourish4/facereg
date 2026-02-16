"""WebSocket routes for real-time stream processing."""
import base64
import json
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.models.stream_detector import StreamDetector
from app.models.anti_spoof import FrameBuffer, check_liveness_stream
from app.services.kafka_producer import publish_detection_event, publish_spoof_alert

router = APIRouter(tags=["Stream"])


def _decode_frame(data: str):
    """Decode base64 JPEG to BGR numpy array."""
    # Handle data URL format: "data:image/jpeg;base64,..."
    if "," in data:
        data = data.split(",", 1)[1]
    jpeg_bytes = base64.b64decode(data)
    np_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


@router.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Real-time face detection via WebSocket.
    Client sends base64 JPEG frames, server returns face detections."""
    await ws.accept()
    detector = StreamDetector()

    try:
        while True:
            data = await ws.receive_text()
            frame = _decode_frame(data)
            if frame is None:
                await ws.send_json({"error": "cannot_decode_frame"})
                continue

            faces = detector.process_frame(frame)
            await ws.send_json({
                "frame_id": detector.frame_count,
                "fps": detector.fps,
                "total_faces": len(faces),
                "faces": faces,
            })
    except WebSocketDisconnect:
        detector.reset()


@router.websocket("/ws/liveness")
async def ws_liveness(ws: WebSocket):
    """Stream-based liveness check via WebSocket.
    Collects frames for 2 seconds, then runs multi-frame anti-spoofing."""
    await ws.accept()
    detector = StreamDetector()
    buffer = FrameBuffer(max_frames=60)

    await ws.send_json({"status": "start_recording"})

    try:
        while True:
            data = await ws.receive_text()
            frame = _decode_frame(data)
            if frame is None:
                continue

            # Detect face first
            faces = detector.process_frame(frame)
            if not faces:
                await ws.send_json({"status": "no_face", "progress": buffer.progress})
                continue

            # Crop first face and add to buffer
            face = faces[0]
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                buffer.add_frame(face_crop)

            await ws.send_json({
                "status": "recording",
                "progress": round(buffer.progress, 2),
            })

            # When buffer is ready, run liveness check
            if buffer.is_ready():
                result = check_liveness_stream(buffer.get_frames())

                if not result["is_real"]:
                    publish_spoof_alert(result["score"], result["details"])

                await ws.send_json({"status": "done", "result": result})
                break

    except WebSocketDisconnect:
        buffer.clear()
        detector.reset()


@router.websocket("/ws/monitor")
async def ws_monitor(ws: WebSocket):
    """Continuous face monitoring via WebSocket.
    Detects and attempts to identify faces in real-time."""
    await ws.accept()
    detector = StreamDetector()
    frame_number = 0

    try:
        while True:
            data = await ws.receive_text()
            frame = _decode_frame(data)
            if frame is None:
                continue

            frame_number += 1
            faces = detector.process_frame(frame)

            # Only attempt identification every 30 frames (1 per second at 30fps)
            identifications = []
            if frame_number % 30 == 0 and faces:
                from PIL import Image
                from app.services.recognition import identify_face

                for face in faces[:3]:  # max 3 faces to avoid overload
                    x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(face_rgb)
                    result = identify_face(pil_image)

                    identifications.append({
                        "bbox": face["bbox"],
                        "candidates": result.get("candidates", []),
                    })

            await ws.send_json({
                "frame_id": frame_number,
                "fps": detector.fps,
                "total_faces": len(faces),
                "faces": faces,
                "identifications": identifications,
            })

    except WebSocketDisconnect:
        detector.reset()


@router.post("/stream/video")
async def stream_video(file: UploadFile):
    """Upload a video file → detect faces frame-by-frame → return SSE stream."""

    video_bytes = await file.read()
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    def generate():
        detector = StreamDetector()
        cap = cv2.VideoCapture(temp_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            faces = detector.process_frame(frame)
            result = json.dumps({
                "frame": detector.frame_count,
                "fps": detector.fps,
                "faces": faces,
            })
            yield f"data: {result}\n\n"

        cap.release()
        import os
        os.remove(temp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")
