"""
FaceReg — Face Recognition System
FastAPI application with REST + WebSocket endpoints.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.routes import detection, recognition, users, websocket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="FaceReg — Face Recognition System",
    description=(
        "Complete face recognition pipeline: detection → quality assessment → "
        "anti-spoofing → alignment → embedding → recognition. "
        "Supports both image upload (REST) and video stream (WebSocket)."
    ),
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection.router)
app.include_router(recognition.router)
app.include_router(users.router)
app.include_router(websocket.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "modules": {
            "detection": "YOLOv11n-face",
            "quality": "OpenCV (blur, brightness, pose)",
            "anti_spoof": "Texture analysis (LBP + FFT + color)",
            "alignment": "Affine transform (5-point → 112x112)",
            "embedding": "ArcFace (InsightFace buffalo_l)",
            "storage": "JSON file",
            "streaming": "WebSocket + SSE",
            "events": "Kafka (optional)",
        },
    }


@app.on_event("startup")
async def startup():
    logging.info("FaceReg starting up...")
    logging.info("REST endpoints: /detect, /quality-check, /anti-spoof, /register, /verify, /identify")
    logging.info("WebSocket endpoints: /ws/detect, /ws/liveness, /ws/monitor")
    logging.info("Frontend: http://localhost:8000/")

    # Ensure data directory exists
    import os
    os.makedirs("data", exist_ok=True)

    # Scan data/faces/ and auto-register users
    from app.services.face_loader import scan_and_register
    logging.info("Scanning data/faces/ for pre-registered users...")
    scan_and_register()


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


# Serve frontend (must be last — catches /static/*)
app.mount("/static", StaticFiles(directory="static"), name="static")