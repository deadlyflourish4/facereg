"""Recognition routes: /register, /verify, /identify"""
import io
from typing import List
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form

from app.services.recognition import register_user, verify_user, identify_face
from app.services.kafka_producer import (
    publish_register_event, publish_verify_event, publish_identify_event,
)

router = APIRouter(tags=["Recognition"])


def _read_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


@router.post("/register")
async def register(
    user_id: str = Form(...),
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Register a new user with one or more face images."""
    images = []
    for f in files:
        data = await f.read()
        images.append(_read_image(data))

    result = register_user(user_id, name, images)

    # Kafka event
    publish_register_event(
        user_id, name, result["status"],
        result.get("images_processed", 0),
    )

    return result


@router.post("/verify")
async def verify(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Verify 1:1 — does this face match this user?"""
    data = await file.read()
    image = _read_image(data)

    result = verify_user(user_id, image)

    # Kafka event
    publish_verify_event(
        user_id,
        result.get("match", False),
        result.get("similarity", 0),
    )

    return result


@router.post("/identify")
async def identify(file: UploadFile = File(...)):
    """Identify 1:N — who is this person?"""
    data = await file.read()
    image = _read_image(data)

    result = identify_face(image)

    # Kafka event
    candidates = result.get("candidates", [])
    top_match = candidates[0] if candidates else None
    publish_identify_event(len(candidates), top_match)

    return result
