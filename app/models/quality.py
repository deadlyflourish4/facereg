import cv2
import numpy as np
import math
from app.config import (
    BLUR_THRESHOLD, BRIGHTNESS_LOW, BRIGHTNESS_HIGH,
    MIN_FACE_WIDTH, MIN_FACE_HEIGHT, MAX_POSE_ANGLE,
)


def check_blur(image_array):
    """Check image blur using Laplacian variance."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "is_blurry": score < BLUR_THRESHOLD,
        "score": round(score, 2),
    }


def check_brightness(image_array):
    """Check if image is too dark or too bright."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    mean_val = float(gray.mean())
    if mean_val < BRIGHTNESS_LOW:
        status = "too_dark"
    elif mean_val > BRIGHTNESS_HIGH:
        status = "too_bright"
    else:
        status = "normal"
    return {"status": status, "mean_brightness": round(mean_val, 2)}


def check_face_size(width, height):
    """Check if face crop is large enough."""
    return {
        "width": round(width, 2),
        "height": round(height, 2),
        "is_too_small": width < MIN_FACE_WIDTH or height < MIN_FACE_HEIGHT,
    }


def check_pose(landmarks):
    """Estimate roll angle from eye landmarks."""
    if landmarks is None or len(landmarks) < 2:
        return {"angle": 0.0, "is_tilted": False}
    eye_l, eye_r = landmarks[0], landmarks[1]
    dx = eye_r[0] - eye_l[0]
    dy = eye_r[1] - eye_l[1]
    angle = abs(math.degrees(math.atan2(dy, dx))) if dx != 0 else 90.0
    return {"angle": round(angle, 2), "is_tilted": angle > MAX_POSE_ANGLE}


def check_quality(face_crop, landmarks=None):
    """Run all quality checks on a face crop (PIL Image)."""
    img_array = np.array(face_crop)
    issues = []

    blur = check_blur(img_array)
    if blur["is_blurry"]:
        issues.append("too_blurry")

    brightness = check_brightness(img_array)
    if brightness["status"] != "normal":
        issues.append(brightness["status"])

    w, h = face_crop.size
    size = check_face_size(w, h)
    if size["is_too_small"]:
        issues.append("face_too_small")

    pose = check_pose(landmarks)
    if pose["is_tilted"]:
        issues.append("extreme_pose")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "scores": {
            "blur": blur["score"],
            "brightness": brightness["mean_brightness"],
            "face_size": {"width": w, "height": h},
            "angle": pose["angle"],
        },
    }
