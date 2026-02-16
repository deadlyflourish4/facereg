"""
Face Anti-Spoofing Module
Uses texture analysis (LBP + Laplacian) as a lightweight approach.
For production, replace with MiniFASNet or Silent-Face-Anti-Spoofing models.
"""
import cv2
import numpy as np
from app.config import SPOOF_THRESHOLD


def _compute_lbp_score(gray_image):
    """Compute Local Binary Pattern variance as a texture feature.
    Real faces have richer micro-texture than printed/screen images."""
    h, w = gray_image.shape
    lbp = np.zeros_like(gray_image, dtype=np.float32)

    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]:
        shifted = np.roll(np.roll(gray_image, dy, axis=0), dx, axis=1)
        lbp += (shifted >= gray_image).astype(np.float32)

    # Variance of LBP histogram as texture richness indicator
    hist, _ = np.histogram(lbp[1:-1, 1:-1].ravel(), bins=9, range=(0, 9))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return float(hist.var())


def _compute_frequency_score(gray_image):
    """High-frequency content analysis.
    Printed/screen images lose high-frequency details."""
    f_transform = np.fft.fft2(gray_image.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4

    # Ratio of high-freq to total energy
    total_energy = magnitude.sum() + 1e-6
    center_mask = np.zeros_like(magnitude)
    cv2.circle(center_mask, (cx, cy), radius, 1, -1)
    low_energy = (magnitude * center_mask).sum()
    high_ratio = 1.0 - (low_energy / total_energy)

    return float(high_ratio)


def _compute_color_score(image_bgr):
    """Color distribution analysis.
    Screen/print images have different color distribution than real skin."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Real faces have specific saturation range
    sat_mean = float(s.mean()) / 255.0
    sat_std = float(s.std()) / 255.0

    # Score: higher for natural skin-like saturation
    score = min(1.0, sat_std * 3.0 + 0.3) if 0.1 < sat_mean < 0.6 else 0.3
    return score


def check_liveness_image(face_crop_bgr):
    """Check if a single face crop is real or spoofed.

    Args:
        face_crop_bgr: Face crop as BGR numpy array

    Returns:
        dict with is_real, score, and details
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return {"is_real": False, "score": 0.0, "details": {}}

    # Resize for consistency
    face_resized = cv2.resize(face_crop_bgr, (128, 128))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    # Compute multiple signals
    lbp_score = _compute_lbp_score(gray)
    freq_score = _compute_frequency_score(gray)
    color_score = _compute_color_score(face_resized)

    # Laplacian sharpness (screens often have moiré patterns)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, laplacian_var / 500.0)

    # Weighted fusion
    final_score = (
        0.25 * min(1.0, lbp_score * 100)
        + 0.30 * freq_score
        + 0.25 * color_score
        + 0.20 * sharpness_score
    )
    final_score = round(min(1.0, max(0.0, final_score)), 4)

    return {
        "is_real": final_score > SPOOF_THRESHOLD,
        "score": final_score,
        "details": {
            "lbp_texture": round(lbp_score, 6),
            "frequency": round(freq_score, 4),
            "color": round(color_score, 4),
            "sharpness": round(sharpness_score, 4),
        },
    }


class FrameBuffer:
    """Collect frames for multi-frame liveness analysis."""

    def __init__(self, max_frames=60):
        self.max_frames = max_frames
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def is_ready(self):
        return len(self.frames) >= 30

    @property
    def progress(self):
        return min(1.0, len(self.frames) / self.max_frames)

    def get_frames(self):
        return list(self.frames)

    def clear(self):
        self.frames = []


def check_liveness_stream(frames):
    """Analyze multiple frames for liveness.

    Checks:
    1. Micro-movement (face position variance)
    2. Texture consistency across frames
    3. Per-frame anti-spoof scores

    Args:
        frames: List of BGR face crops

    Returns:
        dict with is_real, score, and details per signal
    """
    if len(frames) < 5:
        return {"is_real": False, "score": 0.0, "details": {"error": "not_enough_frames"}}

    # 1. Micro-movement analysis
    centers = []
    for frame in frames:
        h, w = frame.shape[:2]
        centers.append((w / 2.0, h / 2.0))

    if len(centers) > 1:
        displacements = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            displacements.append(np.sqrt(dx ** 2 + dy ** 2))
        movement_std = float(np.std(displacements))
    else:
        movement_std = 0.0

    if movement_std < 0.1:
        movement_label = "static"
        movement_pass = False
    elif movement_std < 0.5:
        movement_label = "suspicious"
        movement_pass = False
    else:
        movement_label = "natural"
        movement_pass = True

    # 2. Per-frame anti-spoof scores
    sample_indices = np.linspace(0, len(frames) - 1, min(10, len(frames)), dtype=int)
    fas_scores = []
    for idx in sample_indices:
        result = check_liveness_image(frames[idx])
        fas_scores.append(result["score"])

    avg_fas = float(np.mean(fas_scores))
    fas_pass = avg_fas > SPOOF_THRESHOLD

    # 3. Texture consistency
    consistency = 1.0 - float(np.std(fas_scores))
    consistency_pass = consistency > 0.7

    # Fusion
    signals = [movement_pass, fas_pass, consistency_pass]
    pass_count = sum(signals)
    final_score = round((avg_fas * 0.5 + (pass_count / 3.0) * 0.5), 4)

    return {
        "is_real": pass_count >= 2,
        "score": final_score,
        "details": {
            "movement": {"std": round(movement_std, 4), "label": movement_label, "passed": movement_pass},
            "texture_fas": {"avg_score": round(avg_fas, 4), "passed": fas_pass},
            "consistency": {"score": round(consistency, 4), "passed": consistency_pass},
        },
    }
