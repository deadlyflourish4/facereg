"""
Face Alignment Module
Aligns detected faces to a standard 112x112 template using affine transform.
"""
import cv2
import numpy as np

# ArcFace standard reference landmarks for 112x112 output
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def align_face(image, landmarks=None, bbox=None):
    """Align a face to 112x112 using landmarks or bbox fallback.

    Args:
        image: PIL Image or numpy array (RGB)
        landmarks: List of 5 landmark points [[x,y], ...]
        bbox: Bounding box [x1, y1, x2, y2] as fallback

    Returns:
        Aligned face as numpy array (RGB), shape (112, 112, 3)
    """
    img_array = np.array(image) if not isinstance(image, np.ndarray) else image

    if landmarks is not None and len(landmarks) >= 5:
        src_pts = np.array(landmarks[:5], dtype=np.float32)
        # Estimate affine transform
        transform, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_LANDMARKS)
        if transform is not None:
            aligned = cv2.warpAffine(img_array, transform, (112, 112))
            return aligned

    # Fallback: crop bbox and resize
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = img_array.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cropped = img_array[y1:y2, x1:x2]
        if cropped.size > 0:
            aligned = cv2.resize(cropped, (112, 112))
            return aligned

    # Last resort: resize entire image
    return cv2.resize(img_array, (112, 112))
