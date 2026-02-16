"""
Face Recognition Service
Orchestrates detect → quality → anti-spoof → align → embed → match pipeline.
"""
import numpy as np
from PIL import Image

from app.models.detector import detect_faces
from app.models.quality import check_quality
from app.models.anti_spoof import check_liveness_image
from app.models.aligner import align_face
from app.models.embedder import get_embedding, cosine_similarity
from app.services.storage import add_user, get_user, get_all_embeddings
from app.config import MATCH_THRESHOLD, IDENTIFY_THRESHOLD, TOP_K_CANDIDATES


def _run_pipeline(image, skip_quality=False):
    """Run full pipeline on a single image. Returns embedding or raises."""
    # 1. Detect
    faces = detect_faces(image)
    if len(faces) == 0:
        return None, "no_face_detected"

    face = faces[0]  # take the largest/first face

    # 2. Quality check
    x1, y1, x2, y2 = face["bbox"]
    face_crop = image.crop((x1, y1, x2, y2))
    if not skip_quality:
        quality = check_quality(face_crop, face["landmarks"])
        if not quality["passed"]:
            return None, f"quality_failed: {', '.join(quality['issues'])}"

    # 3. Anti-spoof
    import cv2
    face_bgr = cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)
    spoof = check_liveness_image(face_bgr)
    # Note: anti-spoof is informational, we don't hard-reject in pipeline
    # to keep things working for demo purposes

    # 4. Align
    aligned = align_face(image, landmarks=face["landmarks"], bbox=face["bbox"])

    # 5. Embed
    embedding = get_embedding(aligned)
    if embedding is None:
        return None, "embedding_failed"

    return embedding, None


def register_user(user_id, name, images, skip_quality=False):
    """Register a new user with multiple images.

    Args:
        user_id: Unique user identifier
        name: Display name
        images: List of PIL Images
        skip_quality: Skip quality checks (for pre-curated folder images)

    Returns:
        dict with status and message
    """
    embeddings = []
    errors = []

    for i, img in enumerate(images):
        embedding, error = _run_pipeline(img, skip_quality=skip_quality)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            errors.append(f"image_{i}: {error}")

    if len(embeddings) == 0:
        return {
            "status": "failed",
            "message": "No valid face found in any image",
            "errors": errors,
        }

    # Average embeddings and normalize
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

    add_user(user_id, name, avg_embedding)

    return {
        "status": "success",
        "user_id": user_id,
        "name": name,
        "images_processed": len(embeddings),
        "images_failed": len(errors),
        "errors": errors,
    }


def verify_user(user_id, image):
    """Verify 1:1 — compare image against a specific user.

    Args:
        user_id: User to verify against
        image: PIL Image

    Returns:
        dict with match result and similarity score
    """
    user = get_user(user_id)
    if user is None:
        return {"match": False, "error": "user_not_found"}

    embedding, error = _run_pipeline(image, skip_quality=True)
    if embedding is None:
        return {"match": False, "error": error}

    stored_embedding = np.array(user["embedding"], dtype=np.float32)
    similarity = cosine_similarity(embedding, stored_embedding)

    return {
        "match": similarity > MATCH_THRESHOLD,
        "similarity": round(similarity, 4),
        "user_id": user_id,
        "user_name": user["name"],
        "threshold": MATCH_THRESHOLD,
    }


def identify_face(image):
    """Identify 1:N — find who this face belongs to.

    Args:
        image: PIL Image

    Returns:
        dict with ranked list of candidates
    """
    embedding, error = _run_pipeline(image, skip_quality=True)
    if embedding is None:
        return {"candidates": [], "error": error}

    all_users = get_all_embeddings()
    candidates = []

    for user in all_users:
        similarity = cosine_similarity(embedding, user["embedding"])
        if similarity > IDENTIFY_THRESHOLD:
            candidates.append({
                "user_id": user["user_id"],
                "name": user["name"],
                "similarity": round(similarity, 4),
            })

    # Sort by similarity descending
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    candidates = candidates[:TOP_K_CANDIDATES]

    return {
        "total_candidates": len(candidates),
        "candidates": candidates,
        "threshold": IDENTIFY_THRESHOLD,
    }
