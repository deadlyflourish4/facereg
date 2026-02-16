"""
Startup face loader.
Scans data/faces/ for subfolders. Each subfolder = one user.
Folder name = user display name, images inside = face photos.
Builds embedding DB automatically on app startup.

Structure:
    data/faces/
    ├── nguyen_van_a/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── 3.jpg
    └── tran_van_b/
        ├── 1.jpg
        └── 2.jpg
"""
import os
import logging
from pathlib import Path
from PIL import Image

from app.config import FACES_DIR
from app.services.storage import load_db, get_user
from app.services.recognition import register_user

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_and_register():
    """Scan FACES_DIR and register any new users not already in DB."""
    faces_dir = Path(FACES_DIR)
    if not faces_dir.exists():
        os.makedirs(faces_dir, exist_ok=True)
        logger.info(f"Created faces directory: {faces_dir}")
        logger.info("Put subfolders with images here to auto-register users on startup.")
        return

    subfolders = [f for f in faces_dir.iterdir() if f.is_dir()]
    if not subfolders:
        logger.info(f"No user folders found in {faces_dir}")
        return

    logger.info(f"Found {len(subfolders)} user folder(s) in {faces_dir}")

    registered = 0
    skipped = 0

    for folder in sorted(subfolders):
        user_name = folder.name
        user_id = user_name.lower().replace(" ", "_")

        # Skip if already registered
        existing = get_user(user_id)
        if existing:
            logger.info(f"  ⏭  {user_name} (already registered)")
            skipped += 1
            continue

        # Collect images
        images = []
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    img = Image.open(file).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning(f"  ⚠  Cannot read {file.name}: {e}")

        if not images:
            logger.warning(f"  ⚠  {user_name}: no valid images found")
            continue

        # Register via the full pipeline
        logger.info(f"  📸 Registering {user_name} ({len(images)} images)...")
        try:
            result = register_user(user_id, user_name, images, skip_quality=True)
            if result["status"] == "success":
                logger.info(f"  ✅ {user_name} registered ({result['images_processed']} images)")
                registered += 1
            else:
                logger.warning(f"  ❌ {user_name} failed: {result.get('message', result.get('errors', ''))}")
        except Exception as e:
            logger.error(f"  ❌ {user_name} error: {e}")

    logger.info(f"Startup scan done: {registered} new, {skipped} existing")
