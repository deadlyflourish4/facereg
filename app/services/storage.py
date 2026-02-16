"""
Face Database Storage
Simple JSON file storage for face embeddings.
"""
import json
import os
import numpy as np
from datetime import datetime
from app.config import FACE_DB_PATH


def _ensure_dir():
    os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)


def load_db():
    """Load the face database from JSON file."""
    _ensure_dir()
    if not os.path.exists(FACE_DB_PATH):
        return {"users": {}}
    with open(FACE_DB_PATH, "r") as f:
        return json.load(f)


def save_db(db):
    """Save the face database to JSON file."""
    _ensure_dir()
    with open(FACE_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def add_user(user_id, name, embedding):
    """Add a user with their face embedding."""
    db = load_db()
    db["users"][user_id] = {
        "name": name,
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        "registered_at": datetime.now().isoformat(),
    }
    save_db(db)


def get_user(user_id):
    """Get a user by ID. Returns None if not found."""
    db = load_db()
    user = db["users"].get(user_id)
    if user:
        user["user_id"] = user_id
    return user


def get_all_users():
    """Get all registered users (without embeddings for listing)."""
    db = load_db()
    users = []
    for uid, data in db["users"].items():
        users.append({
            "user_id": uid,
            "name": data["name"],
            "registered_at": data.get("registered_at", ""),
        })
    return users


def delete_user(user_id):
    """Delete a user. Returns True if deleted, False if not found."""
    db = load_db()
    if user_id in db["users"]:
        del db["users"][user_id]
        save_db(db)
        return True
    return False


def get_all_embeddings():
    """Get all users with their embeddings for matching."""
    db = load_db()
    result = []
    for uid, data in db["users"].items():
        result.append({
            "user_id": uid,
            "name": data["name"],
            "embedding": np.array(data["embedding"], dtype=np.float32),
        })
    return result
