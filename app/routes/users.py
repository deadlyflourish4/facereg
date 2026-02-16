"""User management routes: /users"""
from fastapi import APIRouter
from app.services.storage import get_all_users, delete_user

router = APIRouter(tags=["Users"])


@router.get("/users")
async def list_users():
    """List all registered users."""
    users = get_all_users()
    return {"total_users": len(users), "users": users}


@router.delete("/users/{user_id}")
async def remove_user(user_id: str):
    """Delete a registered user."""
    deleted = delete_user(user_id)
    if deleted:
        return {"status": "deleted", "user_id": user_id}
    return {"status": "not_found", "user_id": user_id}
