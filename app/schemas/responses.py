"""Pydantic response schemas."""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class FaceDetection(BaseModel):
    bbox: List[float]
    confidence: float
    landmarks: Optional[List[List[float]]] = None


class DetectResponse(BaseModel):
    total_faces: int
    processing_time: float
    faces: List[FaceDetection]


class QualityScores(BaseModel):
    blur: float
    brightness: float
    face_size: Dict[str, float]
    angle: Optional[float] = None


class QualityResult(BaseModel):
    passed: bool
    issues: List[str]
    scores: QualityScores


class QualityCheckResponse(BaseModel):
    total_faces: int
    results: List[Dict[str, Any]]


class AntiSpoofResponse(BaseModel):
    is_real: bool
    score: float
    details: Dict[str, Any]


class RegisterResponse(BaseModel):
    status: str
    user_id: Optional[str] = None
    name: Optional[str] = None
    images_processed: Optional[int] = None
    images_failed: Optional[int] = None
    errors: List[str] = []
    message: Optional[str] = None


class VerifyResponse(BaseModel):
    match: bool
    similarity: Optional[float] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    threshold: Optional[float] = None
    error: Optional[str] = None


class IdentifyCandidate(BaseModel):
    user_id: str
    name: str
    similarity: float


class IdentifyResponse(BaseModel):
    total_candidates: int
    candidates: List[IdentifyCandidate]
    threshold: float
    error: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    name: str
    registered_at: str


class UsersListResponse(BaseModel):
    total_users: int
    users: List[UserInfo]


class HealthResponse(BaseModel):
    status: str
    version: str
    modules: Dict[str, str]
