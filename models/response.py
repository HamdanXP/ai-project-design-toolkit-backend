from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Generic, TypeVar

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "Success"
    data: Optional[T] = None
    errors: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    errors: List[str] = []
    details: Optional[Dict[str, Any]] = None

class ProjectCreateRequest(BaseModel):
    description: str

class ProjectResponse(BaseModel):
    id: str
    title: str
    description: str
    status: str
    current_phase: str
    created_at: str
    updated_at: str

class PhaseProgressResponse(BaseModel):
    phase: str
    completed: bool
    data: Optional[Dict[str, Any]] = None
    next_phase: Optional[str] = None