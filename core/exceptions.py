from fastapi import HTTPException
from typing import Optional, Any, Dict

class ToolkitException(Exception):
    """Base exception for toolkit-specific errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ProjectNotFoundError(ToolkitException):
    """Raised when a project is not found"""
    pass

class PhaseValidationError(ToolkitException):
    """Raised when phase validation fails"""
    pass

class EthicalAssessmentError(ToolkitException):
    """Raised when ethical assessment fails"""
    pass

class DataProcessingError(ToolkitException):
    """Raised when data processing fails"""
    pass

def create_http_exception(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create standardized HTTP exception"""
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "details": details or {}
        }
    )