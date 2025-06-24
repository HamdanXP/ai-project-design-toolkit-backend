"""
Core functionality for AI Toolkit
"""

from .database import connect_to_mongo, close_mongo_connection, database_manager
from .llm_service import llm_service, LLMService
from .exceptions import (
    ToolkitException,
    ProjectNotFoundError,
    PhaseValidationError,
    EthicalAssessmentError,
    DataProcessingError,
    create_http_exception
)

__all__ = [
    "connect_to_mongo",
    "close_mongo_connection", 
    "database_manager",
    "llm_service",
    "LLMService",
    "ToolkitException",
    "ProjectNotFoundError",
    "PhaseValidationError",
    "EthicalAssessmentError",
    "DataProcessingError",
    "create_http_exception"
]