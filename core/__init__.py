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
from typing import Optional
from services.rag_service import RAGService

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
    "create_http_exception",
    "rag_service",
]

rag_service: Optional[RAGService] = None
