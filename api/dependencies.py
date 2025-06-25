from fastapi import Depends, HTTPException, Request
from typing import Optional
from services.project_service import ProjectService
from services.llm_analyzer import LLMAnalyzer
from services.document_service import DocumentService
from services.datasets.management_service import DatasetService
from services.phase_services.reflection import ReflectionService
from services.phase_services.scoping import ScopingService
from services.phase_services.development import DevelopmentService
from services.phase_services.evaluation import EvaluationService
from models.project import Project
from core.exceptions import ProjectNotFoundError, create_http_exception
import logging

logger = logging.getLogger(__name__)

# Service Dependencies
def get_project_service() -> ProjectService:
    """Get project service instance"""
    return ProjectService()

def get_llm_analyzer() -> LLMAnalyzer:
    """Get LLM analyzer service instance"""
    return LLMAnalyzer()

def get_document_service() -> DocumentService:
    """Get document service instance"""
    return DocumentService()

def get_dataset_service() -> DatasetService:
    """Get dataset service instance"""
    return DatasetService()

def get_reflection_service() -> ReflectionService:
    """Get reflection service instance"""
    return ReflectionService()

def get_scoping_service() -> ScopingService:
    """Get scoping service instance"""
    return ScopingService()

def get_development_service() -> DevelopmentService:
    """Get development service instance"""
    return DevelopmentService()

def get_evaluation_service() -> EvaluationService:
    """Get evaluation service instance"""
    return EvaluationService()

# Project Dependencies
async def get_project_by_id(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
) -> Project:
    """Get project by ID with error handling"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise create_http_exception(404, f"Project {project_id} not found")
        return project
    except Exception as e:
        logger.error(f"Failed to get project {project_id}: {e}")
        raise create_http_exception(500, f"Error retrieving project: {str(e)}")

async def get_project_with_phase(
    project_id: str,
    required_phase: str,
    project_service: ProjectService = Depends(get_project_service)
) -> Project:
    """Get project and validate it has completed required phase"""
    project = await get_project_by_id(project_id, project_service)
    
    phase_data_map = {
        "reflection": project.reflection_data,
        "scoping": project.scoping_data, 
        "development": project.development_data,
        "evaluation": project.evaluation_data
    }
    
    if required_phase in phase_data_map and phase_data_map[required_phase] is None:
        raise create_http_exception(
            400, 
            f"Project must complete {required_phase} phase first"
        )
    
    return project

# Validation Dependencies
def validate_file_upload(
    file_size: int,
    content_type: str,
    max_size: int = 50 * 1024 * 1024  # 50MB
) -> bool:
    """Validate uploaded file"""
    if file_size > max_size:
        raise create_http_exception(413, "File too large")
    
    allowed_types = [
        "application/pdf",
        "text/plain", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]
    
    if content_type not in allowed_types:
        raise create_http_exception(415, "Unsupported file type")
    
    return True

# Rate Limiting Dependencies (placeholder for future implementation)
async def rate_limit_check(request: Request) -> bool:
    """Check rate limiting (placeholder)"""
    # TODO: Implement actual rate limiting logic
    return True
