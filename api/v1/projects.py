from fastapi import APIRouter, HTTPException, Depends
from typing import Any, Dict, List, Optional
from models.project import Project, ProjectStatus
from models.response import (
    APIResponse, ProjectCreateRequest, ProjectResponse, 
    PhaseProgressResponse
)
from services.project_service import ProjectService
from core.llm_service import llm_service
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"])

def get_project_service() -> ProjectService:
    return ProjectService()

@router.post("/", response_model=APIResponse[ProjectResponse])
async def create_project(
    request: ProjectCreateRequest,
    project_service: ProjectService = Depends(get_project_service)
):
    """Create new project from description"""
    try:
        # Extract project info using LLM
        project_info = await llm_service.extract_project_info(request.description)
        
        # Create project
        project = await project_service.create_project(
            description=request.description,
            title=project_info.get("title", "Humanitarian AI Project"),
            context=project_info.get("context"),
            tags=project_info.get("tags", [])
        )
        
        project_response = ProjectResponse(
            id=str(project.id),
            title=project.title,
            description=project.description,
            status=project.status,
            current_phase=project.current_phase,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat()
        )
        
        return APIResponse(
            data=project_response,
            message="Project created successfully"
        )
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}", response_model=APIResponse[ProjectResponse])
async def get_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Get project by ID"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_response = ProjectResponse(
            id=str(project.id),
            title=project.title,
            description=project.description,
            status=project.status,
            current_phase=project.current_phase,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat()
        )
        
        return APIResponse(data=project_response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=APIResponse[List[ProjectResponse]])
async def list_projects(
    skip: int = 0,
    limit: int = 10,
    status: Optional[str] = None,
    project_service: ProjectService = Depends(get_project_service)
):
    """List projects with pagination and filtering"""
    try:
        projects = await project_service.list_projects(
            skip=skip, 
            limit=limit, 
            status=status
        )
        
        project_responses = [
            ProjectResponse(
                id=str(project.id),
                title=project.title,
                description=project.description,
                status=project.status,
                current_phase=project.current_phase,
                created_at=project.created_at.isoformat(),
                updated_at=project.updated_at.isoformat()
            )
            for project in projects
        ]
        
        return APIResponse(data=project_responses)
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Delete project"""
    try:
        success = await project_service.delete_project(project_id)
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return APIResponse(message="Project deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/status", response_model=APIResponse[PhaseProgressResponse])
async def get_project_status(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Get detailed project status and phase progress"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Determine next phase
        phase_order = ["reflection", "scoping", "development", "evaluation", "completed"]
        current_index = phase_order.index(project.current_phase)
        next_phase = phase_order[current_index + 1] if current_index < len(phase_order) - 1 else None
        
        # Get phase-specific data
        phase_data = None
        if project.current_phase == "reflection":
            phase_data = project.reflection_data
        elif project.current_phase == "scoping":
            phase_data = project.scoping_data
        elif project.current_phase == "development":
            phase_data = project.development_data
        elif project.current_phase == "evaluation":
            phase_data = project.evaluation_data
        
        response = PhaseProgressResponse(
            phase=project.current_phase,
            completed=project.status != ProjectStatus.CREATED,
            data=phase_data,
            next_phase=next_phase
        )
        
        return APIResponse(data=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/sync", response_model=APIResponse[Dict[str, Any]])
async def get_project_sync_data(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Get project data with sync metadata for frontend"""
    try:
        project_data = await project_service.get_project_with_metadata(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return APIResponse(data=project_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project sync data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
