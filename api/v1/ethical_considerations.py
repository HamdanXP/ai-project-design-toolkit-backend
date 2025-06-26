from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from models.response import APIResponse
from services.project_service import ProjectService
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ethical-considerations", tags=["ethical-considerations"])

def get_project_service() -> ProjectService:
    return ProjectService()

class AcknowledgeRequest(BaseModel):
    acknowledged_considerations: Optional[List[str]] = None

@router.get("/{project_id}", response_model=APIResponse[List[Dict[str, Any]]])
async def get_ethical_considerations(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Get ethical considerations for a project"""
    try:
        considerations = await project_service.get_ethical_considerations(project_id)
        
        return APIResponse(
            data=considerations,
            message="Ethical considerations retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Failed to get ethical considerations for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/acknowledge", response_model=APIResponse[Dict[str, bool]])
async def acknowledge_ethical_considerations(
    project_id: str,
    request: AcknowledgeRequest,
    project_service: ProjectService = Depends(get_project_service)
):
    """Mark ethical considerations as acknowledged by user"""
    try:
        success = await project_service.acknowledge_ethical_considerations(
            project_id, 
            request.acknowledged_considerations
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return APIResponse(
            data={"acknowledged": True},
            message="Ethical considerations acknowledged successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge ethical considerations for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/refresh", response_model=APIResponse[List[Dict[str, Any]]])
async def refresh_ethical_considerations(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Refresh ethical considerations for a project (fetch latest from RAG)"""
    try:
        considerations = await project_service.refresh_ethical_considerations(project_id)
        
        return APIResponse(
            data=considerations,
            message="Ethical considerations refreshed successfully"
        )
    except Exception as e:
        logger.error(f"Failed to refresh ethical considerations for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

