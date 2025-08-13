from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from models.response import APIResponse
from models.development import (
    DevelopmentPhaseData, ProjectGenerationRequest, 
    ProjectGenerationResponse, GeneratedProject,
    ProjectContextOnly, SolutionsData, SolutionModificationRequest
)
from services.phase_services.development import DevelopmentService
from services.project_service import ProjectService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/development", tags=["development"])

def get_development_service() -> DevelopmentService:
    return DevelopmentService()

def get_project_service() -> ProjectService:
    return ProjectService()

@router.get("/{project_id}/context", response_model=APIResponse[ProjectContextOnly])
async def get_development_context(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    development_service: DevelopmentService = Depends(get_development_service)
):
    """Get basic development context (fast) - NO solution generation"""
    try:
        logger.info(f"Loading basic development context for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.reflection_data:
            raise HTTPException(
                status_code=400, 
                detail="Reflection phase must be completed before development"
            )
        
        if not project.scoping_data:
            raise HTTPException(
                status_code=400, 
                detail="Scoping phase must be completed before development"
            )
        
        context_data = await development_service.get_basic_context(project)
        
        logger.info(f"Successfully loaded basic context for project: {project_id}")
        
        return APIResponse(
            data=context_data,
            message="Basic development context loaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get development context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/solutions", response_model=APIResponse[SolutionsData])
async def generate_solutions(
    project_id: str,
    user_input: Dict[str, Any] = None,
    project_service: ProjectService = Depends(get_project_service),
    development_service: DevelopmentService = Depends(get_development_service)
):
    """Generate AI solutions (slow) - optionally with user input for modifications"""
    try:
        logger.info(f"Generating AI solutions for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.reflection_data or not project.scoping_data:
            raise HTTPException(
                status_code=400, 
                detail="Previous phases must be completed before generating solutions"
            )
        
        # Extract user feedback if provided
        user_feedback = None
        if user_input and "feedback" in user_input:
            user_feedback = user_input["feedback"]
            logger.info(f"User provided feedback: {user_feedback[:100]}...")
        
        solutions_data = await development_service.generate_dynamic_solutions(project, user_feedback)
        
        development_data = project.development_data or {}
        development_data.update({
            "solutions_generated": True,
            "available_solutions": [solution.dict() for solution in solutions_data.available_solutions],
            "solution_rationale": solutions_data.solution_rationale,
            "solutions_generated_at": datetime.utcnow().isoformat(),
            "user_feedback": user_feedback,
            "phase_status": "solutions_generated"
        })
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="development",
            phase_data=development_data,
            advance_phase=False
        )
        
        logger.info(f"Successfully generated {len(solutions_data.available_solutions)} solutions for project: {project_id}")
        
        return APIResponse(
            data=solutions_data,
            message=f"Generated {len(solutions_data.available_solutions)} AI solutions successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate solutions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/select-solution", response_model=APIResponse[Dict[str, Any]])
async def select_solution(
    project_id: str,
    solution_selection: Dict[str, Any],
    project_service: ProjectService = Depends(get_project_service)
):
    """Save the selected AI solution - allows re-selection for better UX"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        development_data = project.development_data or {}
        
        available_solutions = development_data.get("available_solutions", [])
        if not available_solutions:
            raise HTTPException(
                status_code=400, 
                detail="No AI solutions available. Please generate solutions first."
            )
        
        solution_id = solution_selection.get("solution_id")
        if not solution_id:
            raise HTTPException(
                status_code=400,
                detail="Solution ID is required"
            )
        
        valid_solution_ids = [sol.get("id") for sol in available_solutions if isinstance(sol, dict)]
        if solution_id not in valid_solution_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid solution ID: {solution_id}. Available solutions: {valid_solution_ids}"
            )
        
        previous_selection = development_data.get("selected_solution")
        if previous_selection:
            logger.info(f"User re-selecting solution for project {project_id}: {previous_selection.get('solution_id')} → {solution_id}")
            
            if development_data.get("generated_project") and previous_selection.get("solution_id") != solution_id:
                development_data["requires_regeneration"] = True
                development_data["previous_generations"] = development_data.get("previous_generations", [])
                development_data["previous_generations"].append({
                    "solution_id": previous_selection.get("solution_id"),
                    "solution_title": previous_selection.get("solution_title"),
                    "generated_at": development_data.get("generation_timestamp"),
                    "replaced_at": datetime.utcnow().isoformat()
                })
                
                development_data.pop("generated_project", None)
                development_data.pop("generation_timestamp", None)
                development_data["phase_status"] = "solution_selected"
        
        development_data.update({
            "selected_solution": solution_selection,
            "selection_timestamp": datetime.utcnow().isoformat(),
            "phase_status": "solution_selected",
            "selection_count": development_data.get("selection_count", 0) + 1
        })
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="development",
            phase_data=development_data,
            advance_phase=False
        )
        
        logger.info(f"Solution selected for project {project_id}: {solution_selection.get('solution_title')}")
        
        response_data = {
            "selected_solution": solution_selection,
            "can_generate": True,
            "requires_regeneration": development_data.get("requires_regeneration", False)
        }
        
        if development_data.get("requires_regeneration"):
            response_data["message"] = "Solution changed. Previous project will be replaced when you generate."
        
        return APIResponse(
            data=response_data,
            message="Solution selected successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select solution: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/{project_id}/generate", response_model=APIResponse[ProjectGenerationResponse])
async def generate_project(
    project_id: str,
    generation_request: ProjectGenerationRequest,
    project_service: ProjectService = Depends(get_project_service),
    development_service: DevelopmentService = Depends(get_development_service)
):
    """Generate complete AI project with real code validation - STORES project for evaluation phase"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.development_data or not project.development_data.get("selected_solution"):
            raise HTTPException(
                status_code=400, 
                detail="Solution must be selected before project generation"
            )
        
        logger.info(f"Generating project for {project_id} with solution {generation_request.solution_id}")
        
        generated_project = await development_service.generate_project_with_validation(project, generation_request)
        
        development_data = project.development_data or {}
        development_data.update({
            "generated_project": generated_project.dict(),
            "generation_request": generation_request.dict(),
            "generation_timestamp": datetime.utcnow().isoformat(),
            "phase_status": "project_generated",
            "completed_at": datetime.utcnow().isoformat(),
            "code_validation_passed": True
        })
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="development",
            phase_data=development_data,
            advance_phase=True
        )
        
        response = ProjectGenerationResponse(
            success=True,
            project=generated_project,
            generation_steps=[
                "Requirements analysis completed",
                "Architecture designed", 
                "Code generated and validated",
                "Ethical safeguards integrated",
                "Documentation generated",
                "Project ready for evaluation and testing"
            ],
            estimated_completion_time="Project generated successfully",
            next_steps=[
                "Proceed to evaluation phase",
                "Test the solution with your data",
                "Validate the results meet your expectations",
                "Download the complete project when satisfied"
            ]
        )
        
        return APIResponse(
            data=response,
            message="Project generated successfully - proceed to evaluation phase for testing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/reset", response_model=APIResponse[Dict[str, Any]])
async def reset_development_progress(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    try:
        logger.info(f"Resetting development progress for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="development",
            phase_data={},
            advance_phase=False
        )
        
        logger.info(f"Successfully reset development progress for project: {project_id}")
        
        return APIResponse(
            data={"reset": True, "phase": "development"},
            message="Development progress reset successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset development progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{project_id}/status", response_model=APIResponse[Dict[str, Any]])
async def get_development_status(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Get development phase status and progress"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        development_data = project.development_data or {}
        
        status_data = {
            "completed": project.development_data is not None and development_data.get("completed_at") is not None,
            "phase_status": development_data.get("phase_status", "not_started"),
            "context_loaded": project.development_data is not None,
            "solutions_generated": development_data.get("solutions_generated", False),
            "selected_solution": development_data.get("selected_solution"),
            "generated_project": development_data.get("generated_project") is not None,
            "development_data": development_data,
            "can_proceed": development_data.get("completed_at") is not None,
            "performance_metrics": {
                "context_load_time_ms": development_data.get("context_load_time_ms"),
                "solutions_generation_time_ms": development_data.get("solutions_generation_time_ms"),
                "total_load_time_ms": development_data.get("total_load_time_ms")
            }
        }
        
        return APIResponse(data=status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get development status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    