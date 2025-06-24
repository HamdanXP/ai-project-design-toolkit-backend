# Updated development_routes.py - Split context and solutions endpoints

from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from models.response import APIResponse
from models.development import (
    DevelopmentPhaseData, ProjectGenerationRequest, 
    ProjectGenerationResponse, GeneratedProject,
    ProjectContextOnly, SolutionsData
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
        
        # Get project with all necessary data
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if previous phases are completed
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
        
        # Generate ONLY basic context (fast operation)
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
    project_service: ProjectService = Depends(get_project_service),
    development_service: DevelopmentService = Depends(get_development_service)
):
    """Generate AI solutions (slow) - called when user navigates to solutions step"""
    try:
        logger.info(f"Generating AI solutions for project: {project_id}")
        
        # Get project with all necessary data
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if context phase is completed
        if not project.reflection_data or not project.scoping_data:
            raise HTTPException(
                status_code=400, 
                detail="Previous phases must be completed before generating solutions"
            )
        
        # Generate AI solutions (intensive operation)
        solutions_data = await development_service.generate_solutions(project)
        
        # Cache the generated solutions in project data for future retrieval
        development_data = project.development_data or {}
        development_data.update({
            "solutions_generated": True,
            "available_solutions": [solution.dict() for solution in solutions_data.available_solutions],
            "solution_rationale": solutions_data.solution_rationale,
            "solutions_generated_at": datetime.utcnow().isoformat(),
            "phase_status": "solutions_generated"
        })
        
        # Save to project (don't advance phase yet)
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
        
        # IMPROVED: Check what actually matters for solution selection
        development_data = project.development_data or {}
        
        # 1. Verify solutions exist (either generated or cached)
        available_solutions = development_data.get("available_solutions", [])
        if not available_solutions:
            raise HTTPException(
                status_code=400, 
                detail="No AI solutions available. Please generate solutions first."
            )
        
        # 2. Validate the selected solution exists
        solution_id = solution_selection.get("solution_id")
        if not solution_id:
            raise HTTPException(
                status_code=400,
                detail="Solution ID is required"
            )
        
        # Check if the solution ID exists in available solutions
        valid_solution_ids = [sol.get("id") for sol in available_solutions if isinstance(sol, dict)]
        if solution_id not in valid_solution_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid solution ID: {solution_id}. Available solutions: {valid_solution_ids}"
            )
        
        # 3. Handle re-selection gracefully
        previous_selection = development_data.get("selected_solution")
        if previous_selection:
            logger.info(f"User re-selecting solution for project {project_id}: {previous_selection.get('solution_id')} → {solution_id}")
            
            # If they had a generated project and are changing solutions, mark for regeneration
            if development_data.get("generated_project") and previous_selection.get("solution_id") != solution_id:
                development_data["requires_regeneration"] = True
                development_data["previous_generations"] = development_data.get("previous_generations", [])
                development_data["previous_generations"].append({
                    "solution_id": previous_selection.get("solution_id"),
                    "solution_title": previous_selection.get("solution_title"),
                    "generated_at": development_data.get("generation_timestamp"),
                    "replaced_at": datetime.utcnow().isoformat()
                })
                
                # Clear the old generated project since they're selecting a new solution
                development_data.pop("generated_project", None)
                development_data.pop("generation_timestamp", None)
                development_data["phase_status"] = "solution_selected"  # Reset to solution selected
        
        # 4. Update with new selection
        development_data.update({
            "selected_solution": solution_selection,
            "selection_timestamp": datetime.utcnow().isoformat(),
            "phase_status": "solution_selected",
            "selection_count": development_data.get("selection_count", 0) + 1
        })
        
        # Save to project
        await project_service.update_project_phase(
            project_id=project_id,
            phase="development",
            phase_data=development_data,
            advance_phase=False  # Don't advance phase, just update selection
        )
        
        logger.info(f"Solution selected for project {project_id}: {solution_selection.get('solution_title')}")
        
        # Provide helpful response
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
    """Generate complete AI project based on selected solution"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if solution is selected
        if not project.development_data or not project.development_data.get("selected_solution"):
            raise HTTPException(
                status_code=400, 
                detail="Solution must be selected before project generation"
            )
        
        logger.info(f"Generating project for {project_id} with solution {generation_request.solution_id}")
        
        # Generate the project
        generated_project = await development_service.generate_project(project, generation_request)
        
        # Update project with generation results
        development_data = project.development_data or {}
        development_data.update({
            "generated_project": generated_project.dict(),
            "generation_request": generation_request.dict(),
            "generation_timestamp": datetime.utcnow().isoformat(),
            "phase_status": "project_generated",
            "completed_at": datetime.utcnow().isoformat()
        })
        
        # Save to project and advance phase
        await project_service.update_project_phase(
            project_id=project_id,
            phase="development",
            phase_data=development_data,
            advance_phase=True  # Move to evaluation phase
        )
        
        response = ProjectGenerationResponse(
            success=True,
            project=generated_project,
            generation_steps=[
                "Project structure created",
                "Frontend code generated",
                "Backend API implemented",
                "Ethical safeguards integrated",
                "Documentation generated",
                "Deployment scripts created"
            ],
            estimated_completion_time="Project generated successfully",
            next_steps=[
                "Download the complete project",
                "Review the ethical audit report",
                "Follow the setup instructions",
                "Proceed to evaluation phase for testing"
            ]
        )
        
        return APIResponse(
            data=response,
            message="Project generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/download/{file_type}")
async def download_project_file(
    project_id: str,
    file_type: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Download specific files from the generated project"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise HTTPException(status_code=404, detail="No generated project found")
        
        generated_project_data = project.development_data["generated_project"]
        
        if file_type == "complete":
            return {"message": "Complete project download", "files": generated_project_data.get("files", {})}
        elif file_type == "documentation":
            return {"content": generated_project_data.get("documentation", "")}
        elif file_type == "setup":
            return {"content": generated_project_data.get("setup_instructions", "")}
        elif file_type == "ethical-report":
            return {"content": generated_project_data.get("ethical_audit_report", "")}
        elif file_type == "deployment":
            return {"content": generated_project_data.get("deployment_guide", "")}
        else:
            raise HTTPException(status_code=404, detail="File type not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download project file: {e}")
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
            "solutions_generated": development_data.get("solutions_generated", False),
            "selected_solution": development_data.get("selected_solution"),
            "generated_project": development_data.get("generated_project") is not None,
            "development_data": development_data,
            "can_proceed": development_data.get("completed_at") is not None
        }
        
        return APIResponse(data=status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get development status: {e}")
        raise HTTPException(status_code=500, detail=str(e))