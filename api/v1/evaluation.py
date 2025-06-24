from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from models.response import APIResponse
from models.project import EvaluationResults
from models.phase import EvaluationPlan
from services.phase_services.evaluation import EvaluationService
from services.project_service import ProjectService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

def get_evaluation_service() -> EvaluationService:
    return EvaluationService()

def get_project_service() -> ProjectService:
    return ProjectService()

@router.get("/{project_id}/plan", response_model=APIResponse[EvaluationPlan])
async def get_evaluation_plan(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    """Get evaluation plan for the project"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.development_data:
            raise HTTPException(status_code=400, detail="Development phase not completed")
        
        # Extract model configuration
        model_config_data = project.development_data.get("model_configuration", {})
        model_config = None
        
        # Get target population from project context
        target_population = "humanitarian beneficiaries"  # Could be extracted from project data
        
        plan = await evaluation_service.create_evaluation_plan(
            project.description,
            model_config,
            target_population
        )
        
        return APIResponse(
            data=plan,
            message="Evaluation plan created"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/run", response_model=APIResponse[EvaluationResults])
async def run_evaluation(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    """Run evaluation simulation"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.development_data:
            raise HTTPException(status_code=400, detail="Development phase not completed")
        
        # Create evaluation plan
        model_config_data = project.development_data.get("model_configuration", {})
        model_config = None
        
        plan = await evaluation_service.create_evaluation_plan(
            project.description, model_config, "humanitarian beneficiaries"
        )
        
        # Run simulation
        results = await evaluation_service.run_simulation(plan, model_config)
        
        # Update project with evaluation results
        evaluation_data = {
            "evaluation_plan": plan.dict(),
            "results": results.dict(),
            "completed_at": "2024-01-01T00:00:00Z"
        }
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="evaluation",
            phase_data=evaluation_data,
            advance_phase=results.ready_for_deployment
        )
        
        return APIResponse(
            data=results,
            message="Evaluation completed"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
