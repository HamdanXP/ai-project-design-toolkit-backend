from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from models.response import APIResponse
from models.evaluation import (
    EvaluationContext, SimulationRequest, SimulationResult, 
    EvaluationResult, EvaluationStatusData, TestingScenario,
    ScenarioRegenerationRequest, TestingMethod
)
from services.phase_services.evaluation import EvaluationService
from services.project_service import ProjectService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

def get_evaluation_service() -> EvaluationService:
    return EvaluationService()

def get_project_service() -> ProjectService:
    return ProjectService()

@router.get("/{project_id}/context", response_model=APIResponse[EvaluationContext])
async def get_evaluation_context(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    try:
        logger.info(f"Loading evaluation context for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise HTTPException(
                status_code=400,
                detail="Development phase must be completed before evaluation"
            )
        
        context_data = await evaluation_service.get_evaluation_context(project)
        
        logger.info(f"Successfully loaded evaluation context for project: {project_id}")
        
        return APIResponse(
            data=context_data,
            message="Evaluation context loaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/simulate", response_model=APIResponse[SimulationResult])
async def simulate_project(
    project_id: str,
    simulation_request: SimulationRequest,
    project_service: ProjectService = Depends(get_project_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    try:
        logger.info(f"Running simulation for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise HTTPException(
                status_code=400,
                detail="No generated project found for simulation"
            )
        
        if simulation_request.simulation_type in ["statistics_based", "suitability_assessment"]:
            simulation_results = await evaluation_service.simulate_with_dataset_stats(
                project, simulation_request.dataset_statistics
            )
        else:
            simulation_results = await evaluation_service.simulate_without_dataset(
                project, simulation_request.custom_scenarios
            )
        
        evaluation_data = project.evaluation_data or {}
        evaluation_data.update({
            "simulation_results": simulation_results.dict(),
            "simulation_timestamp": datetime.utcnow().isoformat(),
            "testing_method": simulation_results.testing_method.value,
            "phase_status": "simulation_completed"
        })
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="evaluation",
            phase_data=evaluation_data,
            advance_phase=False
        )
        
        logger.info(f"Successfully completed simulation for project: {project_id}")
        
        return APIResponse(
            data=simulation_results,
            message="Simulation completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/regenerate-scenarios", response_model=APIResponse[List[TestingScenario]])
async def regenerate_scenarios(
    project_id: str,
    request: ScenarioRegenerationRequest,
    project_service: ProjectService = Depends(get_project_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    try:
        logger.info(f"Regenerating scenarios for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        new_scenarios = await evaluation_service.regenerate_scenarios(project, request)
        
        logger.info(f"Successfully regenerated {len(new_scenarios)} scenarios for project: {project_id}")
        
        return APIResponse(
            data=new_scenarios,
            message=f"Regenerated {len(new_scenarios)} scenarios successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to regenerate scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/evaluate", response_model=APIResponse[EvaluationResult])
async def evaluate_results(
    project_id: str,
    evaluation_data: Dict[str, Any],
    project_service: ProjectService = Depends(get_project_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    try:
        logger.info(f"Evaluating results for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        simulation_result_data = evaluation_data.get("simulation_result")
        if not simulation_result_data:
            raise HTTPException(status_code=400, detail="Simulation result is required")
        
        simulation_result = SimulationResult(**simulation_result_data)
        
        evaluation_result = await evaluation_service.evaluate_results(project, simulation_result)
        
        project_evaluation_data = project.evaluation_data or {}
        project_evaluation_data.update({
            "evaluation_result": evaluation_result.dict(),
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "phase_status": "evaluation_completed",
            "completed_at": datetime.utcnow().isoformat()
        })
        
        advance_phase = evaluation_result.status in ["ready_for_deployment", "needs_minor_improvements"]
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="evaluation",
            phase_data=project_evaluation_data,
            advance_phase=advance_phase
        )
        
        logger.info(f"Successfully evaluated results for project: {project_id}")
        
        return APIResponse(
            data=evaluation_result,
            message="Evaluation completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/download/{file_type}")
async def download_project_file(
    project_id: str,
    file_type: str,
    project_service: ProjectService = Depends(get_project_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    try:
        logger.info(f"Downloading {file_type} for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise HTTPException(status_code=404, detail="No generated project found")
        
        download_files = await evaluation_service.get_download_files(project)
        
        file_handlers = {
            "complete_project": lambda: {
                "message": "Complete project download",
                "files": download_files.complete_project.files,
                "description": download_files.complete_project.description
            },
            "documentation": lambda: {
                "content": download_files.documentation.content,
                "description": download_files.documentation.description
            },
            "setup_instructions": lambda: {
                "content": download_files.setup_instructions.content,
                "description": download_files.setup_instructions.description
            },
            "deployment_guide": lambda: {
                "content": download_files.deployment_guide.content,
                "description": download_files.deployment_guide.description
            },
            "ethical_assessment_guide": lambda: {
                "content": download_files.ethical_assessment_guide.content,
                "description": download_files.ethical_assessment_guide.description
            },
            "technical_handover_package": lambda: {
                "content": download_files.technical_handover_package.content,
                "description": download_files.technical_handover_package.description
            }
        }
        
        if file_type in file_handlers:
            return file_handlers[file_type]()
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"File type '{file_type}' not found. Available types: {list(file_handlers.keys())}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download project file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/status", response_model=APIResponse[EvaluationStatusData])
async def get_evaluation_status(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        evaluation_data = project.evaluation_data or {}
        
        testing_method_value = evaluation_data.get("testing_method")
        testing_method = TestingMethod(testing_method_value) if testing_method_value else None
        
        if not testing_method and evaluation_data.get("simulation_results"):
            sim_results = evaluation_data["simulation_results"]
            testing_method_value = sim_results.get("testing_method")
            testing_method = TestingMethod(testing_method_value) if testing_method_value else None
        
        if testing_method:
            status_data = EvaluationStatusData(
                completed=evaluation_data.get("completed_at") is not None,
                phase_status=evaluation_data.get("phase_status", "not_started"),
                has_simulation=evaluation_data.get("simulation_results") is not None,
                has_evaluation=evaluation_data.get("evaluation_result") is not None,
                testing_method=testing_method,
                evaluation_result=evaluation_data.get("evaluation_result"),
                can_download=evaluation_data.get("evaluation_result", {}).get("status") in [
                    "ready_for_deployment", "needs_minor_improvements"
                ],
                evaluation_data=evaluation_data
            )
        else:
            raise HTTPException(status_code=400, detail="No testing method found for this project")
        
        return APIResponse(data=status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/{project_id}/reset", response_model=APIResponse[Dict[str, Any]])
async def reset_evaluation_progress(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    try:
        logger.info(f"Resetting evaluation progress for project: {project_id}")
        
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="evaluation",
            phase_data={},
            advance_phase=False
        )
        
        logger.info(f"Successfully reset evaluation progress for project: {project_id}")
        
        return APIResponse(
            data={"reset": True, "phase": "evaluation"},
            message="Evaluation progress reset successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset evaluation progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))