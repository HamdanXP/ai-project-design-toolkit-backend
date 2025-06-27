from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from models.response import APIResponse
from services.phase_services.reflection import ReflectionService
from services.project_service import ProjectService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reflection", tags=["reflection"])

def get_reflection_service() -> ReflectionService:
    return ReflectionService()

def get_project_service() -> ProjectService:
    return ProjectService()

@router.get("/{project_id}/questions", response_model=APIResponse[Dict[str, Any]])
async def get_reflection_questions(
    project_id: str,
    include_guidance: bool = True,  # Parameter to include guidance sources
    reflection_service: ReflectionService = Depends(get_reflection_service)
):
    """Get reflection questions for a project, optionally with guidance sources"""
    try:
        if include_guidance:
            questions_data = await reflection_service.get_or_create_reflection_questions_with_guidance(project_id)
            
            return APIResponse(
                data=questions_data,
                message="Reflection questions with guidance retrieved successfully"
            )
        else:
            # Backward compatibility - return simple questions format
            questions_dict = await reflection_service.get_or_create_reflection_questions(project_id)
            
            return APIResponse(
                data={
                    "questions": {
                        key: {"question": question, "guidance_sources": []}
                        for key, question in questions_dict.items()
                    },
                    "has_guidance": False
                },
                message="Reflection questions retrieved successfully"
            )
            
    except ValueError as e:
        # Handle "Project not found" specifically
        logger.error(f"Project {project_id} not found: {e}")
        raise HTTPException(status_code=404, detail="Project not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reflection questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/complete", response_model=APIResponse[Dict[str, Any]])
async def complete_reflection_phase(
    project_id: str,
    answers: Dict[str, str],
    project_service: ProjectService = Depends(get_project_service),
    reflection_service: ReflectionService = Depends(get_reflection_service)
):
    """Analyze responses and complete reflection phase with comprehensive project readiness assessment"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get the questions for this project
        questions = project.reflection_questions or {}
        
        # Create comprehensive project readiness assessment (combines ethical + AI appropriateness)
        assessment_result = await reflection_service.create_comprehensive_project_readiness_assessment(
            answers, project.description, questions
        )
        
        # Save reflection data with the comprehensive assessment
        reflection_data = {
            "answers": answers,
            "questions": questions,  # Store questions used
            "ethical_score": assessment_result["ethical_score"],
            "ai_appropriateness_score": assessment_result["ai_appropriateness_score"],
            "overall_readiness_score": assessment_result["overall_readiness_score"],
            "summary": assessment_result["summary"],
            "actionable_recommendations": assessment_result["actionable_recommendations"],
            "question_flags": assessment_result["question_flags"],
            "completed_at": datetime.utcnow().isoformat(),
        }
        
        # Update project with both assessments
        await project_service.update_project_phase(
            project_id=project_id,
            phase="reflection",
            phase_data=reflection_data,
            project_readiness_assessment=assessment_result["project_readiness_assessment"],
            ethical_assessment=assessment_result["ethical_assessment"],  # Keep for backward compatibility
            advance_phase=False
        )
        
        # Return the comprehensive assessment data for frontend
        return APIResponse(
            data={
                # Overall assessment
                "overall_readiness_score": assessment_result["overall_readiness_score"],
                "proceed_recommendation": assessment_result["proceed_recommendation"],
                "summary": assessment_result["summary"],
                "actionable_recommendations": assessment_result["actionable_recommendations"],
                "question_flags": assessment_result["question_flags"],
                "threshold_met": assessment_result["threshold_met"],
                "can_proceed": assessment_result["threshold_met"],
                
                # Ethical assessment details
                "ethical_score": assessment_result["ethical_score"],
                "ethical_summary": assessment_result.get("ethical_summary", ""),
                
                # AI appropriateness assessment details
                "ai_appropriateness_score": assessment_result["ai_appropriateness_score"],
                "ai_appropriateness_summary": assessment_result["ai_appropriateness_summary"],
                "ai_recommendation": assessment_result["ai_recommendation"],
                "alternative_solutions": assessment_result.get("alternative_solutions"),
            },
            message="Project readiness assessment completed"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete reflection phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{project_id}/advance")
async def advance_to_next_phase(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Advance to next phase after user confirms"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.reflection_data:
            raise HTTPException(status_code=400, detail="Reflection phase not completed")
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="reflection",
            phase_data=project.reflection_data,
            advance_phase=True
        )
        
        return APIResponse(
            message="Advanced to scoping phase",
            data={"next_phase": "scoping"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to advance phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/status", response_model=APIResponse[Dict[str, Any]])
async def get_reflection_status(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Get reflection phase status and results"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        status_data = {
            "completed": project.reflection_data is not None,
            "ethical_assessment": project.ethical_assessment.dict() if project.ethical_assessment else None,
            "can_proceed": project.ethical_assessment.approved if project.ethical_assessment else False,
            "reflection_data": project.reflection_data,
            "questions": project.reflection_questions
        }
        
        return APIResponse(data=status_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reflection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))