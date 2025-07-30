from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from config import settings
from models.response import APIResponse
from models.project import UseCase, Dataset, DeploymentEnvironment, DataSuitabilityAssessment
from models.phase import ScopingRequest, ScopingResponse
from models.scoping import (
    ScopingCompletionRequest, 
    ScopingCompletionResponse, 
    ProjectReadinessDecision,
    TechnicalInfrastructure,
    InfrastructureAssessment
)
from services.phase_services.scoping import ScopingService
from services.project_service import ProjectService
from services.infrastructure_assessment_service import InfrastructureAssessmentService
from api.dependencies import get_project_service, get_scoping_service
from utils.session_manager import session_manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scoping", tags=["scoping"])

def get_infrastructure_service() -> InfrastructureAssessmentService:
    return InfrastructureAssessmentService()

@router.post("/{project_id}/use-cases", response_model=APIResponse[List[Dict[str, Any]]])
async def get_similar_use_cases(
    project_id: str,
    request_data: Dict[str, Any] = None,
    project_service: ProjectService = Depends(get_project_service),
    scoping_service: ScopingService = Depends(get_scoping_service)
):
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        problem_domain = project.problem_domain or ""
        
        technical_infrastructure = None
        if request_data and "technical_infrastructure" in request_data:
            technical_infrastructure = request_data["technical_infrastructure"]
        
        try:
            use_cases = await scoping_service.get_educational_use_cases(
                project.description, 
                problem_domain,
                technical_infrastructure
            )
            
            if use_cases:
                return APIResponse(
                    success=True,
                    data=use_cases,
                    message=f"Found {len(use_cases)} relevant AI use cases from academic sources"
                )
            else:
                return APIResponse(
                    success=True,
                    data=[],
                    message="No suitable AI use cases found. You may proceed with general AI principles."
                )
            
        except Exception as e:
            logger.error(f"AI use case retrieval failed: {e}")
            return APIResponse(
                success=True,
                data=[],
                message="Unable to retrieve AI use cases. The search service may be temporarily unavailable."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_similar_use_cases: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        try:
            await session_manager.close_all_sessions()
        except Exception as cleanup_error:
            logger.warning(f"Session cleanup error: {cleanup_error}")


@router.post("/{project_id}/datasets", response_model=APIResponse[List[Dataset]])
async def get_recommended_datasets(
    project_id: str,
    request_data: Dict[str, Any],
    project_service: ProjectService = Depends(get_project_service),
    scoping_service: ScopingService = Depends(get_scoping_service)
):
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        problem_domain = project.problem_domain or ""
        
        use_case_id = request_data.get("use_case_id", "")
        use_case_title = request_data.get("use_case_title", "")
        use_case_description = request_data.get("use_case_description", "")
        
        try:
            datasets = await scoping_service.recommend_datasets(
                project.description,
                use_case_title,
                use_case_description,
                problem_domain
            )
            
            if datasets:
                domain_context = f" for {problem_domain}" if problem_domain else ""
                return APIResponse(
                    success=True,
                    data=datasets,
                    message=f"Found {len(datasets)} relevant datasets from humanitarian sources{domain_context}"
                )
            else:
                domain_context = f" in {problem_domain}" if problem_domain else ""
                return APIResponse(
                    success=True,
                    data=[],
                    message=f"No datasets found from humanitarian data sources{domain_context}. This is common for specialized AI projects. Consider collecting your own data or partnering with organizations that have relevant datasets."
                )
                
        except Exception as e:
            logger.error(f"Dataset retrieval failed: {e}")
            domain_context = f" for {problem_domain}" if problem_domain else ""
            return APIResponse(
                success=True,
                data=[],
                message=f"Dataset search failed{domain_context}. The data discovery service may be temporarily unavailable."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommended datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            await session_manager.close_all_sessions()
        except Exception as cleanup_error:
            logger.warning(f"Session cleanup error: {cleanup_error}")

@router.post("/{project_id}/assess-infrastructure", response_model=APIResponse[InfrastructureAssessment])
async def assess_infrastructure(
    project_id: str,
    infrastructure: TechnicalInfrastructure,
    project_service: ProjectService = Depends(get_project_service),
    infrastructure_service: InfrastructureAssessmentService = Depends(get_infrastructure_service)
):
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        valid_options = {
            'computing_resources': [
                'cloud_platforms', 'organizational_computers', 'partner_shared', 
                'community_shared', 'mobile_devices', 'basic_hardware', 'no_computing'
            ],
            'storage_data': [
                'secure_cloud', 'organizational_servers', 'partner_systems', 
                'government_systems', 'basic_digital', 'paper_based', 'local_storage'
            ],
            'internet_connectivity': [
                'stable_broadband', 'satellite_internet', 'intermittent_connection', 
                'mobile_data_primary', 'shared_community', 'limited_connectivity', 'no_internet'
            ],
            'deployment_environment': [
                'cloud_deployment', 'hybrid_approach', 'organizational_infrastructure', 
                'partner_infrastructure', 'field_mobile', 'offline_systems', 'no_deployment'
            ]
        }
        
        for field, value in infrastructure.dict().items():
            if value not in valid_options.get(field, []):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid value '{value}' for {field}. Valid options: {valid_options.get(field, [])}"
                )
        
        assessment = await infrastructure_service.assess_infrastructure(
            project.description,
            project.problem_domain or "general_humanitarian",
            infrastructure
        )
        
        return APIResponse(
            success=True,
            data=assessment,
            message=f"Infrastructure assessment completed with {assessment.score}% readiness score"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assess infrastructure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            await session_manager.close_all_sessions()
        except Exception as cleanup_error:
            logger.warning(f"Session cleanup error: {cleanup_error}")


@router.post("/{project_id}/complete", response_model=APIResponse[Dict[str, Any]])
async def complete_scoping_phase(
    project_id: str,
    scoping_data: ScopingCompletionRequest,
    project_service: ProjectService = Depends(get_project_service)
):
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        deployment_environment = DeploymentEnvironment(
            computing_resources=scoping_data.technical_infrastructure.computing_resources,
            reliable_internet_connection=scoping_data.technical_infrastructure.internet_connectivity in [
                "stable_broadband", "satellite_internet", "intermittent_connection"
            ],
            local_technology_setup=scoping_data.technical_infrastructure.deployment_environment in [
                "organizational_infrastructure", "hybrid_approach", "offline_systems"
            ],
            
            project_budget="not_specified",
            project_timeline="not_specified", 
            team_size="not_specified",
            ai_ml_experience="not_specified",
            technical_skills="not_specified",
            learning_training_capacity=None,
            stakeholder_buy_in="not_specified",
            change_management_readiness=None,
            data_governance="not_specified",
            regulatory_requirements="not_specified",
            external_partnerships=None,
            long_term_sustainability_plan=None
        )
        
        data_suitability_assessment = None
        if scoping_data.suitability_checks:
            suitability_dict = {c.get("id"): c.get("answer") for c in scoping_data.suitability_checks}
            
            def map_answer(answer: str, question_id: str) -> str:
                if question_id in ["data_completeness", "completeness"]:
                    if answer == "yes":
                        return "looks_clean"
                    elif answer == "unknown":
                        return "some_issues"
                    else:
                        return "many_problems"
                elif question_id in ["population_representativeness", "representativeness"]:
                    if answer == "yes":
                        return "representative"
                    elif answer == "unknown":
                        return "partially"
                    else:
                        return "limited_coverage"
                elif question_id in ["privacy_ethics", "privacy"]:
                    if answer == "yes":
                        return "privacy_safe"
                    elif answer == "unknown":
                        return "need_review"
                    else:
                        return "high_risk"
                elif question_id in ["quality_sufficiency", "sufficiency"]:
                    if answer == "yes":
                        return "sufficient"
                    elif answer == "unknown":
                        return "borderline"
                    else:
                        return "insufficient"
                else:
                    if answer == "yes":
                        return "looks_clean"
                    elif answer == "unknown":
                        return "some_issues"
                    else:
                        return "many_problems"
            
            try:
                data_suitability_assessment = DataSuitabilityAssessment(
                    data_completeness=map_answer(
                        suitability_dict.get("data_completeness", suitability_dict.get("completeness", "unknown")),
                        "data_completeness"
                    ),
                    population_representativeness=map_answer(
                        suitability_dict.get("population_representativeness", suitability_dict.get("representativeness", "unknown")),
                        "population_representativeness"
                    ),
                    privacy_ethics=map_answer(
                        suitability_dict.get("privacy_ethics", suitability_dict.get("privacy", "unknown")),
                        "privacy_ethics"
                    ),
                    quality_sufficiency=map_answer(
                        suitability_dict.get("quality_sufficiency", suitability_dict.get("sufficiency", "unknown")),
                        "quality_sufficiency"
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to create data suitability assessment: {e}")
                data_suitability_assessment = None
        
        complete_scoping_data = {
            "active_step": scoping_data.active_step,
            "selected_use_case": scoping_data.selected_use_case.dict() if scoping_data.selected_use_case else None,
            "selected_dataset": scoping_data.selected_dataset.dict() if scoping_data.selected_dataset else None,
            "technical_infrastructure": scoping_data.technical_infrastructure.dict(),
            "infrastructure_assessment": scoping_data.infrastructure_assessment.dict(),
            "suitability_checks": scoping_data.suitability_checks,
            "data_suitability": scoping_data.data_suitability.dict(),
            "ready_to_proceed": scoping_data.ready_to_proceed,
            "reasoning": scoping_data.reasoning,
            "completed_at": datetime.utcnow().isoformat(),
        }
        
        await project_service.update_project_phase(
            project_id=project_id,
            phase="scoping",
            phase_data=complete_scoping_data,
            advance_phase=scoping_data.ready_to_proceed,
            selected_use_case=scoping_data.selected_use_case,
            selected_dataset=scoping_data.selected_dataset,
            data_suitability_assessment=data_suitability_assessment,
            deployment_environment=deployment_environment
        )
        
        final_decision = ProjectReadinessDecision(
            ready_to_proceed=scoping_data.ready_to_proceed,
            overall_score=int((scoping_data.infrastructure_assessment.score * 0.7) + (scoping_data.data_suitability.percentage * 0.3)),
            infrastructure_score=scoping_data.infrastructure_assessment.score,
            suitability_score=scoping_data.data_suitability.percentage,
            key_recommendations=scoping_data.infrastructure_assessment.recommendations[:3],
            areas_for_improvement=scoping_data.infrastructure_assessment.non_ai_alternatives or []
        )
        
        response_data = {
            "scoping_completed": True,
            "ready_to_proceed": scoping_data.ready_to_proceed,
            "next_phase": "development" if scoping_data.ready_to_proceed else "scoping",
            "final_decision": final_decision.dict(),
            "infrastructure_score": scoping_data.infrastructure_assessment.score,
            "suitability_score": scoping_data.data_suitability.percentage,
            "selected_use_case": scoping_data.selected_use_case.dict() if scoping_data.selected_use_case else None,
            "selected_dataset": scoping_data.selected_dataset.dict() if scoping_data.selected_dataset else None
        }
        
        return APIResponse(
            success=True,
            data=response_data,
            message="Scoping phase completed successfully" + (
                " - ready to advance to development phase" if scoping_data.ready_to_proceed 
                else " - consider revising project setup"
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete scoping phase: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            await session_manager.close_all_sessions()
        except Exception as cleanup_error:
            logger.warning(f"Session cleanup error: {cleanup_error}")

@router.get("/{project_id}/status", response_model=APIResponse[Dict[str, Any]])
async def get_scoping_status(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        status_data = {
            "completed": project.scoping_data is not None,
            "scoping_data": project.scoping_data,
            "selected_use_case": project.selected_use_case.dict() if project.selected_use_case else None,
            "selected_dataset": project.selected_dataset.dict() if project.selected_dataset else None,
            "data_suitability_assessment": project.data_suitability_assessment.dict() if project.data_suitability_assessment else None,
            "deployment_environment": project.deployment_environment.dict() if project.deployment_environment else None,
        }
        
        return APIResponse(data=status_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scoping status: {e}")
        raise HTTPException(status_code=500, detail=str(e))