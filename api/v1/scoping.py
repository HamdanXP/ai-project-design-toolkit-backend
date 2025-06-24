from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from config import settings
from models.response import APIResponse
from models.project import UseCase, Dataset, DeploymentEnvironment, DataSuitabilityAssessment
from models.phase import ScopingRequest, ScopingResponse
from models.scoping import ScopingCompletionRequest, ScopingCompletionResponse, FinalFeasibilityDecision
from services.phase_services.scoping import ScopingService
from services.project_service import ProjectService
from api.dependencies import get_project_service, get_scoping_service
from utils.session_manager import session_manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scoping", tags=["scoping"])

@router.get("/{project_id}/use-cases", response_model=APIResponse[List[Dict[str, Any]]])
async def get_similar_use_cases(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    scoping_service: ScopingService = Depends(get_scoping_service)
):
    """Get AI use cases with enhanced filtering and humanitarian-focused educational content"""
    try:
        # Validate project exists and get domain
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get stored problem domain
        problem_domain = await project_service.get_project_domain(project_id)
        
        # Log the request
        logger.info(f"Fetching AI use cases for project {project_id}, domain: {problem_domain}")
        logger.info(f"Project description: {project.description[:100]}...")
        
        # Get AI use cases with enhanced filtering and targeted enrichment
        try:
            use_cases = await scoping_service.get_educational_use_cases(
                project.description, 
                problem_domain
            )
            logger.info(f"Retrieved and processed {len(use_cases)} AI use cases")
            
            # Validate and clean use cases (no fallback content)
            validated_use_cases = []
            for uc in use_cases:
                try:
                    # Basic validation structure
                    validated_uc = {
                        "id": uc.get("id", f"uc_{hash(uc.get('title', '')) % 10000}"),
                        "title": uc.get("title", "Untitled AI Use Case"),
                        "description": uc.get("description", "No description available"),
                        "source": uc.get("source", "Unknown"),
                        "source_url": uc.get("source_url", ""),
                        "type": uc.get("type", "ai_application"),
                        "category": uc.get("category", "General"),
                        
                        # Educational content - use as-is from enrichment (may be empty)
                        "how_it_works": uc.get("how_it_works", ""),
                        "real_world_impact": uc.get("real_world_impact", ""),
                        "similarity_to_project": uc.get("similarity_to_project", ""),
                        "real_world_examples": uc.get("real_world_examples", ""),
                        "implementation_approach": uc.get("implementation_approach", ""),
                        "decision_guidance": uc.get("decision_guidance", ""),
                        
                        # Practical information (may be empty lists)
                        "key_success_factors": uc.get("key_success_factors", []),
                        "resource_requirements": uc.get("resource_requirements", []),
                        "challenges": uc.get("challenges", [])
                    }
                    
                    # Add optional metadata if available
                    optional_fields = ["authors", "published_date", "organization", "date", "venue", "citation_count"]
                    for field in optional_fields:
                        if uc.get(field):
                            validated_uc[field] = uc[field]
                    
                    # Quality check - only include meaningful use cases
                    if (validated_uc["title"] != "Untitled AI Use Case" and 
                        len(validated_uc["description"]) > 20):
                        validated_use_cases.append(validated_uc)
                    
                except Exception as uc_error:
                    logger.warning(f"Failed to validate use case: {uc_error}")
                    continue
            
            # Return results
            final_use_cases = validated_use_cases[:settings.max_use_cases_returned]
            
            if final_use_cases:
                logger.info(f"Returning {len(final_use_cases)} validated AI use cases")
                
                # Count how many have educational content
                enriched_count = sum(1 for uc in final_use_cases 
                                   if uc.get("how_it_works") or uc.get("real_world_impact"))
                
                if enriched_count > 0:
                    message = f"Found {len(final_use_cases)} relevant AI use cases for {problem_domain} with humanitarian-focused guidance"
                else:
                    message = f"Found {len(final_use_cases)} relevant AI use cases for {problem_domain}. Visit source links for detailed information."
                
                return APIResponse(
                    success=True,
                    data=final_use_cases,
                    message=message
                )
            else:
                logger.info("No valid AI use cases found after processing")
                return APIResponse(
                    success=True,
                    data=[],
                    message=f"No suitable AI use cases found for {problem_domain}. Consider exploring general AI principles or consulting with AI experts."
                )
            
        except Exception as e:
            logger.error(f"AI use case retrieval and processing failed: {e}")
            return APIResponse(
                success=True,
                data=[],
                message=f"Unable to retrieve AI use cases for {problem_domain}. You may proceed with general AI development principles."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_similar_use_cases: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )
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
    """Get recommended datasets for the selected use case - HUMANITARIAN SOURCES ONLY"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get problem domain for better search
        problem_domain = await project_service.get_project_domain(project_id)
        
        use_case_id = request_data.get("use_case_id", "")
        use_case_title = request_data.get("use_case_title", "")
        use_case_description = request_data.get("use_case_description", "")
        
        logger.info(f"Getting datasets for project {project_id}, domain: {problem_domain}, use case: {use_case_title}")
        
        # Use the updated dataset discovery service with domain
        try:
            datasets = await scoping_service.recommend_datasets(
                project.description,
                use_case_title,
                use_case_description,
                problem_domain  # Pass the domain
            )
            
            logger.info(f"Retrieved {len(datasets)} datasets from humanitarian sources")
            
            if datasets:
                return APIResponse(
                    success=True,
                    data=datasets,
                    message=f"Found {len(datasets)} relevant datasets from humanitarian sources for {problem_domain}"
                )
            else:
                logger.info("No datasets found from humanitarian sources")
                return APIResponse(
                    success=True,
                    data=[],
                    message=f"No datasets found from humanitarian data sources for {problem_domain}. This is common for specialized AI projects."
                )
                
        except Exception as e:
            logger.error(f"Dataset retrieval failed: {e}")
            return APIResponse(
                success=True,
                data=[],
                message=f"Dataset search failed for {problem_domain}. No datasets available from humanitarian sources."
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

@router.post("/{project_id}/complete", response_model=APIResponse[Dict[str, Any]])
async def complete_scoping_phase(
    project_id: str,
    scoping_data: ScopingCompletionRequest,
    project_service: ProjectService = Depends(get_project_service)
):
    """Complete the scoping phase and save all scoping decisions and assessments"""
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        logger.info(f"Completing scoping phase for project {project_id}")
        
        # Create deployment environment from constraints
        deployment_environment = None
        if scoping_data.constraints:
            # Extract deployment environment data from constraints
            constraint_dict = {c.get("id"): c.get("value") for c in scoping_data.constraints}
            
            deployment_environment = DeploymentEnvironment(
                # Resources & Budget
                project_budget=constraint_dict.get("budget", "limited"),
                project_timeline=constraint_dict.get("time", "medium-term"),
                team_size=constraint_dict.get("team-size", "small"),
                
                # Technical Infrastructure
                computing_resources=constraint_dict.get("compute", "cloud"),
                reliable_internet_connection=constraint_dict.get("internet", True),
                local_technology_setup=constraint_dict.get("infrastructure", True),
                
                # Team Expertise
                ai_ml_experience=constraint_dict.get("ai-experience", "beginner"),
                technical_skills=constraint_dict.get("technical-skills", "moderate"),
                learning_training_capacity=constraint_dict.get("learning-capacity", True),
                
                # Organizational Readiness
                stakeholder_buy_in=constraint_dict.get("stakeholder-support", "moderate"),
                change_management_readiness=constraint_dict.get("change-management", False),
                data_governance=constraint_dict.get("data-governance", "developing"),
                
                # External Factors
                regulatory_requirements=constraint_dict.get("regulatory-compliance", "moderate"),
                external_partnerships=constraint_dict.get("partnerships", False),
                long_term_sustainability_plan=constraint_dict.get("sustainability", False)
            )
        
        # Create data suitability assessment from suitability checks
        data_suitability_assessment = None
        if scoping_data.suitability_checks:
            # Map suitability check answers to assessment
            suitability_dict = {c.get("id"): c.get("answer") for c in scoping_data.suitability_checks}
            
            # Map frontend answers to backend enum values
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
                    # Default mapping
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
        
        # Prepare complete scoping data for storage
        complete_scoping_data = {
            "active_step": scoping_data.active_step,
            "selected_use_case": scoping_data.selected_use_case.dict() if scoping_data.selected_use_case else None,
            "selected_dataset": scoping_data.selected_dataset.dict() if scoping_data.selected_dataset else None,
            "constraints": scoping_data.constraints,
            "suitability_checks": scoping_data.suitability_checks,
            "feasibility_summary": scoping_data.feasibility_summary.dict(),
            "data_suitability": scoping_data.data_suitability.dict(),
            "ready_to_proceed": scoping_data.ready_to_proceed,
            "reasoning": scoping_data.reasoning,
            "completed_at": datetime.utcnow().isoformat(),
        }
        
        # Update project with all scoping data and core assessments
        await project_service.update_project_phase(
            project_id=project_id,
            phase="scoping",
            phase_data=complete_scoping_data,
            advance_phase=scoping_data.ready_to_proceed,  # Only advance if ready to proceed
            selected_use_case=scoping_data.selected_use_case,
            selected_dataset=scoping_data.selected_dataset,
            data_suitability_assessment=data_suitability_assessment,
            deployment_environment=deployment_environment
        )
        
        # Create final decision summary
        final_decision = FinalFeasibilityDecision(
            ready_to_proceed=scoping_data.ready_to_proceed,
            overall_score=scoping_data.feasibility_summary.overall_percentage,
            feasibility_level=scoping_data.feasibility_summary.feasibility_level,
            suitability_level=scoping_data.data_suitability.suitability_level,
            key_recommendations=[
                f"Project feasibility: {scoping_data.feasibility_summary.feasibility_level}",
                f"Data suitability: {scoping_data.data_suitability.suitability_level}",
            ] + scoping_data.feasibility_summary.key_constraints,
            areas_for_improvement=scoping_data.feasibility_summary.key_constraints
        )
        
        response_data = {
            "scoping_completed": True,
            "ready_to_proceed": scoping_data.ready_to_proceed,
            "next_phase": "development" if scoping_data.ready_to_proceed else "scoping",
            "final_decision": final_decision.dict(),
            "feasibility_score": scoping_data.feasibility_summary.overall_percentage,
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
    """Get scoping phase status and results"""
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