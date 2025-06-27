from typing import List, Optional, Dict, Any
from models.project import EthicalConsideration, Project, ProjectReadinessAssessment, ProjectStatus, EthicalAssessment, UseCase, Dataset, DataSuitabilityAssessment, DeploymentEnvironment, EvaluationResults
from core.exceptions import ProjectNotFoundError
from services.project_analysis_service import ProjectAnalysisService
from bson import ObjectId
from datetime import datetime
import logging
import core

logger = logging.getLogger(__name__)

class ProjectService:
    def __init__(self):
        self.project_analysis = ProjectAnalysisService()
        
    async def create_project(
        self,
        description: str,
        title: str,
        context: Optional[str] = None,
        tags: List[str] = None
    ) -> Project:
        """Create a new project with comprehensive info extraction and ethical considerations"""
        try:
            # Extract comprehensive project information using the new service
            project_info = await self.project_analysis.extract_project_info(description, context)
            
            # Only problem_domain has fallback, others can be None
            problem_domain = project_info.get("problem_domain", "general_humanitarian")
            target_beneficiaries = project_info.get("target_beneficiaries")
            geographic_context = project_info.get("geographic_context")
            urgency_level = project_info.get("urgency_level")
            
            ethical_considerations_data = await core.rag_service.get_ethical_considerations_for_project(
                project_description=description,
                problem_domain=problem_domain,
                target_beneficiaries=target_beneficiaries or ""
            )
            
            # Convert to EthicalConsideration objects
            ethical_considerations = [
                EthicalConsideration(**consideration)
                for consideration in ethical_considerations_data
            ]
            
            project = Project(
                title=title,
                description=description,
                context=context,
                tags=tags or [],
                status=ProjectStatus.CREATED,
                current_phase="reflection",
                problem_domain=problem_domain,
                target_beneficiaries=target_beneficiaries,  # Can be None
                geographic_context=geographic_context,      # Can be None
                urgency_level=urgency_level,                # Can be None
                ethical_considerations=ethical_considerations,
                ethical_considerations_acknowledged=False
            )
            
            await project.insert()
            logger.info(
                f"Created project: {project.id} with domain: {problem_domain}, "
                f"beneficiaries: {target_beneficiaries or 'not extracted'}, "
                f"and {len(ethical_considerations)} ethical considerations"
            )
            return project
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        try:
            if not ObjectId.is_valid(project_id):
                return None
            
            project = await Project.get(ObjectId(project_id))
            return project
        except Exception as e:
            logger.error(f"Failed to get project {project_id}: {e}")
            return None

    async def get_project_domain(self, project_id: str) -> str:
        """Get the problem domain for a project"""
        try:
            project = await self.get_project(project_id)
            if not project:
                return "general_humanitarian"
            
            # Return stored domain or extract if not available
            if hasattr(project, 'problem_domain') and project.problem_domain:
                return project.problem_domain
            else:
                # Extract and store domain if not available
                domain = await self.project_analysis.extract_problem_domain(
                    project.description, project.context
                )
                project.problem_domain = domain
                await project.save()
                logger.info(f"Extracted and stored domain for project {project_id}: {domain}")
                return domain
                
        except Exception as e:
            logger.error(f"Failed to get project domain {project_id}: {e}")
            return "general_humanitarian"

    async def get_project_with_development_context(self, project_id: str) -> Optional[Project]:
        """Get project with all data needed for development phase"""
        try:
            project = await self.get_project(project_id)
            if not project:
                return None
            
            # Validate that required phases are completed
            if not project.reflection_data:
                logger.warning(f"Project {project_id} missing reflection data")
                
            if not project.scoping_data:
                logger.warning(f"Project {project_id} missing scoping data")
            
            # Ensure problem domain is set
            if not hasattr(project, 'problem_domain') or not project.problem_domain:
                project.problem_domain = await self.project_analysis.extract_problem_domain(
                    project.description, project.context
                )
                await project.save()
                
            return project
        except Exception as e:
            logger.error(f"Failed to get project with development context {project_id}: {e}")
            return None
    
    async def list_projects(
        self,
        skip: int = 0,
        limit: int = 10,
        status: Optional[str] = None
    ) -> List[Project]:
        """List projects with pagination and filtering"""
        try:
            query = {}
            if status:
                query["status"] = status
            
            projects = await Project.find(query)\
                .skip(skip)\
                .limit(limit)\
                .sort([("created_at", -1)])\
                .to_list()
            
            return projects
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []
    
    async def get_project_with_metadata(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project with all data and metadata for sync"""
        try:
            project = await self.get_project(project_id)
            if not project:
                return None
                
            return {
                "id": str(project.id),
                "title": project.title,
                "description": project.description,
                "status": project.status,
                "current_phase": project.current_phase,
                "problem_domain": getattr(project, 'problem_domain', 'general_humanitarian'),
                "phases": [
                    {
                        "id": "reflection",
                        "status": "completed" if project.reflection_data else "not-started",
                        "progress": 100 if project.reflection_data else 0
                    },
                    {
                        "id": "scoping", 
                        "status": "completed" if project.scoping_data else "not-started",
                        "progress": 100 if project.scoping_data else 0
                    },
                    {
                        "id": "development",
                        "status": "completed" if project.development_data and project.development_data.get("completed_at") else "not-started", 
                        "progress": 100 if project.development_data and project.development_data.get("completed_at") else 0
                    },
                    {
                        "id": "evaluation",
                        "status": "completed" if project.evaluation_data else "not-started",
                        "progress": 100 if project.evaluation_data else 0
                    }
                ],
                "reflection_questions": project.reflection_questions,
                "reflection_data": project.reflection_data,
                "scoping_data": project.scoping_data,
                "development_data": project.development_data,
                "evaluation_data": project.evaluation_data,
                "updated_at": project.updated_at.isoformat(),
                "version": project.version
            }
        except Exception as e:
            logger.error(f"Failed to get project with metadata: {e}")
            return None
    
    async def update_project_phase(
        self,
        project_id: str,
        phase: str,
        phase_data: Dict[str, Any],
        advance_phase: bool = False,
        project_readiness_assessment: Optional[ProjectReadinessAssessment] = None,  # NEW
        ethical_assessment: Optional[EthicalAssessment] = None,  # Keep for backward compatibility
        selected_use_case: Optional[UseCase] = None,
        selected_dataset: Optional[Dataset] = None,
        data_suitability_assessment: Optional[DataSuitabilityAssessment] = None,
        deployment_environment: Optional[DeploymentEnvironment] = None,
        evaluation_results: Optional[EvaluationResults] = None
    ):
        """Update project phase data and advance phase if needed"""
        try:
            project = await self.get_project(project_id)
            if not project:
                raise ValueError("Project not found")
            
            # Update phase data
            if phase == "reflection":
                project.reflection_data = phase_data
            elif phase == "scoping":
                project.scoping_data = phase_data
            elif phase == "development":
                project.development_data = phase_data
            elif phase == "evaluation":
                project.evaluation_data = phase_data
            
            # NEW: Update comprehensive project readiness assessment
            if project_readiness_assessment:
                project.project_readiness_assessment = project_readiness_assessment
            
            # Update core assessments if provided (keep existing functionality)
            if ethical_assessment:
                project.ethical_assessment = ethical_assessment
            
            if selected_use_case:
                project.selected_use_case = selected_use_case
                
            if selected_dataset:
                project.selected_dataset = selected_dataset
                
            if data_suitability_assessment:
                project.data_suitability_assessment = data_suitability_assessment
                
            if deployment_environment:
                project.deployment_environment = deployment_environment
                
            if evaluation_results:
                project.evaluation_results = evaluation_results
            
            if advance_phase:
                phase_order = ["reflection", "scoping", "development", "evaluation", "completed"]
                current_index = phase_order.index(project.current_phase)
                if current_index < len(phase_order) - 1:
                    project.current_phase = phase_order[current_index + 1]
                    project.status = ProjectStatus(project.current_phase)
            
            # Update timestamp and version
            project.touch()
            
            await project.save()
            return project
        except Exception as e:
            logger.error(f"Failed to update project phase: {e}")
            raise

    async def delete_project(self, project_id: str) -> bool:
        """Delete project"""
        try:
            project = await self.get_project(project_id)
            if not project:
                return False
            
            await project.delete()
            logger.info(f"Deleted project: {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}")
            return False

    async def get_project_development_summary(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of project data relevant for development phase"""
        try:
            project = await self.get_project_with_development_context(project_id)
            if not project:
                return None
            
            # Extract key information for development
            summary = {
                "project_info": {
                    "id": str(project.id),
                    "title": project.title,
                    "description": project.description,
                    "problem_domain": getattr(project, 'problem_domain', 'general_humanitarian'),
                    "tags": project.tags
                },
                "selected_use_case": project.selected_use_case.dict() if project.selected_use_case else None,
                "selected_dataset": project.selected_dataset.dict() if project.selected_dataset else None,
                "deployment_environment": project.deployment_environment.dict() if project.deployment_environment else None,
                "ethical_assessment": project.ethical_assessment.dict() if project.ethical_assessment else None,
                "reflection_insights": self._extract_reflection_insights(project.reflection_data),
                "scoping_insights": self._extract_scoping_insights(project.scoping_data)
            }
            
            return summary
        except Exception as e:
            logger.error(f"Failed to get project development summary: {e}")
            return None
    
    def _extract_reflection_insights(self, reflection_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key insights from reflection phase"""
        if not reflection_data:
            return {}
        
        insights = {
            "ethical_score": reflection_data.get("ethical_score", 0),
            "key_concerns": reflection_data.get("actionable_recommendations", []),
            "target_beneficiaries": "",
            "potential_risks": ""
        }
        
        # Extract target beneficiaries and risks from answers
        answers = reflection_data.get("answers", {})
        for key, answer in answers.items():
            if "beneficiar" in key.lower() or "target" in key.lower():
                insights["target_beneficiaries"] = answer[:200] if answer else ""
            elif "harm" in key.lower() or "risk" in key.lower():
                insights["potential_risks"] = answer[:200] if answer else ""
        
        return insights
    
    def _extract_scoping_insights(self, scoping_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key insights from scoping phase"""
        if not scoping_data:
            return {}
        
        return {
            "feasibility_score": scoping_data.get("feasibility_summary", {}).get("overall_percentage", 0),
            "feasibility_level": scoping_data.get("feasibility_summary", {}).get("feasibility_level", "medium"),
            "data_suitability": scoping_data.get("data_suitability", {}).get("suitability_level", "moderate"),
            "key_constraints": scoping_data.get("feasibility_summary", {}).get("key_constraints", []),
            "ready_to_proceed": scoping_data.get("ready_to_proceed", False)
        }

    async def get_ethical_considerations(self, project_id: str) -> List[Dict[str, Any]]:
        """Get ethical considerations for a project"""
        try:
            project = await self.get_project(project_id)
            if not project or not project.ethical_considerations:
                return []
            
            return [consideration.dict() for consideration in project.ethical_considerations]
        except Exception as e:
            logger.error(f"Failed to get ethical considerations for project {project_id}: {e}")
            return []

    async def acknowledge_ethical_considerations(
        self, 
        project_id: str, 
        acknowledged_considerations: List[str] = None
    ) -> bool:
        """Mark ethical considerations as acknowledged by user"""
        try:
            project = await self.get_project(project_id)
            if not project:
                return False
            
            # Mark overall acknowledgment
            project.ethical_considerations_acknowledged = True
            
            # Mark specific considerations as acknowledged if provided
            if acknowledged_considerations and project.ethical_considerations:
                for consideration in project.ethical_considerations:
                    if consideration.id in acknowledged_considerations:
                        consideration.acknowledged = True
            
            project.touch()
            await project.save()
            
            logger.info(f"Acknowledged ethical considerations for project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to acknowledge ethical considerations for project {project_id}: {e}")
            return False

    async def refresh_ethical_considerations(self, project_id: str) -> List[Dict[str, Any]]:
        """Refresh ethical considerations for a project (useful if RAG data is updated)"""
        try:
            project = await self.get_project(project_id)
            if not project:
                return []
            
            # Use stored target beneficiaries (can be None)
            target_beneficiaries = project.target_beneficiaries
            
            # Check if reflection data has more specific beneficiary information
            if project.reflection_data and project.reflection_data.get("answers"):
                answers = project.reflection_data["answers"]
                for key, answer in answers.items():
                    if ("beneficiar" in key.lower() or "target" in key.lower()) and answer.strip():
                        # Use reflection answer if it's available and target_beneficiaries wasn't extracted
                        if not target_beneficiaries or len(answer.strip()) > len(target_beneficiaries):
                            target_beneficiaries = answer.strip()
                        break
            
            # Get fresh ethical considerations 
            # Use empty string if target_beneficiaries is None since RAG query needs a string
            ethical_considerations_data = await core.rag_service.get_ethical_considerations_for_project(
                project_description=project.description,
                problem_domain=project.problem_domain or "general_humanitarian",
                target_beneficiaries=target_beneficiaries or ""
            )
            
            # Convert to EthicalConsideration objects
            project.ethical_considerations = [
                EthicalConsideration(**consideration)
                for consideration in ethical_considerations_data
            ]
            
            # Reset acknowledgment since considerations have changed
            project.ethical_considerations_acknowledged = False
            
            # Update target beneficiaries if we found better info from reflection
            if target_beneficiaries and target_beneficiaries != project.target_beneficiaries:
                project.target_beneficiaries = target_beneficiaries
            
            project.touch()
            await project.save()
            
            logger.info(f"Refreshed ethical considerations for project {project_id} with beneficiaries: {target_beneficiaries or 'not specified'}")
            return ethical_considerations_data
        except Exception as e:
            logger.error(f"Failed to refresh ethical considerations for project {project_id}: {e}")
            return []