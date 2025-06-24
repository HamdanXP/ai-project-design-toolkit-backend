from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from models.project import Dataset, UseCase, DataSuitabilityAssessment, DeploymentEnvironment


class FeasibilitySummary(BaseModel):
    overall_percentage: int
    feasibility_level: str  # 'high', 'medium', 'low'
    key_constraints: List[str]


class DataSuitabilitySummary(BaseModel):
    percentage: int
    suitability_level: str  # 'excellent', 'good', 'moderate', 'poor'


class ScopingCompletionRequest(BaseModel):
    # Main selections
    selected_use_case: Optional[UseCase] = None
    selected_dataset: Optional[Dataset] = None
    
    # Assessment summaries
    feasibility_summary: FeasibilitySummary
    data_suitability: DataSuitabilitySummary
    
    # All constraint responses for detailed storage
    constraints: List[Dict[str, Any]] = []
    
    # All suitability check responses for detailed storage
    suitability_checks: List[Dict[str, Any]] = []
    
    # Phase completion data
    active_step: int = 5
    ready_to_proceed: bool = False
    reasoning: Optional[str] = None
    
    # Optional deployment environment from constraints
    deployment_environment: Optional[DeploymentEnvironment] = None


class ScopingCompletionResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]
    next_phase: str = "development"


class FinalFeasibilityDecision(BaseModel):
    ready_to_proceed: bool
    overall_score: int
    feasibility_level: str
    suitability_level: str
    key_recommendations: List[str]
    areas_for_improvement: List[str]