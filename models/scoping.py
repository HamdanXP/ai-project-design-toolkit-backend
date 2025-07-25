from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from models.project import Dataset, UseCase


class TechnicalInfrastructure(BaseModel):
    computing_resources: str
    storage_data: str
    internet_connectivity: str
    deployment_environment: str


class InfrastructureScoring(BaseModel):
    score: int
    max_score: int
    reasoning: str


class InfrastructureAssessment(BaseModel):
    score: int
    can_proceed: bool
    reasoning: str
    scoring_breakdown: Dict[str, InfrastructureScoring]
    recommendations: List[str]
    non_ai_alternatives: Optional[List[str]] = None


class DataSuitabilitySummary(BaseModel):
    percentage: int
    suitability_level: str  # 'excellent', 'good', 'moderate', 'poor'


class ScopingCompletionRequest(BaseModel):
    # Main selections
    selected_use_case: Optional[UseCase] = None
    selected_dataset: Optional[Dataset] = None
    
    # Infrastructure assessment
    infrastructure_assessment: InfrastructureAssessment
    data_suitability: DataSuitabilitySummary
    
    # Technical infrastructure details
    technical_infrastructure: TechnicalInfrastructure
    
    # All suitability check responses for detailed storage
    suitability_checks: List[Dict[str, Any]] = []
    
    # Phase completion data
    active_step: int = 5
    ready_to_proceed: bool = False
    reasoning: Optional[str] = None


class ScopingCompletionResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]
    next_phase: str = "development"


class ProjectReadinessDecision(BaseModel):
    ready_to_proceed: bool
    overall_score: int
    infrastructure_score: int
    suitability_score: int
    key_recommendations: List[str]
    areas_for_improvement: List[str]