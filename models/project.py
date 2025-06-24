from beanie import Document, Indexed
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ProjectStatus(str, Enum):
    CREATED = "created"
    REFLECTION = "reflection"
    SCOPING = "scoping"
    DEVELOPMENT = "development"
    EVALUATION = "evaluation"
    COMPLETED = "completed"
    DEPLOYED = "deployed"

class DataAssessmentResponse(str, Enum):
    LOOKS_CLEAN = "looks_clean"
    SOME_ISSUES = "some_issues"
    MANY_PROBLEMS = "many_problems"

class RepresentativenessResponse(str, Enum):
    REPRESENTATIVE = "representative"
    PARTIALLY = "partially"
    LIMITED_COVERAGE = "limited_coverage"

class PrivacyEthicsResponse(str, Enum):
    PRIVACY_SAFE = "privacy_safe"
    NEED_REVIEW = "need_review"
    HIGH_RISK = "high_risk"

class QualitySufficiencyResponse(str, Enum):
    SUFFICIENT = "sufficient"
    BORDERLINE = "borderline"
    INSUFFICIENT = "insufficient"


class EthicalAssessment(BaseModel):
    score: float = Field(..., ge=0, le=1)
    concerns: List[str] = []
    recommendations: List[str] = []
    approved: bool = False


class UseCase(BaseModel):
    id: str
    title: str
    description: str
    category: str
    complexity: str  # low, medium, high
    source_url: Optional[str] = None
    similarity_score: Optional[float] = None
    tags: List[str] = []


class Dataset(BaseModel):
    name: str
    source: str
    url: Optional[str] = None
    description: str
    size_estimate: Optional[str] = None
    data_types: List[str] = []
    ethical_concerns: List[str] = []
    suitability_score: Optional[float] = None


class DeploymentEnvironment(BaseModel):
    # Resources & Budget
    project_budget: str
    project_timeline: str 
    team_size: str
    
    # Technical Infrastructure  
    computing_resources: str 
    reliable_internet_connection: bool
    local_technology_setup: bool
    
    # Team Expertise
    ai_ml_experience: str
    technical_skills: str
    learning_training_capacity: bool
    
    # Organizational Readiness
    stakeholder_buy_in: str
    change_management_readiness: bool
    data_governance: str
    
    # External Factors
    regulatory_requirements: str
    external_partnerships: bool
    long_term_sustainability_plan: bool

class DataSuitabilityAssessment(BaseModel):
    data_completeness: DataAssessmentResponse
    population_representativeness: RepresentativenessResponse
    privacy_ethics: PrivacyEthicsResponse
    quality_sufficiency: QualitySufficiencyResponse

class EvaluationResults(BaseModel):
    accuracy_metrics: Dict[str, float] = {}
    ethical_metrics: Dict[str, float] = {}
    bias_assessment: Dict[str, Any] = {}
    performance_simulation: Dict[str, Any] = {}
    ready_for_deployment: bool = False
    recommendations: List[str] = []

class Project(Document):
    # Basic Information
    title: str
    description: str
    problem_domain: Optional[str] = None
    tags: List[str] = []
    context: Optional[str] = None
    
    # Status and Phases
    status: ProjectStatus = ProjectStatus.CREATED
    current_phase: str = "reflection"
    
    # Phase Data - Simplified
    reflection_questions: Optional[Dict[str, str]] = None
    reflection_data: Optional[Dict[str, Any]] = None
    scoping_data: Optional[Dict[str, Any]] = None
    development_data: Optional[Dict[str, Any]] = None  # Now stores the new development phase data
    evaluation_data: Optional[Dict[str, Any]] = None
    
    # Core Assessments
    ethical_assessment: Optional[EthicalAssessment] = None
    data_suitability_assessment: Optional[DataSuitabilityAssessment] = None
    selected_use_case: Optional[UseCase] = None
    selected_dataset: Optional[Dataset] = None
    deployment_environment: Optional[DeploymentEnvironment] = None    
    evaluation_results: Optional[EvaluationResults] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)
    
    # Add method to update timestamp
    def touch(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    class Settings:
        name = "projects"
        indexes = [
            [("created_at", -1)],
            [("status", 1)],
            [("current_phase", 1)],
        ]