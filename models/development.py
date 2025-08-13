from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
from models.enums import (
    AITechnique,
    DeploymentStrategy,
)
from datetime import datetime


class RequiredFeature(BaseModel):
    name: str
    description: str
    data_type: str
    humanitarian_purpose: str

class TabularDataRequirements(BaseModel):
    required_features: List[RequiredFeature]
    optional_features: List[RequiredFeature]
    minimum_rows: int
    data_types: Dict[str, str]

class ResourceRequirement(BaseModel):
    computing_power: str
    storage_needs: str
    internet_dependency: str
    technical_expertise: str
    setup_time: str

class ProjectRecommendation(BaseModel):
    type: str
    title: str
    description: str
    confidence: int = Field(..., ge=0, le=100)
    reason: str
    deployment_strategy: DeploymentStrategy

class EthicalSafeguard(BaseModel):
    category: str
    measures: List[str]
    icon: str
    priority: str = "medium"

class TechnicalArchitecture(BaseModel):
    ai_technique: AITechnique
    deployment_strategy: DeploymentStrategy
    implementation: str
    ai_component: str
    data_input: str
    output_format: str
    user_interface: str
    deployment_method: str

class AISolution(BaseModel):
    id: str
    title: str
    description: str
    ai_technique: AITechnique
    deployment_strategy: DeploymentStrategy
    recommended: bool = False
    confidence_score: int = Field(..., ge=0, le=100)
    needs_dataset: bool
    dataset_type: Optional[Literal["tabular", "text", "image", "audio", "video", "none"]] = None
    tabular_requirements: Optional[TabularDataRequirements] = None
    capabilities: List[str]
    key_features: List[str]
    technical_architecture: TechnicalArchitecture
    resource_requirements: ResourceRequirement
    best_for: str
    use_case_alignment: str
    implementation_notes: List[str]
    ethical_safeguards: List[EthicalSafeguard]
    estimated_setup_time: str
    maintenance_requirements: List[str]
    data_requirements: List[str]
    output_examples: List[str]

class ProjectContext(BaseModel):
    title: str
    description: str
    target_beneficiaries: str
    problem_domain: str
    
    selected_use_case: Optional[Dict[str, Any]] = None
    use_case_analysis: Optional[Dict[str, Any]] = None

    technical_infrastructure: Optional[Dict[str, Any]] = None
    deployment_analysis: Optional[Dict[str, Any]] = None
    
    recommendations: List[ProjectRecommendation]
    technical_recommendations: List[str]
    deployment_recommendations: List[str]

class ProjectContextOnly(BaseModel):
    project_context: ProjectContext
    ethical_safeguards: List[EthicalSafeguard]
    solution_rationale: Optional[str] = "AI solutions will be generated when you proceed to the next step."

class SolutionsData(BaseModel):
    available_solutions: List[AISolution]
    solution_rationale: str

class FileAnalysis(BaseModel):
    filename: str
    purpose: str
    content_type: str
    key_features: List[str]
    dependencies: List[str]

class EthicalGuardrailStatus(BaseModel):
    category: str
    status: str
    implementation_details: List[str]
    verification_method: str

class GenerationReport(BaseModel):
    solution_approach: str
    files_generated: List[FileAnalysis]
    ethical_implementation: List[EthicalGuardrailStatus]
    architecture_decisions: List[str]
    deployment_considerations: List[str]

class GeneratedProject(BaseModel):
    id: str
    title: str
    description: str
    solution_type: str
    ai_technique: AITechnique
    deployment_strategy: DeploymentStrategy
    
    files: Dict[str, str]
    documentation: str
    setup_instructions: str
    deployment_guide: str
    ethical_assessment_guide: str
    technical_handover_package: str
    
    generation_report: GenerationReport
    
    api_documentation: Optional[str] = None
    integration_examples: Dict[str, str] = {}

class ProjectGenerationRequest(BaseModel):
    solution_id: str
    project_requirements: Optional[Dict[str, Any]] = None
    customizations: Optional[Dict[str, Any]] = None
    ethical_preferences: List[str] = []
    deployment_preferences: Dict[str, Any] = {}
    integration_requirements: List[str] = []

class ProjectGenerationResponse(BaseModel):
    success: bool
    project: Optional[GeneratedProject] = None
    generation_steps: List[str]
    estimated_completion_time: str
    next_steps: List[str]
    alternative_approaches: List[str] = []

class SolutionSelection(BaseModel):
    solution_id: str
    solution_title: str
    selected_at: str
    reasoning: Optional[str] = None

class DevelopmentStatus(BaseModel):
    completed: bool
    phase_status: str
    
    context_loaded: bool = False
    solutions_generated: bool = False
    solutions_loading: bool = False
    
    selected_solution: Optional[SolutionSelection] = None
    generated_project: bool = False
    development_data: Optional[Dict[str, Any]] = None
    can_proceed: bool = False
    
    performance_metrics: Optional[Dict[str, Any]] = None

class DevelopmentPhaseData(BaseModel):
    project_context: ProjectContext
    available_solutions: List[AISolution]
    ethical_safeguards: List[EthicalSafeguard]
    solution_rationale: str
    
    context_loaded: bool = True
    solutions_loaded: bool = True
    loading_metadata: Optional[Dict[str, Any]] = None

class ProjectContextOnlyBackend(BaseModel):
    project_context: ProjectContext
    ethical_safeguards: List[EthicalSafeguard]
    solution_rationale: Optional[str] = None
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    performance_metrics: Dict[str, Any] = {}

class SolutionsDataBackend(BaseModel):
    available_solutions: List[AISolution]
    solution_rationale: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    performance_metrics: Dict[str, Any] = {}
    generation_metadata: Dict[str, Any] = {}

class DevelopmentError(BaseModel):
    type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    project_id: Optional[str] = None
    recoverable: bool = True
    suggested_action: Optional[str] = None

class DevelopmentMetrics(BaseModel):
    project_id: str
    phase: str
    start_time: str
    end_time: str
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    cache_hit: bool = False
    llm_calls: Optional[int] = None
    solutions_generated: Optional[int] = None
    memory_usage_mb: Optional[float] = None

class DevelopmentCacheEntry(BaseModel):
    data: Dict[str, Any]
    cached_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: str
    cache_key: str
    project_id: str
    cache_type: str

class DevelopmentApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class DevelopmentRequest(BaseModel):
    project_id: str
    endpoint: str
    request_id: str = Field(default_factory=lambda: f"req_{int(datetime.utcnow().timestamp() * 1000)}")
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None

class SolutionModificationRequest(BaseModel):
    feedback: str
    current_step: Optional[int] = None
    return_to_development: Optional[bool] = None

class SolutionModificationResponse(BaseModel):
    success: bool
    message: str
    modified_solution: Optional[AISolution] = None
    modifications_made: List[str] = []
    redirect_to_development: Optional[bool] = None