# models/development_models.py - Enhanced models with split loading support

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from models.enums import (
    AITechnique,
    DeploymentStrategy,
    ComplexityLevel,
)
from datetime import datetime


class ResourceRequirement(BaseModel):
    computing_power: str  # "low", "medium", "high"
    storage_needs: str    # "minimal", "moderate", "extensive"
    internet_dependency: str  # "offline", "periodic", "continuous"
    technical_expertise: str  # "basic", "intermediate", "advanced"
    budget_estimate: str  # "low", "medium", "high"

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
    priority: str = "medium"  # "low", "medium", "high", "critical"

class TechnicalArchitecture(BaseModel):
    ai_technique: AITechnique
    deployment_strategy: DeploymentStrategy
    frontend: str
    backend: str
    ai_components: List[str]
    data_processing: str
    deployment: str
    monitoring: str

class AISolution(BaseModel):
    id: str
    title: str
    description: str
    ai_technique: AITechnique
    complexity_level: ComplexityLevel
    deployment_strategy: DeploymentStrategy
    recommended: bool = False
    confidence_score: int = Field(..., ge=0, le=100)
    
    # Capabilities and features
    capabilities: List[str]
    key_features: List[str]
    technical_architecture: TechnicalArchitecture
    resource_requirements: ResourceRequirement
    
    # Context and suitability
    best_for: str
    use_case_alignment: str
    deployment_considerations: List[str]
    
    # Ethical and practical considerations
    ethical_safeguards: List[EthicalSafeguard]
    implementation_timeline: str
    maintenance_requirements: List[str]
    
    # API and integration options
    external_apis: List[str] = []
    integration_complexity: str = "moderate"

class ProjectContext(BaseModel):
    title: str
    description: str
    target_beneficiaries: str
    problem_domain: str
    
    # Enhanced use case analysis
    selected_use_case: Optional[Dict[str, Any]] = None
    use_case_analysis: Optional[Dict[str, Any]] = None  # LLM analysis of the use case
    
    # Enhanced deployment context
    deployment_environment: Optional[Dict[str, Any]] = None
    deployment_analysis: Optional[Dict[str, Any]] = None  # Smart analysis of constraints
    
    # Intelligent recommendations
    recommendations: List[ProjectRecommendation]
    technical_recommendations: List[str]
    deployment_recommendations: List[str]

# NEW: Split Loading Models
class ProjectContextOnly(BaseModel):
    """Fast-loading project context without AI solutions (for improved UX)"""
    project_context: ProjectContext
    ethical_safeguards: List[EthicalSafeguard]
    solution_rationale: Optional[str] = "AI solutions will be generated when you proceed to the next step."

class SolutionsData(BaseModel):
    """On-demand AI solutions generation (slow operation)"""
    available_solutions: List[AISolution]
    solution_rationale: str

# Enhanced with split loading metadata
class DevelopmentPhaseData(BaseModel):
    """Complete development phase data (legacy compatibility + new metadata)"""
    project_context: ProjectContext
    available_solutions: List[AISolution]
    ethical_safeguards: List[EthicalSafeguard]
    solution_rationale: str  # Explanation of why these solutions were chosen
    
    # Split loading metadata
    context_loaded: bool = True
    solutions_loaded: bool = True
    loading_metadata: Optional[Dict[str, Any]] = None

class GeneratedProject(BaseModel):
    id: str
    title: str
    description: str
    solution_type: str
    ai_technique: AITechnique
    deployment_strategy: DeploymentStrategy
    
    # Generated assets
    files: Dict[str, str]  # filename -> content
    documentation: str
    setup_instructions: str
    deployment_guide: str
    
    # Ethical and compliance
    ethical_audit_report: str
    bias_testing_plan: str
    monitoring_recommendations: str
    
    # API and integration
    api_documentation: Optional[str] = None
    integration_examples: Dict[str, str] = {}

class ProjectGenerationRequest(BaseModel):
    solution_id: str
    # Legacy compatibility
    project_requirements: Optional[Dict[str, Any]] = None
    # Enhanced options
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

# Enhanced DevelopmentStatus with split loading support
class DevelopmentStatus(BaseModel):
    completed: bool
    phase_status: str
    
    # Split loading status - NEW
    context_loaded: bool = False
    solutions_generated: bool = False
    solutions_loading: bool = False
    
    selected_solution: Optional[SolutionSelection] = None
    generated_project: bool = False
    development_data: Optional[Dict[str, Any]] = None
    can_proceed: bool = False
    
    # Performance metadata - NEW
    performance_metrics: Optional[Dict[str, Any]] = None

# Backend-specific models for enhanced API responses
class ProjectContextOnlyBackend(BaseModel):
    """Backend response model for context-only loading with performance metrics"""
    project_context: ProjectContext
    ethical_safeguards: List[EthicalSafeguard]
    solution_rationale: Optional[str] = None
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    performance_metrics: Dict[str, Any] = {}

class SolutionsDataBackend(BaseModel):
    """Backend response model for solutions generation with detailed metadata"""
    available_solutions: List[AISolution]
    solution_rationale: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    performance_metrics: Dict[str, Any] = {}
    generation_metadata: Dict[str, Any] = {}

# Error handling models
class DevelopmentError(BaseModel):
    type: str  # 'context_loading', 'solutions_generation', 'solution_selection', 'project_generation'
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    project_id: Optional[str] = None
    recoverable: bool = True
    suggested_action: Optional[str] = None

# Performance monitoring models
class DevelopmentMetrics(BaseModel):
    project_id: str
    phase: str  # 'context', 'solutions', 'generation'
    start_time: str
    end_time: str
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    cache_hit: bool = False
    llm_calls: Optional[int] = None
    solutions_generated: Optional[int] = None
    memory_usage_mb: Optional[float] = None

# Cache models for performance optimization
class DevelopmentCacheEntry(BaseModel):
    data: Dict[str, Any]
    cached_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: str
    cache_key: str
    project_id: str
    cache_type: str  # 'context', 'use_case_analysis', 'deployment_analysis'

# Enhanced API response wrapper
class DevelopmentApiResponse(BaseModel):
    """Enhanced API response with metadata and performance tracking"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# Request tracking for analytics
class DevelopmentRequest(BaseModel):
    project_id: str
    endpoint: str  # '/context', '/solutions', '/generate', etc.
    request_id: str = Field(default_factory=lambda: f"req_{int(datetime.utcnow().timestamp() * 1000)}")
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None

# Legacy compatibility models (maintain backward compatibility)
class LegacyTechnicalSpecs(BaseModel):
    """Legacy technical specs format for backward compatibility"""
    frontend: str
    backend: str
    deployment: str
    data: str

class LegacyAISolution(BaseModel):
    """Legacy AI solution format for backward compatibility"""
    id: str
    title: str
    description: str
    recommended: bool = False
    capabilities: List[str]
    technical_specs: LegacyTechnicalSpecs
    best_for: str
    ethical_safeguards: List[EthicalSafeguard]

class LegacyDevelopmentPhaseData(BaseModel):
    """Legacy development phase data format"""
    project_context: ProjectContext
    available_solutions: List[LegacyAISolution]
    ethical_safeguards: List[EthicalSafeguard]

# Validation helpers
def validate_ai_technique(technique: str) -> AITechnique:
    """Validate and convert AI technique string to enum"""
    try:
        return AITechnique(technique.lower())
    except ValueError:
        # Default fallback
        return AITechnique.CLASSIFICATION

def validate_deployment_strategy(strategy: str) -> DeploymentStrategy:
    """Validate and convert deployment strategy string to enum"""
    try:
        return DeploymentStrategy(strategy.lower())
    except ValueError:
        # Default fallback
        return DeploymentStrategy.CLOUD_NATIVE

def validate_complexity_level(level: str) -> ComplexityLevel:
    """Validate and convert complexity level string to enum"""
    try:
        return ComplexityLevel(level.lower())
    except ValueError:
        # Default fallback
        return ComplexityLevel.MODERATE

# Factory functions for creating default instances
def create_default_resource_requirements() -> ResourceRequirement:
    """Create default resource requirements"""
    return ResourceRequirement(
        computing_power="medium",
        storage_needs="moderate",
        internet_dependency="continuous",
        technical_expertise="intermediate",
        budget_estimate="medium"
    )

def create_default_technical_architecture(
    ai_technique: AITechnique = AITechnique.CLASSIFICATION,
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.CLOUD_NATIVE
) -> TechnicalArchitecture:
    """Create default technical architecture"""
    return TechnicalArchitecture(
        ai_technique=ai_technique,
        deployment_strategy=deployment_strategy,
        frontend="React.js with modern UI components",
        backend="Python FastAPI with ML model serving",
        ai_components=["ML model", "Data preprocessing", "API endpoints"],
        data_processing="Automated data pipeline with validation",
        deployment="Docker containers with orchestration",
        monitoring="Logging, metrics, and alerting system"
    )