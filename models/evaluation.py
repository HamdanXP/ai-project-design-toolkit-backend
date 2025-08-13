from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum

class SimulationType(str, Enum):
    STATISTICS_BASED = "statistics_based"
    EXAMPLE_SCENARIOS = "example_scenarios"
    SUITABILITY_ASSESSMENT = "suitability_assessment"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class EvaluationStatus(str, Enum):
    READY_FOR_DEPLOYMENT = "ready_for_deployment"
    NEEDS_MINOR_IMPROVEMENTS = "needs_minor_improvements"
    NEEDS_SIGNIFICANT_IMPROVEMENTS = "needs_significant_improvements"

class FileType(str, Enum):
    COMPLETE_PROJECT = "complete_project"
    DOCUMENTATION = "documentation"
    SETUP_INSTRUCTIONS = "setup_instructions"
    DEPLOYMENT_GUIDE = "deployment_guide"
    ETHICAL_ASSESSMENT_GUIDE = "ethical_assessment_guide"
    TECHNICAL_HANDOVER_PACKAGE = "technical_handover_package"

class TestingMethod(str, Enum):
    DATASET = "dataset"
    SCENARIOS = "scenarios"
    BYPASS = "bypass"

class EvaluationApproach(str, Enum):
    DATASET_ANALYSIS = "dataset_analysis"
    SCENARIO_BASED = "scenario_based"
    EVALUATION_BYPASS = "evaluation_bypass"

class FeatureCompatibility(BaseModel):
    compatible: bool
    missing_required: List[str]
    missing_optional: List[str]
    available_required: List[str]
    compatibility_score: float
    gap_explanation: str

class DataVolumeAssessment(BaseModel):
    sufficient: bool
    available_rows: int
    required_rows: int
    volume_score: float
    recommendation: str

class DataQualityAssessment(BaseModel):
    quality_score: float
    completeness_percentage: int
    issues_found: List[str]
    recommendations: List[str]

class SuitabilityRecommendation(BaseModel):
    type: Literal["data_collection", "solution_alternative", "improvement"]
    priority: Literal["low", "medium", "high"]
    issue: str
    suggestion: str

class SuitabilityAssessment(BaseModel):
    is_suitable: bool
    overall_score: float
    feature_compatibility: FeatureCompatibility
    data_volume_assessment: DataVolumeAssessment
    data_quality_assessment: DataQualityAssessment
    recommendations: List[SuitabilityRecommendation]
    performance_estimate: Optional[str] = None

class EvaluationBypass(BaseModel):
    message: str
    guidance: str
    can_download: bool
    next_steps: List[str]
    specialist_consultation: str

class SimulationCapabilities(BaseModel):
    testing_method: TestingMethod
    evaluation_approach: EvaluationApproach
    ai_technique: str
    data_formats_supported: List[str]
    explanation: str

class TestingScenario(BaseModel):
    name: str
    description: str
    input_description: str
    process_description: str
    expected_outcome: str
    success_criteria: str
    humanitarian_impact: str

class EvaluationContext(BaseModel):
    generated_project: Dict[str, Any]
    selected_solution: Dict[str, Any]
    simulation_capabilities: SimulationCapabilities
    testing_scenarios: Optional[List[TestingScenario]] = None
    evaluation_bypass: Optional[EvaluationBypass] = None
    available_downloads: List[str]

class SimulationRequest(BaseModel):
    dataset_statistics: Optional[Dict[str, Any]] = None
    simulation_type: SimulationType
    custom_scenarios: Optional[List[str]] = None

class ExampleScenario(BaseModel):
    scenario_name: str
    input_description: str
    process_description: str
    expected_output: str
    humanitarian_impact: str

class SimulationExplanation(BaseModel):
    methodology: str
    data_usage: str
    calculation_basis: List[str]
    limitations: List[str]

class SimulationResult(BaseModel):
    simulation_type: SimulationType
    testing_method: TestingMethod
    confidence_level: ConfidenceLevel
    suitability_assessment: Optional[SuitabilityAssessment] = None
    scenarios: Optional[List[ExampleScenario]] = None
    evaluation_bypass: Optional[EvaluationBypass] = None
    simulation_explanation: SimulationExplanation

class EvaluationDecision(str, Enum):
    PROCEED_WITH_SOLUTION = "proceed_with_solution"
    TRY_DIFFERENT_SOLUTION = "try_different_solution"

class EvaluationSummary(BaseModel):
    overall_assessment: str
    solution_performance: Dict[str, Any]
    deployment_readiness: bool
    recommendation: str
    key_strengths: List[str]
    areas_for_improvement: List[str]

class EvaluationResult(BaseModel):
    status: EvaluationStatus
    evaluation_summary: EvaluationSummary
    simulation_results: SimulationResult
    development_feedback: Optional[str] = None
    decision_options: List[str]
    next_steps: List[str]
    evaluation_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ScenarioRegenerationRequest(BaseModel):
    custom_scenarios: Optional[List[str]] = None
    focus_areas: Optional[List[str]] = None

class DownloadableFile(BaseModel):
    content: Optional[str] = None
    files: Optional[Dict[str, str]] = None
    description: str

class ProjectDownloads(BaseModel):
    complete_project: DownloadableFile
    documentation: DownloadableFile
    setup_instructions: DownloadableFile
    deployment_guide: DownloadableFile
    ethical_assessment_guide: DownloadableFile
    technical_handover_package: DownloadableFile

class EvaluationPhaseData(BaseModel):
    simulation_results: Optional[SimulationResult] = None
    evaluation_result: Optional[EvaluationResult] = None
    simulation_timestamp: Optional[str] = None
    evaluation_timestamp: Optional[str] = None
    testing_method: TestingMethod
    phase_status: str = "not_started"
    completed_at: Optional[str] = None

class EvaluationStatusData(BaseModel):
    completed: bool
    phase_status: str
    has_simulation: bool
    has_evaluation: bool
    testing_method: TestingMethod
    evaluation_result: Optional[EvaluationResult] = None
    can_download: bool
    evaluation_data: Optional[Dict[str, Any]] = None

class ModificationRequest(BaseModel):
    feedback: str
    current_step: Optional[int] = None
    return_to_development: Optional[bool] = None

class ModificationResponse(BaseModel):
    success: bool
    message: str
    redirect_to_development: Optional[bool] = None

class EvaluationError(BaseModel):
    type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    project_id: Optional[str] = None
    recoverable: bool = True
    suggested_action: Optional[str] = None

class EvaluationMetrics(BaseModel):
    project_id: str
    phase: str
    start_time: str
    end_time: str
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    simulation_type: Optional[str] = None
    dataset_used: Optional[bool] = None
    file_size_mb: Optional[float] = None