from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ScoringDetail(BaseModel):
    score: int
    weight: int
    points: float
    reasoning: str

class ScoringBreakdown(BaseModel):
    privacy_score: ScoringDetail
    fairness_score: ScoringDetail
    quality_score: ScoringDetail
    humanitarian_alignment: ScoringDetail

class BiasAssessment(BaseModel):
    level: RiskLevel
    concerns: List[str]
    recommendations: List[str]

class FairnessEvaluation(BaseModel):
    representation_issues: List[str]
    recommendations: List[str]

class PrivacyEvaluation(BaseModel):
    risk_level: RiskLevel
    concerns: List[str]
    recommendations: List[str]
    assessment_reasoning: str

class EthicalAnalysis(BaseModel):
    overall_risk_level: RiskLevel
    bias_assessment: BiasAssessment
    fairness_evaluation: FairnessEvaluation
    privacy_evaluation: PrivacyEvaluation
    overall_recommendation: str
    suitability_score: int
    scoring_breakdown: Optional[ScoringBreakdown] = None

class DatasetEthicsRequest(BaseModel):
    statistics: Dict[str, Any]

class DatasetEthicsResponse(BaseModel):
    success: bool
    data: EthicalAnalysis
    message: str = ""