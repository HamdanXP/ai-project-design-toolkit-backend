from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ReflectionResponse(BaseModel):
    answers: Dict[str, str]
    ai_analysis: Optional[str] = None
    ethical_score: Optional[float] = None
    proceed_recommendation: bool = False
    concerns: List[str] = []

class ScopingRequest(BaseModel):
    project_description: str
    problem_domain: str
    target_population: str
    deployment_context: str

class ScopingResponse(BaseModel):
    suggested_use_cases: List[Dict[str, Any]]
    recommended_datasets: List[Dict[str, Any]]
    feasibility_assessment: Dict[str, Any]
    technical_requirements: List[str]
    
class EvaluationPlan(BaseModel):
    test_scenarios: List[Dict[str, Any]]
    evaluation_metrics: List[str]
    bias_tests: List[str]
    simulation_parameters: Dict[str, Any]
