from typing import Dict, Any, List
from models.scoping import DataSuitabilityAssessment
import logging

logger = logging.getLogger(__name__)

class FinalScopingService:
    """Simplified service for final scoping phase assessment"""
    
    def __init__(self):
        # Humanitarian-focused scoring weights (matching frontend)
        self.suitability_weights = {
            "privacy_ethics": 0.35,           # 35% - Safety and ethics first
            "population_representativeness": 0.30,  # 30% - Core to humanitarian impact
            "data_completeness": 0.20,        # 20% - Important but manageable
            "quality_sufficiency": 0.15       # 15% - Often workable
        }
    
    def assess_data_suitability(
        self, 
        assessment: DataSuitabilityAssessment
    ) -> Dict[str, Any]:
        """Simplified data suitability assessment"""
        
        # Calculate component scores
        component_scores = {
            "data_completeness": self._score_data_completeness(assessment.data_completeness.value),
            "population_representativeness": self._score_representativeness(assessment.population_representativeness.value),
            "privacy_ethics": self._score_privacy_ethics(assessment.privacy_ethics.value),
            "quality_sufficiency": self._score_quality_sufficiency(assessment.quality_sufficiency.value)
        }
        
        # Calculate overall suitability score
        overall_score = sum(
            component_scores[component] * self.suitability_weights[component]
            for component in self.suitability_weights
        )
        
        percentage = int(overall_score * 100)
        suitability_level = self._get_suitability_level(overall_score)
        
        # Generate simple recommendations
        recommendations = self._generate_recommendations(component_scores, assessment)
        
        return {
            "overall_score": round(overall_score, 2),
            "percentage": percentage,
            "suitability_level": suitability_level,
            "component_scores": {k: int(v * 100) for k, v in component_scores.items()},
            "recommendations": recommendations,
            "summary": self._generate_summary(percentage, suitability_level)
        }
    
    def generate_final_summary(
        self,
        project_description: str,
        use_case: Dict[str, Any],
        dataset: Dict[str, Any],
        feasibility_score: int,
        feasibility_level: str,  # Changed from risk to level
        data_suitability_score: int
    ) -> Dict[str, Any]:
        """Generate simplified final project summary"""
        
        # Calculate overall readiness (weighted average)
        overall_readiness = int(feasibility_score * 0.6 + data_suitability_score * 0.4)
        
        # Determine if ready to proceed based on feasibility level and data suitability
        ready_to_proceed = (
            feasibility_level in ['high', 'medium'] and 
            overall_readiness >= 60 and 
            data_suitability_score >= 50
        )
        
        # Generate simple summary text
        if ready_to_proceed:
            summary_text = f"Your project assessment is complete with {overall_readiness}% overall readiness ({feasibility_level} feasibility). You can proceed to the development phase with confidence."
        else:
            summary_text = f"Your project assessment shows {overall_readiness}% readiness ({feasibility_level} feasibility). Consider addressing identified challenges before proceeding to development."
        
        return {
            "overall_readiness_score": overall_readiness,
            "feasibility_score": feasibility_score,
            "feasibility_level": feasibility_level,
            "data_suitability_score": data_suitability_score,
            "ready_to_proceed": ready_to_proceed,
            "summary": summary_text,
            "recommendation": "proceed" if ready_to_proceed else "revise_approach",
            "next_phase": "development" if ready_to_proceed else "scoping_revision"
        }
    
    def _score_data_completeness(self, response: str) -> float:
        """Score data completeness assessment (humanitarian context)"""
        scores = {
            "looks_clean": 1.0,      # Excellent
            "some_issues": 0.7,      # Common and fixable
            "many_problems": 0.3     # Significant work needed
        }
        return scores.get(response, 0.5)
    
    def _score_representativeness(self, response: str) -> float:
        """Score population representativeness (critical for humanitarian work)"""
        scores = {
            "representative": 1.0,    # Excellent coverage
            "partially": 0.6,        # Workable but limited
            "limited_coverage": 0.2   # Major humanitarian concern
        }
        return scores.get(response, 0.5)
    
    def _score_privacy_ethics(self, response: str) -> float:
        """Score privacy and ethics assessment (safety first)"""
        scores = {
            "privacy_safe": 1.0,     # Safe to proceed
            "need_review": 0.4,      # Concerning - needs attention
            "high_risk": 0.0         # Unacceptable - must address
        }
        return scores.get(response, 0.5)
    
    def _score_quality_sufficiency(self, response: str) -> float:
        """Score quality and sufficiency (often workable)"""
        scores = {
            "sufficient": 1.0,       # Good quality and volume
            "borderline": 0.6,       # Might work with limitations
            "insufficient": 0.2      # Need alternatives
        }
        return scores.get(response, 0.5)
    
    def _get_suitability_level(self, score: float) -> str:
        """Convert score to suitability level (humanitarian thresholds)"""
        if score >= 0.7:
            return "excellent"
        elif score >= 0.5:
            return "good"
        elif score >= 0.3:
            return "moderate"
        else:
            return "poor"
    
    def _generate_recommendations(
        self,
        component_scores: Dict[str, float],
        assessment: DataSuitabilityAssessment
    ) -> List[str]:
        """Generate simple recommendations based on assessment"""
        
        recommendations = []
        
        # Data completeness recommendations
        if component_scores["data_completeness"] < 0.7:
            if assessment.data_completeness.value == "some_issues":
                recommendations.append("Plan for data cleaning and preprocessing before model development")
            elif assessment.data_completeness.value == "many_problems":
                recommendations.append("Consider seeking alternative data sources or extensive data cleaning")
        
        # Representativeness recommendations
        if component_scores["population_representativeness"] < 0.7:
            if assessment.population_representativeness.value == "partially":
                recommendations.append("Consider supplementing with additional data to improve population coverage")
            elif assessment.population_representativeness.value == "limited_coverage":
                recommendations.append("Limit project scope to populations well-represented in your data")
        
        # Privacy/ethics recommendations
        if component_scores["privacy_ethics"] < 0.8:
            if assessment.privacy_ethics.value == "need_review":
                recommendations.append("Conduct formal ethical review and implement privacy safeguards")
            elif assessment.privacy_ethics.value == "high_risk":
                recommendations.append("Seek expert consultation on data ethics before proceeding")
        
        # Quality/sufficiency recommendations
        if component_scores["quality_sufficiency"] < 0.7:
            if assessment.quality_sufficiency.value == "borderline":
                recommendations.append("Consider starting with a smaller pilot project to validate data adequacy")
            elif assessment.quality_sufficiency.value == "insufficient":
                recommendations.append("Gather additional data or reduce project scope significantly")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Your data appears suitable for AI development - proceed with confidence")
        
        return recommendations
    
    def _generate_summary(self, percentage: int, level: str) -> str:
        """Generate humanitarian-focused suitability summary"""
        
        if level == "excellent":
            return f"Your data is ready for AI development ({percentage}%). Strong foundations for humanitarian impact."
        elif level == "good":
            return f"Your data shows good suitability ({percentage}%) with some preparation needed."
        elif level == "moderate":
            return f"Your data has potential ({percentage}%) but requires significant preparation before AI development."
        else:
            return f"Your data needs fundamental improvements ({percentage}%) before it can be used safely for humanitarian AI."