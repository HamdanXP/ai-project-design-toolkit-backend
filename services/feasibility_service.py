from typing import Dict, Any, List
from models.project import DeploymentEnvironment
import logging

logger = logging.getLogger(__name__)

class FeasibilityService:
    """Simplified feasibility assessment service for humanitarian toolkit"""
    
    def __init__(self):
        # Simplified scoring weights
        self.category_weights = {
            "budget": 0.30,
            "time": 0.25, 
            "ai_experience": 0.25,
            "technical_skills": 0.15,
            "infrastructure": 0.05
        }
    
    def calculate_simple_feasibility(
        self, 
        deployment_env: DeploymentEnvironment
    ) -> Dict[str, Any]:
        """Calculate simple feasibility score based on deployment environment"""
        
        # Calculate individual scores
        budget_score = self._score_budget(deployment_env.project_budget)
        time_score = self._score_timeline(deployment_env.project_timeline)
        ai_exp_score = self._score_ai_experience(deployment_env.ai_ml_experience)
        tech_score = self._score_technical_skills(deployment_env.technical_skills)
        infra_score = self._score_infrastructure(
            deployment_env.reliable_internet_connection,
            deployment_env.local_technology_setup
        )
        
        # Calculate weighted overall score
        overall_score = (
            budget_score * self.category_weights["budget"] +
            time_score * self.category_weights["time"] +
            ai_exp_score * self.category_weights["ai_experience"] +
            tech_score * self.category_weights["technical_skills"] +
            infra_score * self.category_weights["infrastructure"]
        )
        
        # Convert to percentage
        overall_percentage = int(overall_score * 100)
        
        # Determine feasibility level
        feasibility_level = self._get_feasibility_level(overall_score)
        
        # Identify key constraints
        key_constraints = self._identify_constraints(deployment_env)
        
        # Generate simple summary
        summary = self._generate_summary(overall_percentage, feasibility_level)
        
        return {
            "overall_percentage": overall_percentage,
            "feasibility_level": feasibility_level,  # Changed from previous risk terminology
            "summary": summary,
            "key_constraints": key_constraints,
            "category_scores": {
                "budget": int(budget_score * 100),
                "time": int(time_score * 100),
                "ai_experience": int(ai_exp_score * 100),
                "technical_skills": int(tech_score * 100),
                "infrastructure": int(infra_score * 100)
            }
        }
    
    def _score_budget(self, budget: str) -> float:
        """Score budget constraint"""
        budget_scores = {
            "unlimited": 1.0,
            "over_200k": 0.9,
            "50k_200k": 0.75,
            "10k_50k": 0.5,
            "under_10k": 0.25
        }
        return budget_scores.get(budget, 0.5)
    
    def _score_timeline(self, timeline: str) -> float:
        """Score timeline constraint"""
        timeline_scores = {
            "ongoing": 1.0,
            "over_1_year": 0.9,
            "6_12_months": 0.7,
            "3_6_months": 0.5,
            "1_3_months": 0.3
        }
        return timeline_scores.get(timeline, 0.5)
    
    def _score_ai_experience(self, experience: str) -> float:
        """Score AI/ML experience"""
        exp_scores = {
            "expert_level": 1.0,
            "previous_projects": 0.8,
            "some_courses": 0.4,
            "none": 0.1
        }
        return exp_scores.get(experience, 0.5)
    
    def _score_technical_skills(self, skills: str) -> float:
        """Score technical skills"""
        skill_scores = {
            "professional_developers": 1.0,
            "strong_tech": 0.8,
            "basic_programming": 0.5,
            "limited": 0.2
        }
        return skill_scores.get(skills, 0.5)
    
    def _score_infrastructure(self, internet: bool, local_tech: bool) -> float:
        """Score infrastructure readiness"""
        if internet and local_tech:
            return 1.0
        elif internet or local_tech:
            return 0.6
        else:
            return 0.2
    
    def _get_feasibility_level(self, score: float) -> str:
        """Convert score to feasibility level"""
        if score >= 0.75:
            return "high"
        elif score >= 0.50:
            return "medium"
        else:
            return "low"
    
    def _identify_constraints(self, env: DeploymentEnvironment) -> List[str]:
        """Identify key constraints from deployment environment"""
        constraints = []
        
        if env.project_budget in ["under_10k", "10k_50k"]:
            constraints.append("Limited Budget")
        
        if env.ai_ml_experience in ["none", "some_courses"]:
            constraints.append("Limited AI Experience")
        
        if env.project_timeline in ["1_3_months", "3_6_months"]:
            constraints.append("Tight Timeline")
        
        if env.technical_skills in ["limited", "basic_programming"]:
            constraints.append("Technical Skills Gap")
        
        if not env.reliable_internet_connection:
            constraints.append("Connectivity Issues")
        
        if env.stakeholder_buy_in == "low":
            constraints.append("Low Stakeholder Support")
        
        return constraints[:3]  # Return top 3 constraints
    
    def _generate_summary(self, percentage: int, level: str) -> str:
        """Generate human-readable summary"""
        if level == "high":
            return f"Your project shows strong feasibility ({percentage}%). You have good foundations to proceed with confidence."
        elif level == "medium":
            return f"Your project has moderate feasibility ({percentage}%). Address key constraints before proceeding to development."
        else:
            return f"Your project faces significant challenges ({percentage}%). Consider strengthening foundations before starting development."