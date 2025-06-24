from typing import Dict, Any, List
from core.llm_service import llm_service
from models.project import EvaluationResults
from models.phase import EvaluationPlan
import json
import logging
import random

logger = logging.getLogger(__name__)

class EvaluationService:
    def __init__(self):
        self.deployment_readiness_threshold = 0.8
    
    async def create_evaluation_plan(
        self,
        project_description: str,
        model_config,
        target_population: str
    ) -> EvaluationPlan:
        """Create comprehensive evaluation plan"""
        
        prompt = f"""
        Create an evaluation plan for this humanitarian AI project:
        
        Project: {project_description}
        Model: {model_config.model_type} - {model_config.algorithm}
        Target Population: {target_population}
        
        Design evaluation covering:
        1. Performance metrics appropriate for humanitarian context
        2. Bias and fairness tests
        3. Real-world simulation scenarios
        4. Ethical impact assessment
        
        Return JSON:
        {{
            "test_scenarios": [
                {{
                    "name": "Scenario Name",
                    "description": "Test description",
                    "expected_outcome": "What should happen",
                    "success_criteria": "How to measure success"
                }}
            ],
            "evaluation_metrics": ["accuracy", "precision", "recall", "fairness"],
            "bias_tests": ["demographic_parity", "equal_opportunity"],
            "simulation_parameters": {{
                "sample_size": 1000,
                "test_duration": "2_weeks",
                "monitoring_frequency": "daily"
            }}
        }}
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            plan_data = json.loads(response)
            
            return EvaluationPlan(**plan_data)
        except Exception as e:
            logger.error(f"Failed to create evaluation plan: {e}")
            return self._get_default_evaluation_plan()
    
    async def run_simulation(
        self,
        evaluation_plan: EvaluationPlan,
        model_config
    ) -> EvaluationResults:
        """Run simulated evaluation of the model"""
        
        # Simulate performance metrics
        accuracy_metrics = self._simulate_performance_metrics(model_config)
        
        # Simulate ethical metrics
        ethical_metrics = self._simulate_ethical_metrics(model_config)
        
        # Simulate bias assessment
        bias_assessment = self._simulate_bias_assessment(evaluation_plan)
        
        # Simulate performance in different scenarios
        performance_simulation = self._simulate_scenario_performance(evaluation_plan)
        
        # Determine deployment readiness
        overall_score = (
            accuracy_metrics.get("f1_score", 0) * 0.3 +
            ethical_metrics.get("fairness_score", 0) * 0.3 +
            ethical_metrics.get("transparency_score", 0) * 0.2 +
            (1 - bias_assessment.get("overall_bias_score", 1)) * 0.2
        )
        
        ready_for_deployment = overall_score >= self.deployment_readiness_threshold
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            accuracy_metrics, ethical_metrics, bias_assessment, ready_for_deployment
        )
        
        return EvaluationResults(
            accuracy_metrics=accuracy_metrics,
            ethical_metrics=ethical_metrics,
            bias_assessment=bias_assessment,
            performance_simulation=performance_simulation,
            ready_for_deployment=ready_for_deployment,
            recommendations=recommendations
        )
    
    def _simulate_performance_metrics(self, model_config) -> Dict[str, float]:
        """Simulate realistic performance metrics based on model complexity"""
        
        # Base performance varies by algorithm complexity
        algorithm = model_config.algorithm.lower()
        
        if "simple" in algorithm or "baseline" in algorithm:
            base_accuracy = 0.75
        elif "advanced" in algorithm or "neural" in algorithm:
            base_accuracy = 0.85
        else:
            base_accuracy = 0.80
        
        # Add some realistic variance
        return {
            "accuracy": base_accuracy + random.uniform(-0.05, 0.05),
            "precision": base_accuracy + random.uniform(-0.03, 0.07),
            "recall": base_accuracy + random.uniform(-0.07, 0.03),
            "f1_score": base_accuracy + random.uniform(-0.02, 0.02)
        }
    
    def _simulate_ethical_metrics(self, model_config) -> Dict[str, float]:
        """Simulate ethical performance metrics"""
        
        # Better ethics score if more guardrails are implemented
        guardrail_bonus = len(model_config.ethical_guardrails) * 0.02
        
        base_ethics = 0.75 + guardrail_bonus
        
        return {
            "fairness_score": min(1.0, base_ethics + random.uniform(-0.05, 0.05)),
            "transparency_score": min(1.0, base_ethics + random.uniform(-0.03, 0.07)),
            "accountability_score": min(1.0, base_ethics + random.uniform(-0.02, 0.02)),
            "privacy_score": min(1.0, base_ethics + random.uniform(-0.04, 0.04))
        }
    
    def _simulate_bias_assessment(self, evaluation_plan: EvaluationPlan) -> Dict[str, Any]:
        """Simulate bias assessment results"""
        
        bias_tests = evaluation_plan.bias_tests
        
        return {
            "overall_bias_score": random.uniform(0.1, 0.3),  # Lower is better
            "demographic_bias": random.uniform(0.05, 0.25),
            "geographic_bias": random.uniform(0.08, 0.22),
            "temporal_bias": random.uniform(0.03, 0.15),
            "bias_mitigation_effectiveness": random.uniform(0.7, 0.9),
            "detected_biases": [
                "Slight age-related bias detected",
                "Regional representation imbalance identified"
            ]
        }
    
    def _simulate_scenario_performance(self, evaluation_plan: EvaluationPlan) -> Dict[str, Any]:
        """Simulate performance across different scenarios"""
        
        scenarios = evaluation_plan.test_scenarios
        scenario_results = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_results[f"scenario_{i+1}"] = {
                "name": scenario.get("name", f"Scenario {i+1}"),
                "success_rate": random.uniform(0.7, 0.95),
                "response_time": random.uniform(0.5, 2.0),
                "user_satisfaction": random.uniform(0.75, 0.9),
                "issues_identified": random.randint(0, 3)
            }
        
        return scenario_results
    
    async def _generate_recommendations(
        self,
        accuracy_metrics: Dict[str, float],
        ethical_metrics: Dict[str, float],
        bias_assessment: Dict[str, Any],
        ready_for_deployment: bool
    ) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        
        recommendations = []
        
        # Performance-based recommendations
        if accuracy_metrics.get("accuracy", 0) < 0.8:
            recommendations.append("Consider additional training data or feature engineering")
        
        if accuracy_metrics.get("recall", 0) < 0.75:
            recommendations.append("Focus on reducing false negatives to better serve beneficiaries")
        
        # Ethics-based recommendations
        if ethical_metrics.get("fairness_score", 0) < 0.8:
            recommendations.append("Implement additional fairness constraints")
        
        if ethical_metrics.get("transparency_score", 0) < 0.8:
            recommendations.append("Enhance model explainability features")
        
        # Bias-based recommendations
        if bias_assessment.get("overall_bias_score", 0) > 0.2:
            recommendations.append("Conduct thorough bias mitigation before deployment")
        
        # Deployment recommendations
        if ready_for_deployment:
            recommendations.extend([
                "Model is ready for pilot deployment",
                "Implement continuous monitoring system",
                "Establish feedback collection mechanism"
            ])
        else:
            recommendations.extend([
                "Address identified issues before deployment",
                "Consider additional testing phases",
                "Engage with stakeholders for feedback"
            ])
        
        return recommendations
    
    def _get_default_evaluation_plan(self) -> EvaluationPlan:
        """Fallback default evaluation plan"""
        return EvaluationPlan(
            test_scenarios=[
                {
                    "name": "Normal Operation",
                    "description": "Test under typical operating conditions",
                    "expected_outcome": "Accurate predictions",
                    "success_criteria": "85% accuracy"
                }
            ],
            evaluation_metrics=["accuracy", "precision", "recall", "f1_score"],
            bias_tests=["demographic_parity", "equal_opportunity"],
            simulation_parameters={
                "sample_size": 500,
                "test_duration": "1_week",
                "monitoring_frequency": "daily"
            }
        )