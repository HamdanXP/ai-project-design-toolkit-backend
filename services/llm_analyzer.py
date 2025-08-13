from typing import Dict, Any, List, Optional
from core.llm_service import llm_service
import json
import logging

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Service for LLM-powered analysis and content generation"""
    
    def __init__(self):
        self.llm = llm_service
    
    async def analyze_project_description(self, description: str) -> Dict[str, Any]:
        """Analyze project description and extract key information"""
        prompt = f"""
        Analyze this humanitarian AI project description and extract key information:
        
        Description: "{description}"
        
        Extract and return JSON with:
        {{
            "domain": "primary humanitarian domain (health, education, disaster, etc.)",
            "target_population": "who this project aims to help",
            "ai_techniques": ["list", "of", "relevant", "ai", "techniques"],
            "urgency": "low/medium/high", 
            "data_requirements": ["types", "of", "data", "needed"],
            "potential_challenges": ["list", "of", "challenges"],
            "success_indicators": ["how", "to", "measure", "success"]
        }}
        """
        
        try:
            response = await self.llm.analyze_text(description, prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to analyze project description: {e}")
            return self._get_default_analysis()
    
    async def generate_ethical_concerns(
        self, 
        project_description: str, 
        target_population: str
    ) -> List[str]:
        """Generate potential ethical concerns for a project"""
        prompt = f"""
        Identify potential ethical concerns for this humanitarian AI project:
        
        Project: {project_description}
        Target Population: {target_population}
        
        Consider concerns related to:
        - Data privacy and consent
        - Algorithmic bias and fairness
        - Cultural sensitivity
        - Power dynamics
        - Transparency and accountability
        - Potential for harm or misuse
        
        Return a JSON array of specific ethical concerns:
        ["concern1", "concern2", "concern3"]
        """
        
        try:
            response = await self.llm.analyze_text("", prompt)
            concerns = json.loads(response)
            return concerns if isinstance(concerns, list) else []
        except Exception as e:
            logger.error(f"Failed to generate ethical concerns: {e}")
            return [
                "Potential for algorithmic bias affecting vulnerable populations",
                "Privacy concerns with sensitive humanitarian data",
                "Need for cultural sensitivity in AI design",
                "Ensuring transparency in AI decision-making"
            ]
    
    async def assess_dataset_relevance(
        self,
        dataset_description: str,
        project_description: str,
        use_case: str
    ) -> Dict[str, Any]:
        """Assess how relevant a dataset is for a project"""
        prompt = f"""
        Assess the relevance of this dataset for the humanitarian AI project:
        
        Project: {project_description}
        Use Case: {use_case}
        Dataset: {dataset_description}
        
        Evaluate and return JSON:
        {{
            "relevance_score": 0.8,
            "strengths": ["what makes this dataset good for the project"],
            "limitations": ["potential issues or gaps"],
            "preprocessing_needs": ["what preparation would be needed"],
            "ethical_considerations": ["privacy, consent, bias concerns"],
            "recommendation": "highly_recommended/recommended/consider/not_recommended"
        }}
        """
        
        try:
            response = await self.llm.analyze_text("", prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to assess dataset relevance: {e}")
            return {
                "relevance_score": 0.5,
                "strengths": ["May contain relevant information"],
                "limitations": ["Needs further evaluation"],
                "preprocessing_needs": ["Data cleaning and validation"],
                "ethical_considerations": ["Standard privacy protections needed"],
                "recommendation": "consider"
            }
    
    async def generate_implementation_guidance(
        self,
        project_description: str,
        selected_approach: Dict[str, Any],
        deployment_environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate implementation guidance for selected approach"""
        prompt = f"""
        Generate implementation guidance for this humanitarian AI project:
        
        Project: {project_description}
        Selected Approach: {json.dumps(selected_approach)}
        Deployment Environment: {json.dumps(deployment_environment)}
        
        Provide detailed guidance in JSON format:
        {{
            "implementation_steps": [
                {{
                    "step": 1,
                    "title": "Data Preparation",
                    "description": "Detailed description",
                    "estimated_time": "2-3 weeks",
                    "key_considerations": ["important points to remember"]
                }}
            ],
            "technical_requirements": ["required tools, libraries, infrastructure"],
            "team_roles": ["data scientist", "domain expert", "etc"],
            "timeline_estimate": "overall project timeline",
            "success_metrics": ["how to measure progress"],
            "common_pitfalls": ["what to avoid"],
            "next_steps": ["immediate actions to take"]
        }}
        """
        
        try:
            response = await self.llm.analyze_text("", prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to generate implementation guidance: {e}")
            return self._get_default_implementation_guidance()
    
    async def summarize_evaluation_results(
        self,
        evaluation_results: Dict[str, Any],
        project_context: str
    ) -> str:
        """Generate human-readable summary of evaluation results"""
        prompt = f"""
        Create a clear, non-technical summary of these AI evaluation results for humanitarian stakeholders:
        
        Project Context: {project_context}
        Evaluation Results: {json.dumps(evaluation_results)}
        
        Write a summary that explains:
        - Overall performance in plain language
        - Key strengths and areas for improvement
        - Readiness for deployment
        - Specific recommendations for next steps
        - Any ethical concerns identified
        
        Keep the language accessible for non-technical humanitarian professionals.
        """
        
        try:
            response = await self.llm.analyze_text("", prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to summarize evaluation results: {e}")
            return "Evaluation completed. Please review detailed results for specific metrics and recommendations."
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Default analysis when LLM analysis fails"""
        return {
            "domain": "general_humanitarian",
            "target_population": "humanitarian_beneficiaries",
            "ai_techniques": ["machine_learning", "data_analysis"],
            "urgency": "medium",
            "data_requirements": ["structured_data", "historical_records"],
            "potential_challenges": ["data_quality", "resource_constraints"],
            "success_indicators": ["improved_outcomes", "user_satisfaction"]
        }
    
    def _get_default_implementation_guidance(self) -> Dict[str, Any]:
        """Default implementation guidance when LLM generation fails"""
        return {
            "implementation_steps": [
                {
                    "step": 1,
                    "title": "Project Setup",
                    "description": "Set up development environment and gather requirements",
                    "estimated_time": "1-2 weeks",
                    "key_considerations": ["Establish clear project goals"]
                },
                {
                    "step": 2,
                    "title": "Data Collection",
                    "description": "Gather and prepare training data",
                    "estimated_time": "2-4 weeks", 
                    "key_considerations": ["Ensure data quality and ethical compliance"]
                }
            ],
            "technical_requirements": ["Python environment", "Machine learning libraries"],
            "team_roles": ["Project manager", "Data scientist", "Domain expert"],
            "timeline_estimate": "3-6 months",
            "success_metrics": ["Model accuracy", "User adoption"],
            "common_pitfalls": ["Insufficient data", "Lack of stakeholder engagement"],
            "next_steps": ["Begin data collection", "Set up development environment"]
        }

