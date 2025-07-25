from typing import Dict, List
from models.scoping import TechnicalInfrastructure, InfrastructureAssessment, InfrastructureScoring
from core.llm_service import llm_service
import logging
import json

logger = logging.getLogger(__name__)

class InfrastructureAssessmentService:
    """Service for assessing technical infrastructure suitability for AI projects"""
    
    def __init__(self):
        pass
    
    async def assess_infrastructure(
        self,
        project_description: str,
        problem_domain: str,
        infrastructure: TechnicalInfrastructure
    ) -> InfrastructureAssessment:
        """Assess if the technical infrastructure can support the AI project"""
        
        assessment_prompt = f"""
        You are assessing technical infrastructure for a specific humanitarian AI project. 
        Your assessment MUST be tailored to the actual requirements of this specific project.

        Project: "{project_description}"
        Domain: {problem_domain}
        
        Available Technical Infrastructure:
        - Computing Resources: {infrastructure.computing_resources}
        - Storage & Data: {infrastructure.storage_data}
        - Internet Connectivity: {infrastructure.internet_connectivity}
        - Deployment Environment: {infrastructure.deployment_environment}

        CRITICAL: Before scoring, first analyze what this specific project actually needs:

        1. ANALYZE THE PROJECT REQUIREMENTS:
        - What type of AI solution would address this problem?
        - What are the minimum technical requirements for this specific solution?
        - What deployment constraints exist in this humanitarian context?
        - Could simpler approaches work just as well?

        2. MATCH INFRASTRUCTURE TO PROJECT NEEDS:
        - Score based on how well the infrastructure supports THIS specific project
        - Consider that some projects work better with simpler, more robust infrastructure
        - Factor in field conditions and humanitarian context constraints

        Example project-specific considerations:
        - SMS-based health alerts: Mobile connectivity sufficient, cloud not needed
        - Offline crop monitoring: Mobile devices + periodic sync better than cloud dependency
        - Community education tools: Simple tablets may be more appropriate than sophisticated systems
        - Emergency response: Offline capability more valuable than high-end computing
        - Real-time epidemic tracking: Reliable connectivity essential, cloud processing needed

        Provide assessment in JSON format:
        {{
            "overall_score": 85,
            "can_proceed": true,
            "project_analysis": {{
                "ai_solution_type": "Describe the type of AI solution needed for this project",
                "minimum_requirements": "List the actual minimum technical requirements",
                "deployment_constraints": "Humanitarian field constraints for this project",
                "infrastructure_match": "How well the available infrastructure matches project needs"
            }},
            "reasoning": "Explain why this infrastructure can/cannot support THIS SPECIFIC project, considering the actual requirements vs. available resources",
            "scoring_breakdown": {{
                "computing": {{
                    "score": 35,
                    "max_score": 40,
                    "reasoning": "Explain how computing resources match THIS project's needs"
                }},
                "storage": {{
                    "score": 20,
                    "max_score": 25,
                    "reasoning": "Explain how storage matches THIS project's data requirements"
                }},
                "connectivity": {{
                    "score": 18,
                    "max_score": 20,
                    "reasoning": "Explain how connectivity matches THIS project's operational needs"
                }},
                "deployment": {{
                    "score": 12,
                    "max_score": 15,
                    "reasoning": "Explain how deployment environment suits THIS project's context"
                }}
            }},
            "recommendations": [
                "Project-specific recommendations based on the actual solution requirements",
                "Infrastructure optimizations that would benefit THIS particular project",
                "Alternative approaches that might work better with available infrastructure"
            ],
            "non_ai_alternatives": [
                "If can_proceed is false, provide alternatives specific to this project",
                "Traditional methods that could address this particular humanitarian problem",
                "Simpler technology solutions that work with available infrastructure"
            ]
        }}

        PROJECT-SPECIFIC SCORING APPROACH:
        
        Instead of fixed scores, assess based on project requirements:
        
        For SIMPLE AI projects (SMS systems, basic alerts, simple classification):
        - Mobile devices or basic hardware may score higher than cloud if more appropriate
        - Intermittent connectivity may be sufficient and score well
        - Local deployment may be preferred over cloud dependency
        
        For MODERATE AI projects (mobile image classification, basic prediction):
        - Balance between processing power and deployment simplicity
        - Periodic connectivity often sufficient
        - Hybrid approaches may score highest
        
        For COMPLEX AI projects (real-time analysis, large-scale prediction):
        - Cloud platforms and reliable connectivity essential
        - Secure storage critical for large datasets
        - Traditional scoring hierarchy applies
        
        For EMERGENCY/FIELD projects:
        - Offline capability often more valuable than computing power
        - Mobile deployment may score higher than cloud
        - Robust, simple solutions preferred
        
        HUMANITARIAN CONTEXT CONSIDERATIONS:
        - In resource-constrained environments, simpler often scores better
        - Offline-first solutions may be more appropriate than cloud-dependent ones
        - Community adoption and sustainability matter more than technical sophistication
        - Field conditions and local capacity should influence scoring
        
        Set can_proceed to true if overall score is 60 or above.
        Set can_proceed to false if overall score is below 60 or if critical infrastructure is missing.
        
        Focus on practical humanitarian context and real-world constraints.
        Consider the specific needs of humanitarian organizations including data security, field accessibility, and resource constraints.
        """
        
        response = await llm_service.analyze_text("", assessment_prompt)
        assessment_data = self._parse_assessment_response(response)
        
        return InfrastructureAssessment(
            score=assessment_data["overall_score"],
            can_proceed=assessment_data["can_proceed"],
            reasoning=assessment_data["reasoning"],
            scoring_breakdown={
                category: InfrastructureScoring(**breakdown)
                for category, breakdown in assessment_data["scoring_breakdown"].items()
            },
            recommendations=assessment_data["recommendations"],
            non_ai_alternatives=assessment_data.get("non_ai_alternatives") if not assessment_data["can_proceed"] else None,
            project_analysis=assessment_data.get("project_analysis")  # Optional field for enhanced analysis
        )
    
    def _parse_assessment_response(self, response: str) -> Dict:
        """Parse LLM response and extract assessment data"""
        try:
            cleaned_response = self._clean_json_response(response)
            assessment_data = json.loads(cleaned_response)
            
            required_fields = ["overall_score", "can_proceed", "reasoning", "scoring_breakdown", "recommendations"]
            for field in required_fields:
                if field not in assessment_data:
                    raise ValueError(f"Missing required field: {field}")
            
            required_categories = ["computing", "storage", "connectivity", "deployment"]
            for category in required_categories:
                if category not in assessment_data["scoring_breakdown"]:
                    raise ValueError(f"Missing scoring category: {category}")
                
                breakdown = assessment_data["scoring_breakdown"][category]
                if not all(key in breakdown for key in ["score", "max_score", "reasoning"]):
                    raise ValueError(f"Invalid breakdown for {category}")
            
            return assessment_data
            
        except Exception as e:
            logger.error(f"Failed to parse assessment response: {e}")
            raise ValueError("Invalid assessment response format")
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract JSON"""
        import re
        
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        start_chars = ['{', '[']
        end_chars = ['}', ']']
        
        start_idx = -1
        start_char = None
        for char in start_chars:
            idx = response.find(char)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx
                start_char = char
        
        if start_idx == -1:
            raise ValueError("No JSON found in response")
        
        end_char = '}' if start_char == '{' else ']'
        end_idx = response.rfind(end_char)
        
        if end_idx == -1:
            raise ValueError(f"No closing {end_char} found")
        
        return response[start_idx:end_idx + 1].strip()