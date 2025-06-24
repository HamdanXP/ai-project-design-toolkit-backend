from typing import Dict, Any, List
from core.llm_service import llm_service
from services.rag_service import rag_service
from models.project import EthicalAssessment
import services.project_service as project_service
import json
import logging

logger = logging.getLogger(__name__)

class ReflectionService:
    def __init__(self):
        self.ethical_threshold = 0.7
        # Core questions that are always relevant
        self.core_questions = [
            "problem_definition",
            "target_beneficiaries", 
            "potential_harm"
        ]
    
    async def get_reflection_questions(self, project_description: str) -> Dict[str, str]:
        """Generate contextual reflection questions with dynamic selection"""
        
        # Get context from knowledge base
        context = await rag_service.get_context_for_reflection(project_description)
        
        prompt = f"""
        Context from humanitarian AI best practices:
        {context}
        
        Based on this humanitarian AI project description: "{project_description}"
        
        Generate 4-6 reflection questions that are most relevant for this specific project.
        
        ALWAYS INCLUDE these 3 core questions (customize them for this project):
        1. problem_definition - What specific problem is this project solving?
        2. target_beneficiaries - Who will benefit and how?
        3. potential_harm - What negative impacts could this project cause?
        
        THEN SELECT 1-3 additional questions from these areas based on project relevance:
        
        • data_availability - For projects involving data collection, ML, or data processing
        • stakeholder_involvement - For community-centered or participatory projects  
        • cultural_sensitivity - For cross-cultural or vulnerable population projects
        • resource_constraints - For projects with implementation or resource challenges
        • technical_feasibility - For technically complex projects
        • success_metrics - For projects requiring measurable impact assessment
        • sustainability - For long-term projects requiring ongoing support
        • privacy_security - For projects handling sensitive personal data
        
        Create specific, contextual questions tailored to this project. 
        Use clear, direct language that helps users reflect on their specific situation.
        
        Return ONLY a JSON object with exactly these keys (no additional text):
        {{
            "problem_definition": "your project-specific question here",
            "target_beneficiaries": "your project-specific question here", 
            "potential_harm": "your project-specific question here",
            "most_relevant_additional_key": "your project-specific question here",
            "another_relevant_key_if_needed": "your project-specific question here"
        }}
        
        Use the exact key names listed above for additional questions.
        """
        
        try:
            response = await llm_service.analyze_text(project_description, prompt)
            questions_data = json.loads(response)
            
            # Simple validation - ensure core questions exist
            for core_q in self.core_questions:
                if core_q not in questions_data:
                    questions_data[core_q] = self._get_default_question(core_q)
            
            # Ensure proper question count
            if len(questions_data) < 4:
                # Add a generic fourth question
                questions_data["project_context"] = "What specific context or environment will this project operate in?"
            elif len(questions_data) > 6:
                # Keep core questions + first 3 additional
                core_data = {k: v for k, v in questions_data.items() if k in self.core_questions}
                additional_data = {k: v for k, v in questions_data.items() if k not in self.core_questions}
                limited_additional = dict(list(additional_data.items())[:3])
                questions_data = {**core_data, **limited_additional}
            
            return questions_data
            
        except Exception as e:
            logger.error(f"Failed to generate reflection questions: {e}")
            return self._get_default_questions()

    def _get_default_question(self, question_type: str) -> str:
        """Get default question text for a given type"""
        defaults = {
            "problem_definition": "What specific humanitarian problem are you trying to solve with AI?",
            "target_beneficiaries": "Who are the primary beneficiaries and how were they identified?",
            "potential_harm": "What potential negative impacts could this AI system have?",
            "data_availability": "What data do you have access to and how was it collected?",
            "stakeholder_involvement": "How are affected communities involved in the design process?",
            "cultural_sensitivity": "How does your solution account for local cultural contexts?",
            "resource_constraints": "What are your main resource limitations (technical, financial, human)?",
            "technical_feasibility": "What technical challenges do you anticipate and how will you address them?",
            "success_metrics": "How will you measure the success and impact of your AI solution?",
            "sustainability": "How will you ensure the long-term sustainability of this solution?",
            "privacy_security": "How will you protect user privacy and data security?",
            "project_context": "What specific context or environment will this project operate in?"
        }
        return defaults.get(question_type, f"Please describe your approach to {question_type.replace('_', ' ')}")
    
    def _get_default_questions(self) -> Dict[str, str]:
        """Fallback default questions (4 core questions)"""
        return {
            "problem_definition": self._get_default_question("problem_definition"),
            "target_beneficiaries": self._get_default_question("target_beneficiaries"),
            "potential_harm": self._get_default_question("potential_harm"),
            "project_context": self._get_default_question("project_context")
        }
    async def get_or_create_reflection_questions(self, project_id: str) -> Dict[str, str]:
        """Get existing reflection questions or generate new ones"""
        try:
            project = await project_service.ProjectService().get_project(project_id)
            if not project:
                raise ValueError("Project not found")
            
            # Return existing questions if they exist
            if project.reflection_questions:
                return project.reflection_questions
            
            # Generate new questions if none exist
            questions = await self.get_reflection_questions(project.description)
            
            # Save questions to project
            project.reflection_questions = questions
            project.touch()
            await project.save()
            
            return project.reflection_questions
            
        except Exception as e:
            logger.error(f"Failed to get/create reflection questions: {e}")
            raise

    async def create_comprehensive_ethical_assessment(
        self, 
        answers: Dict[str, str], 
        project_description: str,
        questions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create comprehensive ethical assessment with dynamic question analysis"""

        # Get context from knowledge base
        context = await rag_service.get_context_for_reflection(project_description)
        
        # Build questions and answers context
        qa_context = ""
        for key, answer in answers.items():
            if key in questions:
                qa_context += f"Q: {questions[key]}\nA: {answer}\n\n"
        
        prompt = f"""
        Context from humanitarian AI best practices:
        {context}

        You are evaluating reflection responses for a humanitarian AI project.
        
        Project: {project_description}
        
        Questions and Responses:
        {qa_context}
        
        CRITICAL SCORING GUIDELINES:
        - Gibberish, nonsensical, or extremely vague answers should receive 0.0-0.2 scores
        - Answers that completely avoid the question should be scored 0.1-0.3
        - Superficial answers without substance should be scored 0.2-0.4
        - Answers showing some understanding but major gaps should be scored 0.4-0.6
        - Good answers with minor issues should be scored 0.6-0.8
        - Excellent, comprehensive answers should be scored 0.8-1.0
        
        ANSWER QUALITY EVALUATION:
        - Does each answer directly address the question asked?
        - Are answers specific and detailed rather than vague platitudes?
        - Do answers demonstrate understanding of ethical AI principles?
        - Are potential risks and challenges acknowledged realistically?
        - Is there evidence of thoughtful consideration rather than copy-paste responses?
        
        If most answers are poor quality, uninformative, or nonsensical, the overall score should be very low (0.0-0.3).
        
        Provide a comprehensive but concise ethical assessment:
        
        1. **Ethical Score (0-1)**: Overall ethical readiness based on answer quality and content
        2. **Actionable Recommendations**: Specific next steps for this project
        3. **Question Quality**: Flag ALL answers that are unclear, vague, or problematic
        4. **Decision**: Should this project proceed as planned?
        5. **Summary**: Honest assessment of response quality and ethical readiness
        
        Be honest and strict in your evaluation - poor answers indicate the person is not ready to proceed ethically.
        
        Return JSON in this exact format (no markdown formatting, no code blocks):
        {{
            "ethical_score": 0.15,
            "proceed_recommendation": false,
            "summary": "Honest assessment focusing on actual response quality and ethical preparedness",
            "actionable_recommendations": [
                "Specific action for this project based on the gaps identified",
                "Another specific action to address deficiencies"
            ],
            "question_flags": [
                {{
                    "question_key": "question_key_here",
                    "issue": "Specific issue with this answer (e.g., 'Answer is gibberish and does not address the question')",
                    "severity": "high"
                }}
            ]
        }}
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            assessment_data = json.loads(response)
            
            # Additional validation - if most questions are flagged as high severity, force low score
            question_flags = assessment_data.get("question_flags", [])
            high_severity_count = sum(1 for flag in question_flags if flag.get("severity") == "high")
            total_questions = len(questions)
            
            ethical_score = assessment_data.get("ethical_score", 0.0)
            
            # Force lower score if too many high-severity issues
            if high_severity_count >= total_questions * 0.6:  # 60% or more questions flagged as high severity
                ethical_score = min(ethical_score, 0.3)  # Cap at 30%
                assessment_data["proceed_recommendation"] = False
                
            if high_severity_count >= total_questions * 0.8:  # 80% or more questions flagged
                ethical_score = min(ethical_score, 0.2)  # Cap at 20%
                
            if high_severity_count == total_questions:  # All questions flagged
                ethical_score = min(ethical_score, 0.15)  # Cap at 15%
            
            # Create the EthicalAssessment object for database storage
            ethical_assessment = EthicalAssessment(
                score=ethical_score,
                concerns=[],  # No longer using separate concerns
                recommendations=assessment_data.get("actionable_recommendations", []),
                approved=ethical_score >= self.ethical_threshold
            )
            
            # Return updated data for frontend
            return {
                "ethical_assessment": ethical_assessment,
                "ethical_score": ethical_score,
                "proceed_recommendation": assessment_data.get("proceed_recommendation", False),
                "summary": assessment_data.get("summary", ""),
                "actionable_recommendations": assessment_data.get("actionable_recommendations", []),
                "question_flags": question_flags,
                "threshold_met": ethical_score >= self.ethical_threshold,
                "can_proceed": ethical_score >= self.ethical_threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to create ethical assessment: {e}")
            # Return fallback assessment
            fallback_assessment = EthicalAssessment(
                score=0.0,  # Changed from 0.5 to 0.0 for failed assessments
                concerns=[],
                recommendations=["Complete all reflection questions with detailed, thoughtful responses", "Consult with ethics experts"],
                approved=False
            )
            return {
                "ethical_assessment": fallback_assessment,
                "ethical_score": 0.0,
                "proceed_recommendation": False,
                "summary": "Assessment failed - all questions must be completed with meaningful responses",
                "actionable_recommendations": ["Provide detailed, thoughtful answers to all reflection questions"],
                "question_flags": [],
                "threshold_met": False,
                "can_proceed": False
            }
    
    def _get_default_question(self, question_type: str) -> str:
        """Get default question text for a given type"""
        defaults = {
            "problem_definition": "What specific humanitarian problem are you trying to solve with AI?",
            "target_beneficiaries": "Who are the primary beneficiaries and how were they identified?",
            "potential_harm": "What potential negative impacts could this AI system have?",
            "data_availability": "What data do you have access to and how was it collected?",
            "resource_constraints": "What are your main resource limitations (technical, financial, human)?",
            "success_metrics": "How will you measure the success and impact of your AI solution?",
            "stakeholder_involvement": "How are affected communities involved in the design process?",
            "cultural_sensitivity": "How does your solution account for local cultural contexts?",
            "technical_feasibility": "What technical challenges do you anticipate and how will you address them?",
            "sustainability": "How will you ensure the long-term sustainability of this solution?",
            "privacy_security": "How will you protect user privacy and data security?"
        }
        return defaults.get(question_type, f"Please describe your approach to {question_type.replace('_', ' ')}")
    
    def _get_default_questions(self) -> Dict[str, str]:
        """Fallback default questions (4 core questions)"""
        return {
            "problem_definition": self._get_default_question("problem_definition"),
            "target_beneficiaries": self._get_default_question("target_beneficiaries"),
            "potential_harm": self._get_default_question("potential_harm"),
            "data_availability": self._get_default_question("data_availability")
        }