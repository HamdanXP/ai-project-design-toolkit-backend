from typing import Dict, Any, List
from core.llm_service import llm_service
from models.project import (
    EthicalAssessment, 
    ProjectReadinessAssessment,
    QuestionFlag,
    QuestionFlagCategory,
    AIRecommendation,
    AlternativeSolutions,
    GuidanceSource, 
    ReflectionQuestion
)
import services.project_service as project_service
import json
import logging
import asyncio
import core as ctx

logger = logging.getLogger(__name__)

class ReflectionService:
    def __init__(self):
        self.ethical_threshold = 0.6
        self.ai_appropriateness_threshold = 0.5
        self.overall_threshold = 0.6
        
        self.ethical_weight = 0.7
        self.ai_appropriateness_weight = 0.3

        self.core_questions = [
            "problem_definition",
            "target_beneficiaries", 
            "potential_harm"
        ]
    
    async def get_reflection_questions(self, project_description: str) -> Dict[str, str]:
        """Generate contextual reflection questions with dynamic selection"""
        
        context = await ctx.rag_service.get_context_for_reflection(project_description)
        
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
            
            for core_q in self.core_questions:
                if core_q not in questions_data:
                    questions_data[core_q] = self._get_default_question(core_q)
            
            if len(questions_data) < 4:
                questions_data["project_context"] = "What specific context or environment will this project operate in?"
            elif len(questions_data) > 6:
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
            
            if project.reflection_questions:
                return project.reflection_questions
            
            questions = await self.get_reflection_questions(project.description)
            
            project.reflection_questions = questions
            project.touch()
            await project.save()
            
            return project.reflection_questions
            
        except Exception as e:
            logger.error(f"Failed to get/create reflection questions: {e}")
            raise

    async def create_comprehensive_project_readiness_assessment(
        self, 
        answers: Dict[str, str], 
        project_description: str,
        questions: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create comprehensive project readiness assessment with parallel LLM calls
        """
        
        context = await ctx.rag_service.get_context_for_reflection(project_description)
        qa_context = self._build_qa_context(answers, questions)
        
        core_assessment_task = self._perform_core_assessment(
            project_description, qa_context, context
        )
        answer_quality_task = self._analyze_answer_quality(
            answers, questions, qa_context
        )
        
        assessment_data, question_flags = await asyncio.gather(
            core_assessment_task,
            answer_quality_task,
            return_exceptions=True
        )
        
        if isinstance(assessment_data, Exception):
            logger.error(f"Core assessment failed: {assessment_data}")
            assessment_data = self._get_fallback_assessment()
        
        if isinstance(question_flags, Exception):
            logger.error(f"Answer quality analysis failed: {question_flags}")
            question_flags = []
        
        alternatives = await self._generate_alternatives_if_needed(
            project_description, qa_context, assessment_data["ai_recommendation"]
        )
        
        final_assessment = await self._finalize_assessment(
            assessment_data, alternatives, answers, questions, question_flags
        )
        
        return self._create_response_objects(final_assessment)

    def _get_fallback_assessment(self) -> Dict[str, Any]:
        """Fallback assessment data in case of LLM failure"""
        return {
            "ethical_score": 0.5,
            "ethical_summary": "Assessment could not be completed due to technical issues",
            "ai_appropriateness_score": 0.5,
            "ai_appropriateness_summary": "Assessment could not be completed due to technical issues",
            "ai_recommendation": "appropriate",
            "overall_summary": "Please try again or contact support",
            "key_concerns": ["Technical assessment failure"],
            "actionable_next_steps": ["Retry assessment", "Review responses"]
        }

    async def _perform_core_assessment(
        self, 
        project_description: str, 
        qa_context: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Single comprehensive assessment
        """
        
        prompt = f"""
        Assess a humanitarian professional's readiness to develop AI projects responsibly. Focus on awareness and engagement, not expertise.

        HUMANITARIAN KNOWLEDGE BASE: {context}

        PROJECT: {project_description}

        USER RESPONSES:
        {qa_context}

        ASSESSMENT FRAMEWORK:

        ETHICAL AWARENESS (0.0-1.0):
        - Harm consideration: Do they acknowledge potential negative impacts?
        - Beneficiary awareness: Do they show consideration for affected communities?
        - Cultural sensitivity: Do they recognize importance of local context?
        - Data responsibility: Do they show awareness of privacy and consent?
        - Learning mindset: Are they open to ethical considerations?

        AI APPROPRIATENESS (0.0-1.0):
        - Problem understanding: Do they articulate why AI might help?
        - Resource awareness: Do they recognize what they need (data, skills, support)?
        - Complexity recognition: Do they understand AI projects require effort and expertise?
        - Alternative consideration: Are they open to simpler solutions?

        SCORING GUIDE:
        0.6-1.0: Shows thoughtful engagement and awareness (ready to proceed with guidance)
        0.4-0.5: Shows awareness but lacks depth in key areas - needs guided reflection before proceeding
        0.0-0.3: Minimal engagement or awareness of responsibilities

        GOOD INDICATORS: Acknowledges complexity, shows concern for beneficiaries, recognizes need for help
        CONCERNING: Dismisses ethical concerns, unrealistic expectations, no consideration of harm

        Return JSON:
        {{
            "ethical_score": 0.0,
            "ethical_summary": "Assessment of their ethical awareness and engagement",
            "ai_appropriateness_score": 0.0,
            "ai_appropriateness_summary": "Assessment of their practical thinking about AI necessity", 
            "ai_recommendation": "highly_appropriate|appropriate|questionable|not_appropriate",
            "overall_summary": "Overall readiness assessment focused on responsible learning approach",
            "key_concerns": ["Areas where they need more awareness"],
            "actionable_next_steps": ["Specific learning steps to strengthen their approach"]
        }}
        """

        response = await llm_service.analyze_text("", prompt)
        cleaned_response = self._clean_json_response(response)
        return json.loads(cleaned_response)

    async def _analyze_answer_quality(
        self, 
        answers: Dict[str, str], 
        questions: Dict[str, str],
        qa_context: str
    ) -> List[Dict[str, Any]]:
        """
        LLM-based analysis of answer quality for question flagging
        """
        
        if not answers or not questions:
            return []
            
        prompt = f"""
        Review these humanitarian project responses for quality gaps that need attention.

        QUESTIONS AND ANSWERS:
        {qa_context}

        Flag responses showing gaps in understanding or awareness:

        HIGH: Dismissive of ethics, unrealistic expectations, complete avoidance of questions
        MEDIUM: Vague responses missing humanitarian considerations, limited complexity awareness
        LOW: Good intent but needs more specificity or depth

        DON'T FLAG: Thoughtful responses, acknowledges uncertainty, shows genuine concern for beneficiaries

        Note: Brief responses can still be thoughtful - focus on engagement quality, not length.

        Return JSON array of flags for responses needing attention:
        [
            {{
                "question_key": "question_identifier",
                "issue": "Specific description of the concern",
                "severity": "low|medium|high",
                "category": "ethical"
            }}
        ]

        Return empty array [] if no responses need flagging.
        """

        response = await llm_service.analyze_text("", prompt)
        cleaned_response = self._clean_json_response(response)
        
        try:
            flags = json.loads(cleaned_response)
            return flags if isinstance(flags, list) else []
        except json.JSONDecodeError:
            return []

    async def _finalize_assessment(
        self, 
        assessment_data: Dict[str, Any],
        alternatives: Dict[str, Any] | None,
        answers: Dict[str, str],
        questions: Dict[str, str],
        question_flags: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Finalize assessment and create response structure
        """
        
        ethical_score = float(assessment_data.get("ethical_score", 0.0))
        ai_appropriateness_score = float(assessment_data.get("ai_appropriateness_score", 0.0))
        
        overall_score = (ethical_score * self.ethical_weight) + (ai_appropriateness_score * self.ai_appropriateness_weight)
        
        proceed_recommendation = (
            ethical_score >= self.ethical_threshold and
            ai_appropriateness_score >= self.ai_appropriateness_threshold and
            overall_score >= self.overall_threshold
        )
        
        return {
            "ethical_score": ethical_score,
            "ethical_summary": assessment_data.get("ethical_summary", ""),
            "ai_appropriateness_score": ai_appropriateness_score,
            "ai_appropriateness_summary": assessment_data.get("ai_appropriateness_summary", ""),
            "ai_recommendation": assessment_data.get("ai_recommendation", "appropriate"),
            "alternative_solutions": alternatives,
            "overall_readiness_score": overall_score,
            "proceed_recommendation": proceed_recommendation,
            "summary": assessment_data.get("overall_summary", ""),
            "actionable_recommendations": assessment_data.get("actionable_next_steps", []),
            "question_flags": question_flags,
            "threshold_met": overall_score >= self.overall_threshold
        }

    def _build_qa_context(self, answers: Dict[str, str], questions: Dict[str, str]) -> str:
        """Build clean Q&A context string"""
        qa_pairs = []
        for key, answer in answers.items():
            if key in questions and answer.strip():
                qa_pairs.append(f"Q: {questions[key]}\nA: {answer.strip()}")
        return "\n\n".join(qa_pairs)

    async def _generate_alternatives_if_needed(
        self, 
        project_description: str, 
        qa_context: str, 
        ai_recommendation: str
    ) -> Dict[str, Any] | None:
        """
        Generate alternatives only when AI is questionable or not appropriate
        """
        
        if ai_recommendation not in ["questionable", "not_appropriate"]:
            return None
            
        prompt = f"""
        Generate practical alternatives for this humanitarian project since AI may not be appropriate.

        PROJECT: {project_description}
        USER RESPONSES: {qa_context}

        Consider typical humanitarian constraints:
        - Limited technical expertise and resources
        - Field conditions (connectivity, infrastructure)
        - Time pressures and organizational capacity

        Provide specific, implementable alternatives for this exact project context.

        RETURN JSON:
        {{
            "digital_alternatives": ["Specific digital solution requiring no AI"],
            "process_improvements": ["Specific workflow or method improvement"],
            "non_digital_solutions": ["Specific manual/community-based approach"],
            "hybrid_approaches": ["Specific simple tech + human approach"],
            "reasoning": "Why these alternatives suit this humanitarian context better than AI"
        }}
        """

        try:
            response = await llm_service.analyze_text("", prompt)
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except Exception as e:
            logger.warning(f"Alternative generation failed: {e}")
            return None

    def _clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response"""
        response = response.strip()
        
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1:
            response = response[first_brace:last_brace + 1]
        
        response = response.replace('": "', '": "').replace('":', '":')
        
        return response.strip()

    def _create_response_objects(self, final_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Create final response objects for the API"""
        
        converted_flags = []
        for flag in final_assessment["question_flags"]:
            try:
                converted_flags.append(QuestionFlag(
                    question_key=flag.get("question_key", ""),
                    issue=flag.get("issue", ""),
                    severity=flag.get("severity", "medium"),
                    category=QuestionFlagCategory(flag.get("category", "ethical"))
                ))
            except Exception as e:
                logger.warning(f"Failed to convert question flag: {e}")

        alt_solutions_obj = None
        if final_assessment["alternative_solutions"]:
            alt_data = final_assessment["alternative_solutions"]
            alt_solutions_obj = AlternativeSolutions(
                digital_alternatives=alt_data.get("digital_alternatives", []),
                process_improvements=alt_data.get("process_improvements", []),
                non_digital_solutions=alt_data.get("non_digital_solutions", []),
                hybrid_approaches=alt_data.get("hybrid_approaches", []),
                reasoning=alt_data.get("reasoning", "")
            )

        project_readiness_assessment = ProjectReadinessAssessment(
            ethical_score=final_assessment["ethical_score"],
            ethical_summary=final_assessment["ethical_summary"],
            ai_appropriateness_score=final_assessment["ai_appropriateness_score"],
            ai_appropriateness_summary=final_assessment["ai_appropriateness_summary"],
            ai_recommendation=AIRecommendation(final_assessment["ai_recommendation"]),
            alternative_solutions=alt_solutions_obj,
            overall_readiness_score=final_assessment["overall_readiness_score"],
            proceed_recommendation=final_assessment["proceed_recommendation"],
            summary=final_assessment["summary"],
            actionable_recommendations=final_assessment["actionable_recommendations"],
            question_flags=converted_flags,
            threshold_met=final_assessment["threshold_met"]
        )

        ethical_assessment = EthicalAssessment(
            score=final_assessment["ethical_score"],
            concerns=[],
            recommendations=final_assessment["actionable_recommendations"],
            approved=final_assessment["ethical_score"] >= self.ethical_threshold
        )

        return {
            "project_readiness_assessment": project_readiness_assessment,
            "ethical_assessment": ethical_assessment,
            "ethical_score": final_assessment["ethical_score"],
            "ai_appropriateness_score": final_assessment["ai_appropriateness_score"],
            "ai_appropriateness_summary": final_assessment["ai_appropriateness_summary"],
            "ai_recommendation": final_assessment["ai_recommendation"],
            "alternative_solutions": final_assessment["alternative_solutions"],
            "overall_readiness_score": final_assessment["overall_readiness_score"],
            "proceed_recommendation": final_assessment["proceed_recommendation"],
            "summary": final_assessment["summary"],
            "actionable_recommendations": final_assessment["actionable_recommendations"],
            "question_flags": final_assessment["question_flags"],
            "threshold_met": final_assessment["threshold_met"],
            "can_proceed": final_assessment["threshold_met"]
        }
    
    async def get_reflection_questions_with_guidance(self, project_description: str) -> Dict[str, Any]:
        """Generate contextual reflection questions with targeted guidance sources"""
        
        questions_data = await self.get_reflection_questions(project_description)
        
        questions_with_guidance = {}
        
        for question_key, question_text in questions_data.items():
            try:
                logger.info(f"Getting guidance for question: {question_text[:50]}...")
                
                guidance_sources = await ctx.rag_service.get_question_specific_guidance_sources(
                    question_text=question_text,
                    question_area=question_key,
                    project_description=project_description,
                    max_sources=2
                )
                
                questions_with_guidance[question_key] = {
                    "question": question_text,
                    "guidance_sources": guidance_sources
                }
                
                logger.info(f"Found {len(guidance_sources)} relevant guidance sources for {question_key}")
                
            except Exception as e:
                logger.warning(f"Failed to get guidance for {question_key}: {e}")
                questions_with_guidance[question_key] = {
                    "question": question_text,
                    "guidance_sources": []
                }
        
        return questions_with_guidance

    async def get_or_create_reflection_questions_with_guidance(self, project_id: str) -> Dict[str, Any]:
        """Get existing reflection questions with guidance or generate new ones"""
        try:
            project = await project_service.ProjectService().get_project(project_id)
            if not project:
                raise ValueError("Project not found")
            
            if project.reflection_questions_with_guidance:
                return {
                    "questions": {
                        key: {
                            "question": enhanced_q.question,
                            "guidance_sources": [source.dict() for source in enhanced_q.guidance_sources]
                        }
                        for key, enhanced_q in project.reflection_questions_with_guidance.items()
                    },
                    "has_guidance": True
                }
            
            logger.info(f"Generating new reflection questions with guidance for project {project_id}")
            questions_with_guidance_raw = await self.get_reflection_questions_with_guidance(project.description)
            
            enhanced_questions = {}
            simple_questions = {}
            
            for key, data in questions_with_guidance_raw.items():
                guidance_sources = []
                for source_data in data.get("guidance_sources", []):
                    try:
                        guidance_sources.append(GuidanceSource(**source_data))
                    except Exception as e:
                        logger.warning(f"Failed to create GuidanceSource: {e}")
                        continue
                
                enhanced_questions[key] = ReflectionQuestion(
                    question=data["question"],
                    guidance_sources=guidance_sources
                )
                
                simple_questions[key] = data["question"]
            
            project.reflection_questions_with_guidance = enhanced_questions
            project.reflection_questions = simple_questions
            project.touch()
            await project.save()
            
            logger.info(f"Saved {len(enhanced_questions)} questions with guidance for project {project_id}")
            
            return {
                "questions": questions_with_guidance_raw,
                "has_guidance": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get/create reflection questions with guidance: {e}")
            try:
                simple_questions = await self.get_or_create_reflection_questions(project_id)
                return {
                    "questions": {
                        key: {"question": question, "guidance_sources": []}
                        for key, question in simple_questions.items()
                    },
                    "has_guidance": False
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise