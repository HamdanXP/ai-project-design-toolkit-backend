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
import core as ctx

logger = logging.getLogger(__name__)

class ReflectionService:
    def __init__(self):
        self.ethical_threshold = 0.7
        self.ai_appropriateness_threshold = 0.6
        self.overall_threshold = 0.65
        # Core questions that are always relevant
        self.core_questions = [
            "problem_definition",
            "target_beneficiaries", 
            "potential_harm"
        ]
    
    async def get_reflection_questions(self, project_description: str) -> Dict[str, str]:
        """Generate contextual reflection questions with dynamic selection"""
        
        # Get context from knowledge base
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

    async def create_comprehensive_project_readiness_assessment(
        self, 
        answers: Dict[str, str], 
        project_description: str,
        questions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create comprehensive project readiness assessment combining ethical and AI appropriateness evaluation"""

        # Get context from knowledge base
        context = await ctx.rag_service.get_context_for_reflection(project_description)
        
        # Build questions and answers context
        qa_context = ""
        for key, answer in answers.items():
            if key in questions:
                qa_context += f"Q: {questions[key]}\nA: {answer}\n\n"
        
        prompt = f"""
        Context from humanitarian AI best practices:
        {context}

        You are evaluating a humanitarian AI project for both ETHICAL READINESS and AI APPROPRIATENESS.
        
        Project: {project_description}
        
        Questions and Responses:
        {qa_context}
        
        EVALUATION FRAMEWORK:
        
        1. ETHICAL ASSESSMENT (Score 0-1):
        - Potential harm to beneficiaries and mitigation strategies
        - Cultural sensitivity and community involvement
        - Privacy, consent, and data protection measures
        - Transparency and accountability mechanisms
        - Bias prevention and fairness considerations
        
        2. AI APPROPRIATENESS ASSESSMENT (Score 0-1):
        - Does this problem genuinely benefit from AI over simpler solutions?
        - Is the technical complexity justified by the expected benefits?
        - Are there sufficient data and resources for AI implementation?
        - Could simpler digital or non-digital solutions be more effective?
        - Is the team prepared for AI development and deployment?
        
        CRITICAL SCORING GUIDELINES:
        - Gibberish, nonsensical, or extremely vague answers should receive 0.0-0.3 scores
        - Answers that completely avoid questions should be scored 0.1-0.3
        - Superficial answers without substance should be scored 0.3-0.5
        - Good answers with minor issues should be scored 0.6-0.8
        - Excellent, comprehensive answers should be scored 0.8-1.0
        
        ALTERNATIVE SOLUTIONS:
        If AI appropriateness score is below 0.6, suggest specific alternatives:
        - Digital alternatives (apps, databases, web platforms)
        - Process improvements (workflows, coordination, training)
        - Non-digital solutions (manual processes, community approaches)
        - Hybrid approaches (AI-assisted rather than AI-driven)
        
        IMPORTANT: Return ONLY valid JSON in the exact format below. No additional text, markdown, or explanations.
        
        Return JSON in this exact format:
        {{
            "ethical_score": 0.75,
            "ethical_summary": "Brief assessment of ethical readiness",
            "ai_appropriateness_score": 0.65,
            "ai_appropriateness_summary": "Brief assessment of AI suitability for this problem",
            "ai_recommendation": "highly_appropriate|appropriate|questionable|not_appropriate",
            "alternative_solutions": {{
                "digital_alternatives": ["Specific alternative 1", "Alternative 2"],
                "process_improvements": ["Process improvement 1"],
                "non_digital_solutions": ["Manual approach 1"],
                "hybrid_approaches": ["AI-assisted approach 1"],
                "reasoning": "Why these alternatives might be better"
            }},
            "overall_readiness_score": 0.70,
            "proceed_recommendation": true,
            "summary": "Combined assessment of project readiness",
            "actionable_recommendations": [
                "Specific action for ethical concerns",
                "Specific action for AI appropriateness",
                "Combined recommendation"
            ],
            "question_flags": [
                {{
                    "question_key": "question_key_here",
                    "issue": "Specific issue with this answer",
                    "severity": "high|medium|low",
                    "category": "ethical|appropriateness"
                }}
            ]
        }}
        """
        
        response = ""  # Initialize response variable
        try:
            logger.info(f"Starting project readiness assessment for project: {project_description[:100]}...")
            logger.info(f"Number of answers provided: {len(answers)}")
            
            response = await llm_service.analyze_text("", prompt)
            
            # Add logging to see what we actually received
            logger.info(f"LLM response received (first 200 chars): {response[:200]}")
            
            # Clean the response - remove markdown formatting if present
            cleaned_response = response.strip()
            
            # Remove markdown code blocks if present
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
                
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Remove any leading/trailing whitespace again
            cleaned_response = cleaned_response.strip()
            
            # Check if we have any content to parse
            if not cleaned_response:
                logger.error("Empty response from LLM after cleaning")
                raise ValueError("Empty response from LLM")
            
            # Add logging to see what we're trying to parse
            logger.info(f"Cleaned response to parse (first 200 chars): {cleaned_response[:200]}")
            
            # Try to parse the JSON
            assessment_data = json.loads(cleaned_response)
            
            # Validate that required fields are present
            required_fields = [
                "ethical_score", "ai_appropriateness_score", "overall_readiness_score",
                "proceed_recommendation", "summary", "actionable_recommendations"
            ]
            
            for field in required_fields:
                if field not in assessment_data:
                    logger.error(f"Missing required field in assessment: {field}")
                    assessment_data[field] = 0.0 if "score" in field else (False if field == "proceed_recommendation" else ("Assessment incomplete" if field == "summary" else []))
            
            logger.info("Successfully parsed assessment data")
            
            # Validation and scoring logic
            ethical_score = assessment_data.get("ethical_score", 0.0)
            ai_appropriateness_score = assessment_data.get("ai_appropriateness_score", 0.0)
            
            # Calculate overall readiness score (weighted average)
            overall_score = (ethical_score * 0.6) + (ai_appropriateness_score * 0.4)
            
            # Additional validation - if most questions are flagged as high severity, force lower scores
            question_flags = assessment_data.get("question_flags", [])
            high_severity_count = sum(1 for flag in question_flags if flag.get("severity") == "high")
            total_questions = len(questions)
            
            if high_severity_count >= total_questions * 0.6:  # 60% or more questions flagged as high severity
                ethical_score = min(ethical_score, 0.4)
                ai_appropriateness_score = min(ai_appropriateness_score, 0.4)
                overall_score = min(overall_score, 0.3)
                assessment_data["proceed_recommendation"] = False
                
            if high_severity_count >= total_questions * 0.8:  # 80% or more questions flagged
                ethical_score = min(ethical_score, 0.3)
                ai_appropriateness_score = min(ai_appropriateness_score, 0.3)
                overall_score = min(overall_score, 0.2)
                
            if high_severity_count == total_questions:  # All questions flagged
                ethical_score = min(ethical_score, 0.2)
                ai_appropriateness_score = min(ai_appropriateness_score, 0.2)
                overall_score = min(overall_score, 0.15)
            
            # Convert question flags to proper format
            converted_flags = []
            for flag in question_flags:
                converted_flags.append(QuestionFlag(
                    question_key=flag.get("question_key", ""),
                    issue=flag.get("issue", ""),
                    severity=flag.get("severity", "medium"),
                    category=QuestionFlagCategory(flag.get("category", "ethical"))
                ))
            
            # Handle alternative solutions
            alt_solutions = assessment_data.get("alternative_solutions")
            alternative_solutions = None
            if alt_solutions:
                alternative_solutions = AlternativeSolutions(
                    digital_alternatives=alt_solutions.get("digital_alternatives", []),
                    process_improvements=alt_solutions.get("process_improvements", []),
                    non_digital_solutions=alt_solutions.get("non_digital_solutions", []),
                    hybrid_approaches=alt_solutions.get("hybrid_approaches", []),
                    reasoning=alt_solutions.get("reasoning", "")
                )
            
            # Create the ProjectReadinessAssessment object
            project_readiness_assessment = ProjectReadinessAssessment(
                ethical_score=ethical_score,
                ethical_summary=assessment_data.get("ethical_summary", ""),
                ai_appropriateness_score=ai_appropriateness_score,
                ai_appropriateness_summary=assessment_data.get("ai_appropriateness_summary", ""),
                ai_recommendation=AIRecommendation(assessment_data.get("ai_recommendation", "appropriate")),
                alternative_solutions=alternative_solutions,
                overall_readiness_score=overall_score,
                proceed_recommendation=assessment_data.get("proceed_recommendation", False),
                summary=assessment_data.get("summary", ""),
                actionable_recommendations=assessment_data.get("actionable_recommendations", []),
                question_flags=converted_flags,
                threshold_met=overall_score >= self.overall_threshold
            )
            
            # Also create legacy EthicalAssessment for backward compatibility
            ethical_assessment = EthicalAssessment(
                score=ethical_score,
                concerns=[],  # No longer using separate concerns
                recommendations=assessment_data.get("actionable_recommendations", []),
                approved=ethical_score >= self.ethical_threshold
            )
            
            # Return data for frontend
            return {
                "project_readiness_assessment": project_readiness_assessment,
                "ethical_assessment": ethical_assessment,  # For backward compatibility
                "ethical_score": ethical_score,
                "ai_appropriateness_score": ai_appropriateness_score,
                "ai_appropriateness_summary": assessment_data.get("ai_appropriateness_summary", ""),
                "ai_recommendation": assessment_data.get("ai_recommendation", "appropriate"),
                "alternative_solutions": assessment_data.get("alternative_solutions"),
                "overall_readiness_score": overall_score,
                "proceed_recommendation": assessment_data.get("proceed_recommendation", False),
                "summary": assessment_data.get("summary", ""),
                "actionable_recommendations": assessment_data.get("actionable_recommendations", []),
                "question_flags": question_flags,  # Return original format for frontend
                "threshold_met": overall_score >= self.overall_threshold,
                "can_proceed": overall_score >= self.overall_threshold
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            
            # Try to extract JSON from the response if it's embedded in text
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    logger.info("Attempting to parse extracted JSON")
                    assessment_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.error("Failed to parse extracted JSON as well")
                    raise ValueError("Could not parse LLM response as JSON")
            else:
                logger.error("No JSON found in response")
                raise ValueError("No JSON found in LLM response")
                
        except Exception as e:
            logger.error(f"Failed to create project readiness assessment: {e}")
            if response:
                logger.error(f"Full response that failed to parse: {response}")
            else:
                logger.error("No response received from LLM")
            
            # Create fallback assessment
            fallback_ethical = EthicalAssessment(
                score=0.0,
                concerns=[],
                recommendations=["Complete all reflection questions with detailed, thoughtful responses", "Consult with ethics experts"],
                approved=False
            )
            
            fallback_readiness = ProjectReadinessAssessment(
                ethical_score=0.0,
                ethical_summary="Assessment failed - please review your responses and try again",
                ai_appropriateness_score=0.0,
                ai_appropriateness_summary="Could not evaluate AI appropriateness due to assessment failure",
                ai_recommendation=AIRecommendation.QUESTIONABLE,
                overall_readiness_score=0.0,
                proceed_recommendation=False,
                summary="Assessment failed - please ensure all questions are answered with detailed, thoughtful responses",
                actionable_recommendations=["Provide detailed, thoughtful answers to all reflection questions", "Ensure responses directly address the questions asked"],
                question_flags=[],
                threshold_met=False
            )
            
            return {
                "project_readiness_assessment": fallback_readiness,
                "ethical_assessment": fallback_ethical,
                "ethical_score": 0.0,
                "ai_appropriateness_score": 0.0,
                "ai_appropriateness_summary": "Could not evaluate AI appropriateness due to assessment failure",
                "ai_recommendation": "questionable",
                "alternative_solutions": None,
                "overall_readiness_score": 0.0,
                "proceed_recommendation": False,
                "summary": "Assessment failed - please ensure all questions are answered with detailed, thoughtful responses",
                "actionable_recommendations": ["Provide detailed, thoughtful answers to all reflection questions", "Ensure responses directly address the questions asked"],
                "question_flags": [],
                "threshold_met": False,
                "can_proceed": False
            }

    # Keep the old method name for backward compatibility, but redirect to new method
    async def create_comprehensive_ethical_assessment(
        self, 
        answers: Dict[str, str], 
        project_description: str,
        questions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Legacy method - redirects to new comprehensive assessment"""
        return await self.create_comprehensive_project_readiness_assessment(
            answers, project_description, questions
        )

    # ... [Rest of the existing methods remain unchanged] ...
    
    async def get_reflection_questions_with_guidance(self, project_description: str) -> Dict[str, Any]:
        """Generate contextual reflection questions with targeted guidance sources"""
        
        # First, get the base questions as before
        questions_data = await self.get_reflection_questions(project_description)
        
        # Now get targeted guidance sources for each specific question
        questions_with_guidance = {}
        
        for question_key, question_text in questions_data.items():
            try:
                logger.info(f"Getting guidance for question: {question_text[:50]}...")
                
                # Get guidance sources for this specific question text
                guidance_sources = await ctx.rag_service.get_question_specific_guidance_sources(
                    question_text=question_text,
                    question_area=question_key,
                    project_description=project_description,
                    max_sources=2  # Only get the 2 most relevant sources
                )
                
                questions_with_guidance[question_key] = {
                    "question": question_text,
                    "guidance_sources": guidance_sources
                }
                
                logger.info(f"Found {len(guidance_sources)} relevant guidance sources for {question_key}")
                
            except Exception as e:
                logger.warning(f"Failed to get guidance for {question_key}: {e}")
                # Fallback without guidance
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
            
            # Check if we have enhanced questions with guidance stored
            if project.reflection_questions_with_guidance:
                # Convert stored EnhancedReflectionQuestion objects back to dict format for API
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
            
            # Generate new questions with targeted guidance if none exist
            logger.info(f"Generating new reflection questions with guidance for project {project_id}")
            questions_with_guidance_raw = await self.get_reflection_questions_with_guidance(project.description)
            
            # Convert to proper model format for storage
            enhanced_questions = {}
            simple_questions = {}
            
            for key, data in questions_with_guidance_raw.items():
                # Create GuidanceSource objects only for relevant sources
                guidance_sources = []
                for source_data in data.get("guidance_sources", []):
                    try:
                        guidance_sources.append(GuidanceSource(**source_data))
                    except Exception as e:
                        logger.warning(f"Failed to create GuidanceSource: {e}")
                        continue
                
                # Create EnhancedReflectionQuestion object
                enhanced_questions[key] = ReflectionQuestion(
                    question=data["question"],
                    guidance_sources=guidance_sources
                )
                
                # Maintain backward compatibility
                simple_questions[key] = data["question"]
            
            # Save to project
            project.reflection_questions_with_guidance = enhanced_questions
            project.reflection_questions = simple_questions
            project.touch()
            await project.save()
            
            logger.info(f"Saved {len(enhanced_questions)} questions with guidance for project {project_id}")
            
            # Return format expected by frontend
            return {
                "questions": questions_with_guidance_raw,
                "has_guidance": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get/create reflection questions with guidance: {e}")
            # Fallback to simple questions without guidance
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
            