from typing import Dict, Any, Optional
from models.ethical_analysis import EthicalAnalysis, RiskLevel, BiasAssessment, FairnessEvaluation, PrivacyEvaluation
from services.project_service import ProjectService
from core.llm_service import llm_service
import core
import json
import logging

logger = logging.getLogger(__name__)

class EthicalAnalysisService:
    def __init__(self):
        self.project_service = ProjectService()
    
    async def analyze_dataset_ethics(
        self, 
        project_id: str, 
        statistics: Dict[str, Any]
    ) -> EthicalAnalysis:
        from fastapi import HTTPException
        
        try:
            project = await self.project_service.get_project(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            problem_domain = await self.project_service.get_project_domain(project_id)
            target_beneficiaries = getattr(project, 'target_beneficiaries', '') or ''
            
            ethical_context = await self._get_essential_ethical_context(
                problem_domain,
                target_beneficiaries
            )
            
            analysis_prompt = self._build_targeted_analysis_prompt(
                project.description,
                problem_domain,
                target_beneficiaries,
                statistics,
                ethical_context
            )
            
            response = await llm_service.analyze_text("", analysis_prompt)
            return self._parse_analysis_response(response)
            
        except ValueError as e:
            logger.error(f"Project validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"AI ethical analysis unavailable for project {project_id}: {e}")
            raise HTTPException(
                status_code=503, 
                detail="AI ethical analysis is temporarily unavailable. Statistical analysis will be shown instead."
            )
    
    async def _get_essential_ethical_context(
        self,
        problem_domain: str,
        target_beneficiaries: str
    ) -> str:
        try:
            context = await core.rag_service.get_ethical_frameworks_context(
                ai_technique="data analysis",
                project_description="",
                target_beneficiaries=target_beneficiaries
            )
            return context[:800] if context else ""
            
        except Exception as e:
            logger.warning(f"Failed to get ethical context: {e}")
            return ""
    
    def _classify_columns_semantic(self, column_analysis: list) -> Dict[str, Any]:
        classifications = {}
        administrative_cols = []
        statistical_cols = []
        potential_identifier_cols = []
        
        for col in column_analysis:
            name = col.get('name', '').lower()
            uniqueness = col.get('uniqueCount', 0) / max(col.get('totalRows', 1), 1)
            
            if any(term in name for term in ['country', 'admin', 'region', 'province', 'state', 'district', 'latitude', 'longitude']):
                classifications[col.get('name', '')] = {
                    "category": "ADMINISTRATIVE_GEOGRAPHIC",
                    "privacy_risk": "low",
                    "reasoning": "Public administrative boundary data"
                }
                administrative_cols.append(col.get('name', ''))
            elif any(term in name for term in ['value', 'total', 'count', 'sum', 'avg', 'indicator', 'metric', 'aggregation']):
                classifications[col.get('name', '')] = {
                    "category": "STATISTICAL_MEASURE", 
                    "privacy_risk": "low",
                    "reasoning": "Aggregated statistical data"
                }
                statistical_cols.append(col.get('name', ''))
            elif any(term in name for term in ['name', 'id', 'identifier', 'email', 'phone']) and uniqueness > 0.8:
                classifications[col.get('name', '')] = {
                    "category": "PERSONAL_IDENTIFIER",
                    "privacy_risk": "high",
                    "reasoning": f"High uniqueness ({uniqueness:.0%}) suggests personal identifiers"
                }
                potential_identifier_cols.append(col.get('name', ''))
            elif any(term in name for term in ['date', 'time', 'year', 'month']):
                classifications[col.get('name', '')] = {
                    "category": "TEMPORAL",
                    "privacy_risk": "low",
                    "reasoning": "Temporal data for analysis"
                }
            else:
                risk_level = "low" if uniqueness < 0.5 else "medium"
                classifications[col.get('name', '')] = {
                    "category": "CATEGORICAL_ATTRIBUTE",
                    "privacy_risk": risk_level,
                    "reasoning": f"Categorical data with {uniqueness:.0%} uniqueness"
                }
        
        if len(administrative_cols) >= 2 and len(statistical_cols) >= 1:
            dataset_type = "administrative_statistics"
            privacy_context = "Low privacy risk - primarily administrative and statistical data"
        elif len(potential_identifier_cols) > 0:
            dataset_type = "individual_records"
            privacy_context = "High privacy risk - contains potential personal identifiers"
        else:
            dataset_type = "aggregated_data"
            privacy_context = "Medium privacy risk - aggregated data with some identifying patterns"
        
        return {
            "classifications": classifications,
            "dataset_type": dataset_type,
            "privacy_context": privacy_context,
            "administrative_columns": administrative_cols,
            "statistical_columns": statistical_cols,
            "identifier_columns": potential_identifier_cols
        }

    def _build_targeted_analysis_prompt(
        self,
        project_description: str,
        problem_domain: str,
        target_beneficiaries: str,
        statistics: Dict[str, Any],
        ethical_context: str
    ) -> str:
        basic_metrics = statistics.get('basicMetrics', {})
        quality_assessment = statistics.get('qualityAssessment', {})
        column_analysis = statistics.get('columnAnalysis', [])
        
        column_semantics = self._classify_columns_semantic(column_analysis)
        domain_guidance = self._get_domain_guidance(problem_domain)
        
        key_columns = []
        for col_name, details in column_semantics['classifications'].items():
            key_columns.append(f"{col_name}: {details['category']} ({details['privacy_risk']} risk)")
        
        return f"""Analyze this {problem_domain} humanitarian AI dataset for ethical considerations using actual data patterns.

PROJECT: {project_description}
TARGET: {target_beneficiaries}
DATASET: {column_semantics['dataset_type']} with {basic_metrics.get('totalRows', 0):,} rows, {basic_metrics.get('totalColumns', 0)} columns

COLUMN ANALYSIS:
{chr(10).join(key_columns[:8])}

DATA QUALITY METRICS:
- Completeness: {quality_assessment.get('completenessScore', 0)}%
- Consistency: {quality_assessment.get('consistencyScore', 0)}%
- Missing Values: {sum(basic_metrics.get('missingValues', {}).values())} total

PRIVACY CONTEXT: {column_semantics['privacy_context']}

{domain_guidance}

{ethical_context}

SCORING INSTRUCTIONS:
Analyze the actual data patterns above and provide scores (0-100, higher=better):
- Privacy Score: Rate privacy protection based on identifier analysis and data type
- Fairness Score: Rate representation quality for {target_beneficiaries} based on data patterns  
- Quality Score: Rate based on completeness ({quality_assessment.get('completenessScore', 0)}%) and consistency ({quality_assessment.get('consistencyScore', 0)}%)
- Humanitarian Alignment: Rate how well this data fits {problem_domain} humanitarian needs

Calculate points as: score × weight ÷ 100, then sum for suitability_score.

Return this exact JSON structure:
{{
    "overall_risk_level": "low|medium|high",
    "bias_assessment": {{
        "level": "low|medium|high", 
        "concerns": ["specific concern based on {target_beneficiaries} data representation"],
        "recommendations": ["actionable {problem_domain} recommendation"]
    }},
    "fairness_evaluation": {{
        "representation_issues": ["specific {target_beneficiaries} representation issue"],
        "recommendations": ["specific {problem_domain} recommendation"]
    }},
    "privacy_evaluation": {{
        "risk_level": "low|medium|high",
        "concerns": ["specific concern based on {column_semantics['dataset_type']} with {len(column_semantics['identifier_columns'])} identifiers"], 
        "recommendations": ["specific {problem_domain} privacy recommendation"],
        "assessment_reasoning": "Privacy assessment: {column_semantics['dataset_type']} with {len(column_semantics['identifier_columns'])} potential identifiers detected"
    }},
    "overall_recommendation": "Specific recommendation for {problem_domain} project serving {target_beneficiaries}",
    "suitability_score": 0,
    "scoring_breakdown": {{
        "privacy_score": {{"score": 0, "weight": 30, "points": 0.0, "reasoning": "Based on {column_semantics['dataset_type']} analysis"}},
        "fairness_score": {{"score": 0, "weight": 25, "points": 0.0, "reasoning": "Based on {target_beneficiaries} representation analysis"}},
        "quality_score": {{"score": 0, "weight": 25, "points": 0.0, "reasoning": "Based on {quality_assessment.get('completenessScore', 0)}% completeness and {quality_assessment.get('consistencyScore', 0)}% consistency"}},
        "humanitarian_alignment": {{"score": 0, "weight": 20, "points": 0.0, "reasoning": "{problem_domain} context alignment assessment"}}
    }}
}}"""

    def _get_domain_guidance(self, problem_domain: str) -> str:
        domain_guidance = {
            "health": "HEALTH CONTEXT: Focus on patient privacy, health equity, and medical data sensitivity.",
            "education": "EDUCATION CONTEXT: Consider student privacy, educational equity, and learning outcome fairness.",
            "agriculture": "AGRICULTURE CONTEXT: Evaluate farmer data protection, crop yield equity, and rural representation.",
            "disaster_response": "DISASTER CONTEXT: Assess emergency data ethics, vulnerable population protection, and rapid deployment needs.",
            "water_sanitation": "WATER/SANITATION CONTEXT: Consider community privacy, infrastructure equity, and access fairness."
        }
        
        return domain_guidance.get(problem_domain.lower(), "HUMANITARIAN CONTEXT: Apply general humanitarian data principles.")

    def _parse_analysis_response(self, response: str) -> EthicalAnalysis:
        try:
            if not response or not response.strip():
                raise ValueError("Empty response from AI analysis")
            
            cleaned_response = response.strip()
            
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            if not cleaned_response:
                raise ValueError("No content after cleaning response")
            
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                json_start = cleaned_response.find('{')
                json_end = cleaned_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(cleaned_response[json_start:json_end])
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Ensure scoring breakdown exists and calculate suitability score
            if 'scoring_breakdown' in data and data['scoring_breakdown']:
                total_points = 0
                for breakdown_item in data['scoring_breakdown'].values():
                    if isinstance(breakdown_item, dict) and 'points' in breakdown_item:
                        total_points += float(breakdown_item['points'])
                data['suitability_score'] = round(total_points)
            
            return EthicalAnalysis(
                overall_risk_level=RiskLevel(data.get('overall_risk_level', 'medium')),
                bias_assessment=BiasAssessment(
                    level=RiskLevel(data.get('bias_assessment', {}).get('level', 'medium')),
                    concerns=data.get('bias_assessment', {}).get('concerns', []),
                    recommendations=data.get('bias_assessment', {}).get('recommendations', [])
                ),
                fairness_evaluation=FairnessEvaluation(
                    representation_issues=data.get('fairness_evaluation', {}).get('representation_issues', []),
                    recommendations=data.get('fairness_evaluation', {}).get('recommendations', [])
                ),
                privacy_evaluation=PrivacyEvaluation(
                    risk_level=RiskLevel(data.get('privacy_evaluation', {}).get('risk_level', 'medium')),
                    concerns=data.get('privacy_evaluation', {}).get('concerns', []),
                    recommendations=data.get('privacy_evaluation', {}).get('recommendations', []),
                    assessment_reasoning=data.get('privacy_evaluation', {}).get('assessment_reasoning', 'Privacy assessment completed using AI analysis.')
                ),
                overall_recommendation=data.get('overall_recommendation', 'Dataset analysis completed'),
                suitability_score=data.get('suitability_score', 50),
                scoring_breakdown=data.get('scoring_breakdown', {})
            )
            
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            raise ValueError(f"Invalid AI analysis response format: {e}")