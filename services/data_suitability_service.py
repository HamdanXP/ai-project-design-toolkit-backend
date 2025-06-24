from typing import Dict, Any, List, Optional
from core.llm_service import llm_service
from models.project import Dataset
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class DataSuitabilityService:
    """Service for assessing data suitability with educational guidance"""
    
    def __init__(self):
        # Known public dataset profiles
        self.known_datasets = {
            "reliefweb": {
                "typical_quality": "high",
                "common_issues": ["Reporting delays", "Language inconsistencies"],
                "strengths": ["Well-documented", "Regular updates", "Standardized format"],
                "coverage": "Global humanitarian crises"
            },
            "un_refugee_data": {
                "typical_quality": "high", 
                "common_issues": ["3-month reporting lag", "Limited demographic detail"],
                "strengths": ["High completeness", "Authoritative source", "Long historical data"],
                "coverage": "Registered refugee populations"
            },
            "who_health_data": {
                "typical_quality": "medium_high",
                "common_issues": ["Country reporting variations", "Data collection gaps"],
                "strengths": ["Medical accuracy", "Global coverage", "Standardized indicators"],
                "coverage": "Health emergencies and epidemics"
            }
        }
    
    async def generate_assessment_questions(
        self, 
        datasets: List[Dataset],
        project_description: str
    ) -> Dict[str, Any]:
        """Generate educational assessment questions with guidance"""
        
        # Analyze if we have known datasets
        known_data_info = self._analyze_known_datasets(datasets)
        
        # Generate context-specific questions
        questions = {
            "data_accessibility": {
                "question": "Can you easily access and view the data?",
                "help_text": "You should be able to open, browse, and understand the basic structure",
                "indicators": {
                    "good_signs": [
                        "Data opens in familiar format (Excel, CSV, database)",
                        "Column headers are clear and understandable", 
                        "You can see actual data values, not just metadata"
                    ],
                    "warning_signs": [
                        "Requires special software or technical setup",
                        "Data is heavily encoded or cryptic",
                        "Only documentation available, not raw data"
                    ]
                },
                "options": ["easily_accessible", "requires_some_setup", "very_difficult", "unsure"]
            },
            "data_completeness": {
                "question": "How complete does the data appear?", 
                "help_text": "Look for missing information that's important for your project",
                "indicators": {
                    "what_to_look_for": [
                        "Blank cells or empty fields",
                        "Entries marked as 'N/A', 'Unknown', or '-'",
                        "Inconsistent number of fields across records",
                        "Time periods with no data"
                    ],
                    "completeness_levels": {
                        "high": "Less than 5% missing data in key fields",
                        "medium": "5-20% missing data, but patterns are clear",
                        "low": "More than 20% missing, or critical fields empty",
                        "unknown": "Can't determine without technical analysis"
                    }
                },
                "options": ["high_completeness", "medium_completeness", "low_completeness", "unsure"]
            },
            "data_relevance": {
                "question": "How relevant is this data to your specific project goals?",
                "help_text": "Consider geographic, temporal, and demographic alignment",
                "indicators": {
                    "relevance_factors": [
                        "Geographic coverage matches your target area",
                        "Time period aligns with your project timeline", 
                        "Population groups match your beneficiaries",
                        "Data indicators relate to your problem"
                    ],
                    "examples": {
                        "high_relevance": "Refugee data from your specific region and time period",
                        "medium_relevance": "Regional data that could be generalized to your area",
                        "low_relevance": "Global data that may not apply to your specific context"
                    }
                },
                "options": ["highly_relevant", "somewhat_relevant", "limited_relevance", "unsure"]
            },
            "data_quality_indicators": {
                "question": "When you examine the data, what do you observe?",
                "help_text": "Look for patterns that might indicate quality issues",
                "checklist": [
                    {
                        "indicator": "Consistent formatting",
                        "description": "Dates, numbers, and categories follow the same format throughout",
                        "good_example": "All dates as DD/MM/YYYY, all currencies in USD",
                        "bad_example": "Mixed date formats, some USD some EUR, inconsistent naming"
                    },
                    {
                        "indicator": "Reasonable values",
                        "description": "Numbers and categories make sense in context",
                        "good_example": "Ages between 0-100, positive population counts",
                        "bad_example": "Negative ages, impossible dates, extreme outliers"
                    },
                    {
                        "indicator": "Clear documentation",
                        "description": "You understand what each field means and how it was collected",
                        "good_example": "Data dictionary provided, collection methods explained",
                        "bad_example": "Unclear column names, no explanation of data source"
                    }
                ],
                "response_type": "checklist"
            },
            "ethical_considerations": {
                "question": "Are there ethical concerns with using this data?",
                "help_text": "Consider privacy, consent, and potential harm from data use",
                "key_considerations": [
                    {
                        "concern": "Personal Information",
                        "description": "Does the data contain names, addresses, or other identifying information?",
                        "mitigation": "Ensure data is properly anonymized before use"
                    },
                    {
                        "concern": "Sensitive Populations", 
                        "description": "Does it involve vulnerable groups (refugees, children, conflict victims)?",
                        "mitigation": "Apply extra privacy protections and ethical review"
                    },
                    {
                        "concern": "Consent and Permission",
                        "description": "Do you have proper permission to use this data for AI?",
                        "mitigation": "Verify data usage rights and obtain necessary permissions"
                    },
                    {
                        "concern": "Potential Harm",
                        "description": "Could using this data in AI cause harm to individuals or communities?",
                        "mitigation": "Conduct impact assessment and implement safeguards"
                    }
                ],
                "options": ["no_concerns", "minor_concerns", "significant_concerns", "needs_review"]
            }
        }
        
        # Add known dataset information
        if known_data_info:
            questions["known_dataset_info"] = known_data_info
        
        return {
            "assessment_questions": questions,
            "guidance": await self._generate_assessment_guidance(project_description),
            "automated_checks": await self._suggest_automated_checks(datasets)
        }
    
    async def analyze_data_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Automated analysis of uploaded data files"""
        
        try:
            analysis = {
                "file_info": {},
                "quality_indicators": {},
                "recommendations": [],
                "automated_score": 0.0
            }
            
            if file_type.lower() in ['.csv', '.xlsx', '.xls']:
                # Read file
                if file_type.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Basic file info
                analysis["file_info"] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
                    "column_names": list(df.columns)
                }
                
                # Quality indicators
                analysis["quality_indicators"] = {
                    "missing_data_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1),
                    "duplicate_rows": df.duplicated().sum(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                    "text_columns": len(df.select_dtypes(include=['object']).columns)
                }
                
                # Calculate automated score
                score = 1.0
                missing_pct = analysis["quality_indicators"]["missing_data_percentage"]
                if missing_pct > 50:
                    score *= 0.3
                elif missing_pct > 20:
                    score *= 0.6
                elif missing_pct > 5:
                    score *= 0.8
                
                if analysis["quality_indicators"]["duplicate_rows"] > len(df) * 0.1:
                    score *= 0.7
                
                analysis["automated_score"] = round(score, 2)
                
                # Generate recommendations
                if missing_pct > 20:
                    analysis["recommendations"].append("High missing data percentage - consider data cleaning")
                
                if analysis["quality_indicators"]["duplicate_rows"] > 0:
                    analysis["recommendations"].append("Duplicate rows detected - remove before analysis")
                
                if len(df.columns) > 50:
                    analysis["recommendations"].append("Large number of columns - consider feature selection")
                
            else:
                analysis["file_info"]["message"] = "Automated analysis not available for this file type"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze data file: {e}")
            return {
                "error": "Could not analyze file",
                "message": "Manual assessment required"
            }
    
    async def assess_data_suitability(
        self,
        responses: Dict[str, Any],
        datasets: List[Dataset],
        automated_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Comprehensive data suitability assessment"""
        
        # Calculate suitability score
        score_components = {}
        
        # Accessibility score
        accessibility_scores = {
            "easily_accessible": 1.0,
            "requires_some_setup": 0.7,
            "very_difficult": 0.3,
            "unsure": 0.5
        }
        score_components["accessibility"] = accessibility_scores.get(
            responses.get("data_accessibility"), 0.5
        )
        
        # Completeness score
        completeness_scores = {
            "high_completeness": 1.0,
            "medium_completeness": 0.7,
            "low_completeness": 0.3,
            "unsure": 0.5
        }
        score_components["completeness"] = completeness_scores.get(
            responses.get("data_completeness"), 0.5
        )
        
        # Relevance score
        relevance_scores = {
            "highly_relevant": 1.0,
            "somewhat_relevant": 0.7,
            "limited_relevance": 0.4,
            "unsure": 0.5
        }
        score_components["relevance"] = relevance_scores.get(
            responses.get("data_relevance"), 0.5
        )
        
        # Quality indicators score (from checklist)
        quality_checklist = responses.get("data_quality_indicators", [])
        quality_score = len(quality_checklist) / 3.0  # Assuming 3 indicators
        score_components["quality"] = min(quality_score, 1.0)
        
        # Ethical concerns (inverse scoring)
        ethical_scores = {
            "no_concerns": 1.0,
            "minor_concerns": 0.8,
            "significant_concerns": 0.5,
            "needs_review": 0.3
        }
        score_components["ethics"] = ethical_scores.get(
            responses.get("ethical_considerations"), 0.5
        )
        
        # Incorporate automated analysis if available
        if automated_analysis and "automated_score" in automated_analysis:
            score_components["automated"] = automated_analysis["automated_score"]
            weights = {
                "accessibility": 0.15,
                "completeness": 0.2,
                "relevance": 0.25,
                "quality": 0.15,
                "ethics": 0.15,
                "automated": 0.1
            }
        else:
            weights = {
                "accessibility": 0.2,
                "completeness": 0.25,
                "relevance": 0.3,
                "quality": 0.15,
                "ethics": 0.1
            }
        
        # Calculate overall score
        overall_score = sum(
            score_components.get(component, 0.5) * weight 
            for component, weight in weights.items()
        )
        
        # Generate assessment summary
        summary = self._generate_suitability_summary(overall_score, score_components, responses)
        
        return {
            "overall_suitability_score": round(overall_score, 2),
            "suitability_percentage": int(overall_score * 100),
            "suitability_level": self._get_suitability_level(overall_score),
            "component_scores": score_components,
            "summary": summary,
            "recommendations": await self._generate_data_recommendations(
                score_components, responses, automated_analysis
            ),
            "next_steps": self._generate_data_next_steps(overall_score)
        }
    
    def _analyze_known_datasets(self, datasets: List[Dataset]) -> Optional[Dict[str, Any]]:
        """Analyze known public datasets and provide context"""
        
        known_info = {}
        
        for dataset in datasets:
            source_key = None
            if "reliefweb" in dataset.source.lower():
                source_key = "reliefweb"
            elif "unhcr" in dataset.source.lower() or "refugee" in dataset.name.lower():
                source_key = "un_refugee_data"
            elif "who" in dataset.source.lower():
                source_key = "who_health_data"
            
            if source_key and source_key in self.known_datasets:
                known_info[dataset.name] = {
                    **self.known_datasets[source_key],
                    "dataset_name": dataset.name,
                    "source": dataset.source
                }
        
        return known_info if known_info else None
    
    async def _generate_assessment_guidance(self, project_description: str) -> Dict[str, Any]:
        """Generate project-specific assessment guidance"""
        
        prompt = f"""
        For this humanitarian AI project: "{project_description}"
        
        Provide specific data assessment guidance in JSON format:
        {{
            "key_data_requirements": ["requirement1", "requirement2"],
            "common_data_challenges": ["challenge1", "challenge2"],
            "quality_priorities": ["priority1", "priority2"],
            "red_flags": ["flag1", "flag2"]
        }}
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Failed to generate assessment guidance: {e}")
            return {
                "key_data_requirements": ["Relevant to target population", "Sufficient historical data"],
                "common_data_challenges": ["Missing values", "Inconsistent reporting"],
                "quality_priorities": ["Completeness", "Accuracy", "Timeliness"],
                "red_flags": ["Too much missing data", "Outdated information", "Unclear methodology"]
            }
    
    async def _suggest_automated_checks(self, datasets: List[Dataset]) -> List[Dict[str, Any]]:
        """Suggest automated checks that could be performed"""
        
        checks = []
        
        for dataset in datasets:
            if dataset.url and any(ext in dataset.url.lower() for ext in ['.csv', '.xlsx', '.json']):
                checks.append({
                    "dataset": dataset.name,
                    "type": "automated_analysis",
                    "description": "We can automatically analyze this file for basic quality indicators",
                    "capabilities": [
                        "Count missing values and duplicates",
                        "Analyze data types and formats",
                        "Generate statistical summary",
                        "Identify potential data quality issues"
                    ]
                })
            else:
                checks.append({
                    "dataset": dataset.name,
                    "type": "manual_assessment",
                    "description": "This dataset requires manual review",
                    "guidance": [
                        "Review dataset documentation thoroughly",
                        "Contact data provider for quality information",
                        "Request sample data for evaluation"
                    ]
                })
        
        return checks
    
    def _generate_suitability_summary(
        self, 
        overall_score: float, 
        components: Dict[str, float], 
        responses: Dict[str, Any]
    ) -> str:
        """Generate human-readable suitability summary"""
        
        if overall_score >= 0.8:
            return "The data appears highly suitable for your project with strong indicators across all areas."
        elif overall_score >= 0.6:
            return "The data shows good suitability with some areas for improvement identified below."
        elif overall_score >= 0.4:
            return "The data has moderate suitability but significant preparation may be needed."
        else:
            return "The data may not be suitable for your project without substantial preprocessing and quality improvement."
    
    def _get_suitability_level(self, score: float) -> str:
        """Convert score to suitability level"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium_high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    async def _generate_data_recommendations(
        self,
        components: Dict[str, float],
        responses: Dict[str, Any],
        automated_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate specific recommendations for data improvement"""
        
        recommendations = []
        
        if components.get("completeness", 1.0) < 0.6:
            recommendations.append({
                "priority": "high",
                "category": "Data Completeness",
                "issue": "Significant missing data detected",
                "recommendations": [
                    "Identify patterns in missing data",
                    "Consider data imputation techniques", 
                    "Seek additional data sources to fill gaps",
                    "Adjust project scope to work with available data"
                ]
            })
        
        if components.get("quality", 1.0) < 0.6:
            recommendations.append({
                "priority": "high",
                "category": "Data Quality",
                "issue": "Quality indicators suggest potential problems",
                "recommendations": [
                    "Conduct thorough data cleaning",
                    "Standardize formats and conventions",
                    "Remove or fix inconsistent entries",
                    "Validate data against known benchmarks"
                ]
            })
        
        if components.get("ethics", 1.0) < 0.7:
            recommendations.append({
                "priority": "critical",
                "category": "Ethical Considerations",
                "issue": "Ethical concerns need to be addressed",
                "recommendations": [
                    "Conduct formal ethical review",
                    "Implement data anonymization",
                    "Establish data governance protocols",
                    "Obtain necessary permissions and consents"
                ]
            })
        
        return recommendations
    
    def _generate_data_next_steps(self, overall_score: float) -> List[str]:
        """Generate next steps based on suitability score"""
        
        if overall_score >= 0.7:
            return [
                "Proceed with data preparation",
                "Begin exploratory data analysis",
                "Start model development planning"
            ]
        elif overall_score >= 0.5:
            return [
                "Address identified data quality issues",
                "Implement recommended improvements",
                "Re-assess data suitability after improvements"
            ]
        else:
            return [
                "Consider alternative data sources",
                "Consult with data experts",
                "Significantly modify project scope or approach"
            ]
