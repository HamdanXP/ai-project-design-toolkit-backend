from datetime import datetime
import json
import logging
from typing import Dict, Any, List, Optional
from core.llm_service import llm_service
from models.project import Project
from models.evaluation import (
    EvaluationContext, SimulationResult, EvaluationResult, TestingMethod,
    SimulationCapabilities, TestingScenario, ExampleScenario,
    SimulationExplanation, EvaluationSummary,
    ScenarioRegenerationRequest, ProjectDownloads, DownloadableFile,
    EvaluationApproach, SuitabilityAssessment, FeatureCompatibility,
    DataVolumeAssessment, DataQualityAssessment, SuitabilityRecommendation,
    EvaluationBypass, SimulationType, ConfidenceLevel
)

logger = logging.getLogger(__name__)

class EvaluationService:
    
    def __init__(self):
        pass
    
    async def get_evaluation_context(self, project: Project) -> EvaluationContext:
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise ValueError("No generated project found for evaluation")
        
        generated_project_data = project.development_data["generated_project"]
        selected_solution = self._get_full_selected_solution(project)
        
        evaluation_approach = self._determine_evaluation_approach(selected_solution)
        
        simulation_capabilities = await self._determine_simulation_capabilities(
            project, generated_project_data, selected_solution, evaluation_approach
        )
        
        testing_scenarios = None
        evaluation_bypass = None
        
        if evaluation_approach == EvaluationApproach.SCENARIO_BASED:
            testing_scenarios = await self._generate_testing_scenarios(
                project, generated_project_data, selected_solution
            )
        elif evaluation_approach == EvaluationApproach.EVALUATION_BYPASS:
            evaluation_bypass = self._create_evaluation_bypass(selected_solution)
        
        return EvaluationContext(
            generated_project=generated_project_data,
            selected_solution=selected_solution,
            simulation_capabilities=simulation_capabilities,
            testing_scenarios=testing_scenarios,
            evaluation_bypass=evaluation_bypass,
            available_downloads=self._get_available_downloads(generated_project_data)
        )
    
    def _get_full_selected_solution(self, project: Project) -> Dict[str, Any]:
        selected_solution_meta = project.development_data.get("selected_solution", {})
        solution_id = selected_solution_meta.get("solution_id")
        
        if not solution_id:
            raise ValueError("No solution selected")
        
        available_solutions = project.development_data.get("available_solutions", [])
        for solution in available_solutions:
            if solution.get("id") == solution_id:
                return solution
        
        raise ValueError(f"Selected solution {solution_id} not found")

    def _determine_evaluation_approach(self, selected_solution: Dict[str, Any]) -> EvaluationApproach:
        needs_dataset = selected_solution.get("needs_dataset", False)
        dataset_type = selected_solution.get("dataset_type")
        
        if not needs_dataset:
            return EvaluationApproach.SCENARIO_BASED
        elif dataset_type == "tabular":
            return EvaluationApproach.DATASET_ANALYSIS
        else:
            return EvaluationApproach.EVALUATION_BYPASS
    
    async def _determine_simulation_capabilities(
        self, 
        project: Project, 
        generated_project_data: Dict[str, Any], 
        selected_solution: Dict[str, Any],
        evaluation_approach: EvaluationApproach
    ) -> SimulationCapabilities:
        
        ai_technique = selected_solution.get("ai_technique", "classification").lower()
        
        if evaluation_approach == EvaluationApproach.DATASET_ANALYSIS:
            testing_method = TestingMethod.DATASET
            explanation = f"This {ai_technique} solution requires dataset analysis to assess compatibility with your data structure and provide suitability recommendations."
            data_formats = ["csv", "excel", "json"]
        elif evaluation_approach == EvaluationApproach.SCENARIO_BASED:
            testing_method = TestingMethod.SCENARIOS
            explanation = f"This {ai_technique} solution works best with scenario-based testing to evaluate how it handles different humanitarian situations."
            data_formats = []
        else:
            testing_method = TestingMethod.BYPASS
            explanation = f"This {ai_technique} solution requires specialized evaluation tools beyond this toolkit's scope."
            data_formats = []
        
        return SimulationCapabilities(
            testing_method=testing_method,
            evaluation_approach=evaluation_approach,
            ai_technique=ai_technique,
            data_formats_supported=data_formats,
            explanation=explanation
        )
    
    def _create_evaluation_bypass(self, selected_solution: Dict[str, Any]) -> EvaluationBypass:
        dataset_type = selected_solution.get("dataset_type", "unknown")
        ai_technique = selected_solution.get("ai_technique", "")
        
        specialist_map = {
            "image": "computer vision specialists",
            "audio": "speech recognition experts", 
            "video": "multimedia AI specialists",
            "text": "natural language processing experts"
        }
        
        specialist = specialist_map.get(dataset_type, "domain specialists")
        
        return EvaluationBypass(
            message=f"This {dataset_type} solution requires specialized evaluation beyond this toolkit's scope.",
            guidance=f"The generated {ai_technique} code is ready for testing, but {dataset_type} data analysis requires specialized tools and expertise.",
            can_download=True,
            next_steps=[
                "Download the generated prototype code",
                f"Consult with {specialist} for proper evaluation",
                "Test with real data in a secure, specialized environment",
                "Validate results with domain experts"
            ],
            specialist_consultation=f"Contact {specialist} for {dataset_type} data evaluation and model performance testing."
        )
    
    async def simulate_with_dataset_stats(self, project: Project, dataset_statistics: Dict[str, Any]) -> SimulationResult:
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise ValueError("No generated project found for simulation")
        
        generated_project_data = project.development_data["generated_project"]
        selected_solution = self._get_full_selected_solution(project)
        
        suitability_assessment = await self._assess_dataset_suitability(
            project, dataset_statistics, selected_solution
        )
        
        simulation_explanation = self._create_simulation_explanation(
            TestingMethod.DATASET, dataset_statistics
        )
        
        return SimulationResult(
            simulation_type=SimulationType.SUITABILITY_ASSESSMENT,
            testing_method=TestingMethod.DATASET,
            confidence_level=self._determine_confidence_level(suitability_assessment),
            suitability_assessment=suitability_assessment,
            simulation_explanation=simulation_explanation
        )
    
    async def simulate_without_dataset(self, project: Project, custom_scenarios: Optional[List[str]] = None) -> SimulationResult:
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise ValueError("No generated project found for simulation")
        
        generated_project_data = project.development_data["generated_project"]
        selected_solution = self._get_full_selected_solution(project)
        
        simulation_explanation = self._create_simulation_explanation(
            TestingMethod.SCENARIOS, None, custom_scenarios
        )
        
        example_scenarios = await self._generate_example_scenarios(
            project, generated_project_data, selected_solution, custom_scenarios
        )
        
        return SimulationResult(
            simulation_type=SimulationType.EXAMPLE_SCENARIOS,
            testing_method=TestingMethod.SCENARIOS,
            scenarios=example_scenarios,
            confidence_level=ConfidenceLevel.MEDIUM,
            simulation_explanation=simulation_explanation
        )
    
    async def _assess_dataset_suitability(
        self, 
        project: Project, 
        dataset_stats: Dict[str, Any], 
        selected_solution: Dict[str, Any]
    ) -> SuitabilityAssessment:
        
        tabular_requirements = selected_solution.get("tabular_requirements")
        if not tabular_requirements:
            return self._create_basic_suitability_assessment(dataset_stats)
        
        feature_compatibility = await self._assess_feature_compatibility(dataset_stats, tabular_requirements)
        data_volume_assessment = self._assess_data_volume(dataset_stats, tabular_requirements)
        data_quality_assessment = self._assess_data_quality(dataset_stats)
        
        overall_score = (
            feature_compatibility.compatibility_score * 0.4 +
            data_volume_assessment.volume_score * 0.3 +
            data_quality_assessment.quality_score * 0.3
        )
        
        is_suitable = (feature_compatibility.compatible and 
                      data_volume_assessment.sufficient and 
                      data_quality_assessment.quality_score >= 0.6)
        
        recommendations = await self._generate_dynamic_recommendations(
            feature_compatibility, data_volume_assessment, data_quality_assessment, 
            project, selected_solution, dataset_stats
        )
        
        performance_estimate = await self._generate_performance_estimate(
            dataset_stats, selected_solution, overall_score
        ) if is_suitable else None
        
        return SuitabilityAssessment(
            is_suitable=is_suitable,
            overall_score=overall_score,
            feature_compatibility=feature_compatibility,
            data_volume_assessment=data_volume_assessment,
            data_quality_assessment=data_quality_assessment,
            recommendations=recommendations,
            performance_estimate=performance_estimate
        )
    
    async def _generate_dynamic_recommendations(
        self,
        feature_compatibility: FeatureCompatibility,
        data_volume_assessment: DataVolumeAssessment,
        data_quality_assessment: DataQualityAssessment,
        project: Project,
        selected_solution: Dict[str, Any],
        dataset_stats: Dict[str, Any]
    ) -> List[SuitabilityRecommendation]:
        
        current_technique = selected_solution.get("ai_technique", "classification")
        available_features = feature_compatibility.available_required
        missing_features = feature_compatibility.missing_required
        data_rows = data_volume_assessment.available_rows
        quality_score = data_quality_assessment.quality_score
        
        context = {
            "project_description": project.description,
            "current_ai_technique": current_technique,
            "available_features": available_features,
            "missing_features": missing_features,
            "data_rows": data_rows,
            "data_quality": quality_score,
            "is_compatible": feature_compatibility.compatible,
            "sufficient_volume": data_volume_assessment.sufficient,
            "project_domain": getattr(project, 'problem_domain', 'humanitarian')
        }
        
        prompt = f"""
        You are analyzing an AI project evaluation and providing recommendations to a non-technical humanitarian professional. 
        Based on the evaluation results, generate actionable recommendations in plain English.

        PROJECT CONTEXT:
        - Project Description: {context['project_description']}
        - Current AI Approach: {context['current_ai_technique']}
        - Available Data Features: {', '.join(available_features) if available_features else 'None identified'}
        - Missing Required Features: {', '.join(missing_features) if missing_features else 'None'}
        - Number of Data Rows: {data_rows}
        - Data Quality Score (0-1): {quality_score:.2f}
        - Data Compatible with Current Approach: {context['is_compatible']}
        - Sufficient Data Volume: {context['sufficient_volume']}
        - Humanitarian Domain: {context['project_domain']}

        EVALUATION INSIGHTS:
        {self._generate_evaluation_insights(context)}

        GUIDELINES:
        - Use simple, non-technical language suitable for humanitarian professionals
        - Avoid AI jargon (no "model accuracy", "feature engineering", "hyperparameters", etc.)
        - Link recommendations to humanitarian values: fairness, transparency, community benefit, ethical impact
        - Be specific and actionable (not vague like "improve data quality")
        - Provide encouraging, constructive guidance

        OUTPUT FORMAT (JSON only):
        {{
            "recommendations": [
                {{
                    "type": "solution_alternative" | "data_collection" | "improvement",
                    "priority": "high" | "medium" | "low",
                    "issue": "Clear statement of the problem found in evaluation",
                    "suggestion": "Specific action the humanitarian professional should take"
                }}
            ]
        }}

        SPECIFIC SCENARIOS TO ADDRESS:
        
        1. If data is insufficient for current AI approach:
            - Recommend data collection OR solution alternative
        
        2. If missing critical features:
            - Prioritize data collection for essential missing features
        
        3. If data quality is poor:
            - Suggest data improvement steps
            - For severe quality issues: recommend approaches that handle noisy data
        
        4. If current approach is unsuitable for humanitarian context:
            - Recommend more appropriate, explainable approaches
            - Emphasize community trust and transparency needs
        
        Generate 2-4 recommendations based on the evaluation results above.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            analysis = self._extract_json_from_response(response)
            
            recommendations = []
            for rec_data in analysis.get("recommendations", []):
                rec = SuitabilityRecommendation(
                    type=rec_data.get("type", "improvement"),
                    priority=rec_data.get("priority", "medium"),
                    issue=rec_data.get("issue", ""),
                    suggestion=rec_data.get("suggestion", "")
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic recommendations: {e}")
            return self._get_fallback_recommendations(
                feature_compatibility, data_volume_assessment, data_quality_assessment
            )

    def _generate_evaluation_insights(self, context: Dict[str, Any]) -> str:
        insights = []
        
        if not context['is_compatible']:
            insights.append(f"- Current {context['current_ai_technique']} approach is not compatible with available data")
        
        if not context['sufficient_volume']:
            insights.append(f"- Dataset has only {context['data_rows']} rows, which may be insufficient for {context['current_ai_technique']}")
        
        if context['data_quality'] < 0.7:
            insights.append(f"- Data quality score of {context['data_quality']:.2f} indicates significant quality issues")
        
        if context['missing_features']:
            insights.append(f"- Missing {len(context['missing_features'])} required features: {', '.join(context['missing_features'][:3])}")
        
        if not insights:
            insights.append("- Data appears suitable for current AI approach, but minor improvements may be beneficial")
        
        return '\n'.join(insights)

    def _get_fallback_recommendations(
        self,
        feature_compatibility: FeatureCompatibility,
        data_volume_assessment: DataVolumeAssessment,
        data_quality_assessment: DataQualityAssessment
    ) -> List[SuitabilityRecommendation]:
        
        recommendations = []
        
        if not feature_compatibility.compatible:
            available_text = ', '.join(feature_compatibility.available_required) if feature_compatibility.available_required else 'your available data'
            missing_text = ', '.join(feature_compatibility.missing_required)
            recommendations.append(SuitabilityRecommendation(
                type="solution_alternative",
                priority="high",
                issue=f"Missing critical data: {missing_text}",
                suggestion="Try simpler AI approaches that work with the data you already have"
            ))
        
        if not data_volume_assessment.sufficient:
            row_count = data_volume_assessment.available_rows
            recommendations.append(SuitabilityRecommendation(
                type="solution_alternative",
                priority="medium",
                issue=f"Limited data: only {row_count} rows available",
                suggestion="Consider simpler AI methods that work well with smaller amounts of data"
            ))
        
        if data_quality_assessment.quality_score < 0.7:
            recommendations.append(SuitabilityRecommendation(
                type="improvement",
                priority="medium",
                issue="Data quality issues may affect results",
                suggestion="Clean your data by filling in missing information before using the AI solution"
            ))
        
        return recommendations
    
    async def _assess_feature_compatibility(
        self, 
        dataset_stats: Dict[str, Any], 
        tabular_requirements: Dict[str, Any]
    ) -> FeatureCompatibility:
        
        columnAnalysis = list(dataset_stats.get("columnAnalysis", []))
        required_features = tabular_requirements.get("required_features", [])
        optional_features = tabular_requirements.get("optional_features", [])

        dataset_columns = [col["name"] for col in columnAnalysis]
        
        required_descriptions = []
        for feature in required_features:
            required_descriptions.append(f"- {feature['name']}: {feature['description']} (Purpose: {feature['humanitarian_purpose']})")
        
        optional_descriptions = []
        for feature in optional_features:
            optional_descriptions.append(f"- {feature['name']}: {feature['description']} (Purpose: {feature['humanitarian_purpose']})")
        
        prompt = f"""
        Analyze dataset compatibility for this humanitarian AI solution.
        
        AVAILABLE DATASET COLUMNS:
        {', '.join(dataset_columns)}
        
        REQUIRED FEATURES FOR AI SOLUTION:
        {chr(10).join(required_descriptions)}
        
        OPTIONAL FEATURES FOR AI SOLUTION:
        {chr(10).join(optional_descriptions)}
        
        Task: Match available columns to required/optional features using semantic understanding.
        Consider variations in naming, case, punctuation, units, and domain terminology.
        
        Respond with ONLY valid JSON:
        {{
            "feature_mappings": [
                {{
                    "required_feature": "feature_name",
                    "available_column": "DATASET_COLUMN_NAME", 
                    "confidence": confidence_score_0_to_1,
                    "explanation": "explanation of why these match"
                }}
            ],
            "missing_required": [
                {{
                    "feature": "feature_name",
                    "reason": "explanation of why not found"
                }}
            ],
            "missing_optional": [
                {{
                    "feature": "feature_name", 
                    "reason": "explanation of why not found"
                }}
            ],
            "compatibility_summary": "brief summary of overall compatibility"
        }}
        
        Be flexible with semantic matching but confident in your assessments.
        """
        
        response = await llm_service.analyze_text("", prompt)
        analysis = self._extract_json_from_response(response)
        
        mapped_required = []
        high_confidence_mappings = []
        
        for mapping in analysis.get("feature_mappings", []):
            mapped_required.append(mapping["required_feature"])
            if mapping.get("confidence", 0) >= 0.8:
                high_confidence_mappings.append(mapping["available_column"])
        
        required_feature_names = {f["name"] for f in required_features}
        missing_required = [item["feature"] for item in analysis.get("missing_required", [])]
        missing_optional = [item["feature"] for item in analysis.get("missing_optional", [])]
        
        compatibility_score = len(mapped_required) / len(required_feature_names) if required_feature_names else 1.0
        compatible = len(missing_required) == 0
        
        if compatible:
            gap_explanation = f"Great! Your dataset has compatible features for this AI solution. Found semantic matches: {', '.join(high_confidence_mappings)}"
        else:
            gap_explanation = analysis.get("compatibility_summary", f"Missing {len(missing_required)} critical features: {', '.join(missing_required)}")
        
        return FeatureCompatibility(
            compatible=compatible,
            missing_required=missing_required,
            missing_optional=missing_optional,
            available_required=mapped_required,
            compatibility_score=compatibility_score,
            gap_explanation=gap_explanation
        )
    
    def _assess_data_volume(
        self, 
        dataset_stats: Dict[str, Any], 
        tabular_requirements: Dict[str, Any]
    ) -> DataVolumeAssessment:
        
        available_rows = dataset_stats.get("basicMetrics", {}).get("totalRows", 0)
        required_rows = tabular_requirements.get("minimum_rows", 100)
        
        volume_score = min(1.0, available_rows / required_rows)
        sufficient = available_rows >= required_rows
        
        if sufficient:
            recommendation = f"Your dataset has {available_rows} rows, which meets the minimum requirement of {required_rows} rows for reliable model training."
        else:
            recommendation = f"Your dataset has only {available_rows} rows, but this AI technique typically needs at least {required_rows} rows for good performance. Consider collecting more data or trying simpler approaches."
        
        return DataVolumeAssessment(
            sufficient=sufficient,
            available_rows=available_rows,
            required_rows=required_rows,
            volume_score=volume_score,
            recommendation=recommendation
        )
    
    def _assess_data_quality(self, dataset_stats: Dict[str, Any]) -> DataQualityAssessment:
        
        quality_assessment = dataset_stats.get("qualityAssessment", {})
        completeness_percentage = quality_assessment.get("completenessScore", 80)
        
        quality_score = completeness_percentage / 100.0
        
        issues_found = []
        recommendations = []
        
        if completeness_percentage < 90:
            issues_found.append(f"Data completeness is {completeness_percentage}% - some missing values detected")
            recommendations.append("Consider data cleaning to handle missing values before training")
        
        if completeness_percentage < 70:
            issues_found.append("Significant missing data may affect model reliability")
            recommendations.append("Investigate patterns in missing data and consider data collection improvements")
        
        return DataQualityAssessment(
            quality_score=quality_score,
            completeness_percentage=completeness_percentage,
            issues_found=issues_found,
            recommendations=recommendations
        )
    
    def _create_basic_suitability_assessment(self, dataset_stats: Dict[str, Any]) -> SuitabilityAssessment:
        
        basic_quality = self._assess_data_quality(dataset_stats)
        available_rows = dataset_stats.get("basicMetrics", {}).get("totalRows", 0)
        
        return SuitabilityAssessment(
            is_suitable=True,
            overall_score=basic_quality.quality_score,
            feature_compatibility=FeatureCompatibility(
                compatible=True,
                missing_required=[],
                missing_optional=[],
                available_required=[],
                compatibility_score=1.0,
                gap_explanation="This solution can work with your dataset structure."
            ),
            data_volume_assessment=DataVolumeAssessment(
                sufficient=True,
                available_rows=available_rows,
                required_rows=available_rows,
                volume_score=1.0,
                recommendation=f"Your dataset with {available_rows} rows is ready for this approach."
            ),
            data_quality_assessment=basic_quality,
            recommendations=[],
            performance_estimate="This solution should work well with your dataset structure."
        )
    
    async def _generate_performance_estimate(
        self,
        dataset_stats: Dict[str, Any],
        selected_solution: Dict[str, Any],
        overall_score: float
    ) -> str:
        
        ai_technique = selected_solution.get("ai_technique", "classification")
        completeness = dataset_stats.get("qualityAssessment", {}).get("completenessScore", 80)
        rows = dataset_stats.get("basicMetrics", {}).get("totalRows", 0)
        
        prompt = f"""
        Based on dataset characteristics and AI technique, provide a realistic performance estimate.
        
        AI TECHNIQUE: {ai_technique}
        DATA ROWS: {rows}
        DATA COMPLETENESS: {completeness}%
        OVERALL SUITABILITY SCORE: {overall_score:.2f}
        
        Provide a 1-2 sentence realistic estimate of what performance to expect, being clear these are estimates.
        Be specific about the technique and dataset characteristics.
        """
        
        response = await llm_service.analyze_text("", prompt)
        return response.strip()
    
    def _determine_confidence_level(self, suitability_assessment: SuitabilityAssessment) -> ConfidenceLevel:
        if suitability_assessment.overall_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif suitability_assessment.overall_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    async def evaluate_results(self, project: Project, simulation_result: SimulationResult) -> EvaluationResult:
        
        evaluation_summary = await self._generate_evaluation_summary(project, simulation_result)
        status = self._determine_evaluation_status(simulation_result, evaluation_summary)
        next_steps = self._generate_next_steps(status)
        
        development_feedback = None
        if status != "ready_for_deployment":
            development_feedback = await self._generate_holistic_development_feedback(
                project, simulation_result, evaluation_summary
            )
        
        return EvaluationResult(
            status=status,
            evaluation_summary=evaluation_summary,
            simulation_results=simulation_result,
            development_feedback=development_feedback,
            decision_options=["proceed_with_solution", "try_different_solution"],
            next_steps=next_steps,
            evaluation_timestamp=datetime.utcnow().isoformat()
        )
    
    async def _generate_holistic_development_feedback(
        self, 
        project: Project, 
        simulation_result: SimulationResult,
        evaluation_summary: EvaluationSummary
    ) -> str:
        
        if not simulation_result.suitability_assessment:
            return "Consider requesting a different AI approach that better fits your project requirements."
        
        assessment = simulation_result.suitability_assessment
        
        context = {
            "project_description": project.description,
            "current_technique": project.development_data.get("selected_solution", {}).get("ai_technique", ""),
            "is_suitable": assessment.is_suitable,
            "overall_score": assessment.overall_score,
            "compatibility_issues": not assessment.feature_compatibility.compatible,
            "volume_issues": not assessment.data_volume_assessment.sufficient,
            "quality_issues": assessment.data_quality_assessment.quality_score < 0.7,
            "available_features": assessment.feature_compatibility.available_required,
            "missing_features": assessment.feature_compatibility.missing_required,
            "data_rows": assessment.data_volume_assessment.available_rows,
            "recommendations": assessment.recommendations
        }
        
        prompt = f"""
        Generate comprehensive development feedback for a humanitarian professional to use when requesting a new AI solution.
        
        CURRENT SITUATION:
        - Project: {context['project_description']}
        - Current AI approach: {context['current_technique']}
        - Overall suitability: {context['overall_score']:.1%}
        - Compatible with data: {not context['compatibility_issues']}
        - Sufficient data volume: {not context['volume_issues']}
        - Good data quality: {not context['quality_issues']}
        
        SPECIFIC ISSUES:
        {self._format_issues_for_prompt(context)}
        
        Generate a single, coherent paragraph that the user can include when requesting a new solution.
        Write it as direct user requirements, not as instructions to a developer.
        Focus on what they need based on their data constraints and humanitarian context.
        
        Example format: "I need an AI solution that works effectively with [specific constraints] and can handle [specific challenges]. My dataset has [characteristics] and the solution should [requirements]."
        """
        
        response = await llm_service.analyze_text("", prompt)
        return response.strip()

    def _format_issues_for_prompt(self, context: Dict[str, Any]) -> str:
        issues = []
        
        if context['compatibility_issues']:
            issues.append(f"- Missing required features: {', '.join(context['missing_features'])}")
        
        if context['volume_issues']:
            issues.append(f"- Limited data: only {context['data_rows']} rows available")
        
        if context['quality_issues']:
            issues.append("- Data quality concerns detected")
        
        for rec in context['recommendations']:
            if rec.priority == "high":
                issues.append(f"- {rec.issue}")
        
        return '\n'.join(issues) if issues else "- General compatibility improvements needed"
    
    async def _generate_evaluation_summary(
        self, 
        project: Project, 
        simulation_result: SimulationResult
    ) -> EvaluationSummary:
        
        if simulation_result.suitability_assessment:
            assessment = simulation_result.suitability_assessment
            
            if assessment.is_suitable:
                overall_assessment = f"Your dataset is well-suited for this AI solution (suitability score: {assessment.overall_score:.1%})"
                deployment_readiness = True
                recommendation = "This solution is ready for implementation with your dataset"
                
                key_strengths = []
                if assessment.feature_compatibility.compatible:
                    key_strengths.append("All required features are available in your dataset")
                if assessment.data_volume_assessment.sufficient:
                    key_strengths.append(f"Data volume is sufficient ({assessment.data_volume_assessment.available_rows} rows)")
                if assessment.data_quality_assessment.quality_score >= 0.8:
                    key_strengths.append(f"High data quality score: {assessment.data_quality_assessment.completeness_percentage}%")
                elif assessment.data_quality_assessment.quality_score >= 0.6:
                    key_strengths.append(f"Acceptable data quality: {assessment.data_quality_assessment.completeness_percentage}%")
                
                areas_for_improvement = []
                for rec in assessment.recommendations:
                    if rec.priority in ["medium", "low"]:
                        areas_for_improvement.append(rec.suggestion)
                        
                if not areas_for_improvement:
                    areas_for_improvement = ["Monitor performance during deployment"]
                    
            else:
                overall_assessment = f"Dataset compatibility issues identified (suitability score: {assessment.overall_score:.1%})"
                deployment_readiness = False
                recommendation = "Address data compatibility issues or try a different AI solution"
                
                key_strengths = []
                if assessment.feature_compatibility.available_required:
                    key_strengths.append(f"Available features: {', '.join(assessment.feature_compatibility.available_required)}")
                if assessment.data_volume_assessment.available_rows > 0:
                    key_strengths.append(f"Dataset contains {assessment.data_volume_assessment.available_rows} rows of data")
                
                areas_for_improvement = [rec.suggestion for rec in assessment.recommendations if rec.priority == "high"]
        
        elif simulation_result.scenarios:
            scenario_count = len(simulation_result.scenarios)
            overall_assessment = f"Scenario testing shows capability across {scenario_count} humanitarian use cases"
            deployment_readiness = True
            recommendation = "Solution demonstrates good potential for humanitarian applications"
            
            key_strengths = [
                "Handles diverse humanitarian scenarios",
                "Context-appropriate responses",
                "Flexible input processing"
            ]
            
            areas_for_improvement = [
                "Test with real user scenarios",
                "Validate outputs with domain experts"
            ]
        
        else:
            overall_assessment = "Evaluation completed successfully"
            deployment_readiness = True
            recommendation = "Solution ready for specialist evaluation"
            key_strengths = ["Generated code is ready for testing"]
            areas_for_improvement = ["Requires domain-specific evaluation"]
        
        return EvaluationSummary(
            overall_assessment=overall_assessment,
            solution_performance={
                "testing_method": simulation_result.testing_method.value,
                "confidence_level": simulation_result.confidence_level.value,
                "suitability_score": simulation_result.suitability_assessment.overall_score if simulation_result.suitability_assessment else None
            },
            deployment_readiness=deployment_readiness,
            recommendation=recommendation,
            key_strengths=key_strengths,
            areas_for_improvement=areas_for_improvement
        )
    
    def _determine_evaluation_status(
        self, 
        simulation_result: SimulationResult, 
        evaluation_summary: EvaluationSummary
    ) -> str:
        
        if evaluation_summary.deployment_readiness:
            if simulation_result.suitability_assessment and simulation_result.suitability_assessment.overall_score >= 0.8:
                return "ready_for_deployment"
            else:
                return "needs_minor_improvements"
        else:
            return "needs_significant_improvements"
    
    def _generate_next_steps(self, status: str) -> List[str]:
        
        if status == "ready_for_deployment":
            return [
                "Download the complete project files",
                "Follow the setup instructions for your environment", 
                "Deploy and begin testing with real humanitarian data",
                "Monitor performance and gather user feedback",
                "Scale usage based on initial results"
            ]
        elif status == "needs_minor_improvements":
            return [
                "Review the improvement suggestions",
                "Test with additional data or scenarios if possible",
                "Download the project when ready",
                "Consider gradual deployment approach"
            ]
        else:
            return [
                "Address the identified compatibility issues",
                "Return to development phase for a different solution",
                "Improve your dataset based on recommendations",
                "Consult with technical specialists if needed"
            ]
    
    async def regenerate_scenarios(
        self, 
        project: Project, 
        request: ScenarioRegenerationRequest
    ) -> List[TestingScenario]:
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise ValueError("No generated project found for scenario regeneration")
        
        generated_project_data = project.development_data["generated_project"]
        selected_solution = self._get_full_selected_solution(project)
        
        return await self._generate_testing_scenarios(project, generated_project_data, selected_solution, request.custom_scenarios)
    
    async def get_download_files(self, project: Project) -> ProjectDownloads:
        
        if not project.development_data or not project.development_data.get("generated_project"):
            raise ValueError("No generated project found for download")
        
        generated_project_data = project.development_data["generated_project"]
        
        return ProjectDownloads(
            complete_project=DownloadableFile(
                files=generated_project_data.get("files", {}),
                description="Complete source code and all project files"
            ),
            documentation=DownloadableFile(
                content=generated_project_data.get("documentation", ""),
                description="Comprehensive project documentation in Markdown format"
            ),
            setup_instructions=DownloadableFile(
                content=generated_project_data.get("setup_instructions", ""),
                description="Step-by-step setup and installation guide in Markdown format"
            ),
            deployment_guide=DownloadableFile(
                content=generated_project_data.get("deployment_guide", ""),
                description="Deployment instructions and best practices in Markdown format"
            ),
            ethical_assessment_guide=DownloadableFile(
                content=generated_project_data.get("ethical_assessment_guide", ""),
                description="Comprehensive ethical assessment and bias testing guide in Markdown format"
            ),
            technical_handover_package=DownloadableFile(
                content=generated_project_data.get("technical_handover_package", ""),
                description="Complete technical handover documentation for production teams in Markdown format"
            )
        )
    
    def _get_available_downloads(self, generated_project_data: Dict[str, Any]) -> List[str]:
        
        downloads = []
        
        if generated_project_data.get("files"):
            downloads.append("complete_project")
        if generated_project_data.get("documentation"):
            downloads.append("documentation")
        if generated_project_data.get("setup_instructions"):
            downloads.append("setup_instructions")
        if generated_project_data.get("deployment_guide"):
            downloads.append("deployment_guide")
        if generated_project_data.get("ethical_assessment_guide"):
            downloads.append("ethical_assessment_guide")
        if generated_project_data.get("technical_handover_package"):
            downloads.append("technical_handover_package")
        
        return downloads
    
    def _create_simulation_explanation(
        self, 
        testing_method: TestingMethod, 
        dataset_statistics: Optional[Dict[str, Any]] = None,
        custom_scenarios: Optional[List[str]] = None
    ) -> SimulationExplanation:
        
        if testing_method == TestingMethod.DATASET:
            methodology = "Dataset Suitability Analysis"
            data_usage = "Dataset statistics are analyzed locally to assess compatibility with the AI solution requirements, without sending raw data to servers."
            calculation_basis = [
                "Feature compatibility assessment based on required vs available columns",
                "Data volume analysis comparing available rows to technique requirements", 
                "Data quality evaluation based on completeness and consistency",
                "No actual model training - assessment based on established ML best practices"
            ]
            limitations = [
                "Suitability assessment based on data characteristics, not actual model performance",
                "Real performance requires proper training and validation with your specific data",
                "Assessment provides guidance for technical implementation decisions",
                "Final validation should be done by technical specialists"
            ]
        else:
            methodology = "Scenario-Based Capability Assessment"
            data_usage = "AI-generated scenarios demonstrate expected solution behavior in typical humanitarian contexts."
            
            if custom_scenarios:
                calculation_basis = [
                    "Custom scenarios provided by user integrated with generated examples",
                    "Solution capabilities analyzed against humanitarian use case requirements",
                    "Expected behavior modeled based on AI technique characteristics",
                    "Contextual response patterns appropriate for humanitarian applications"
                ]
            else:
                calculation_basis = [
                    "Scenarios generated based on project context and AI technique capabilities",
                    "Humanitarian best practices incorporated into test cases",
                    "Typical input-output patterns for this solution type",
                    "Expected behavior modeling without actual data processing"
                ]
            
            limitations = [
                "Scenarios demonstrate expected capabilities, not guaranteed performance",
                "Real effectiveness depends on actual implementation and user data",
                "User testing with real scenarios is essential for validation",
                "Results guide expectations rather than predict specific outcomes"
            ]
        
        return SimulationExplanation(
            methodology=methodology,
            data_usage=data_usage,
            calculation_basis=calculation_basis,
            limitations=limitations
        )
    
    async def _generate_testing_scenarios(
        self, 
        project: Project, 
        generated_project_data: Dict[str, Any], 
        selected_solution: Dict[str, Any],
        custom_scenarios: Optional[List[str]] = None
    ) -> List[TestingScenario]:
        
        custom_context = ""
        if custom_scenarios:
            custom_context = f"\nUSER SCENARIOS: {', '.join(custom_scenarios)}\nIncorporate these specific scenarios."
        
        prompt = f"""
        Generate testing scenarios for this humanitarian AI solution:
        
        PROJECT: {project.title} - {project.description}
        SOLUTION: {selected_solution.get('title', 'AI Solution')}
        AI TECHNIQUE: {selected_solution.get('ai_technique', 'classification')}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        {custom_context}
        
        Generate 4 comprehensive testing scenarios:
        {{
            "scenarios": [
                {{
                    "name": "Scenario name that reflects humanitarian context",
                    "description": "What this scenario tests in humanitarian terms",
                    "input_description": "Specific type of humanitarian data or situation",
                    "process_description": "How the AI processes this humanitarian input",
                    "expected_outcome": "Detailed expected output for humanitarian use",
                    "success_criteria": "How to measure success in humanitarian impact",
                    "humanitarian_impact": "Direct benefit to humanitarian operations"
                }}
            ]
        }}
        
        Respond with ONLY valid JSON.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            data = self._extract_json_from_response(response)
            scenarios = data.get("scenarios", [])
            
            testing_scenarios = []
            for scenario in scenarios:
                testing_scenarios.append(TestingScenario(
                    name=scenario.get("name", ""),
                    description=scenario.get("description", ""),
                    input_description=scenario.get("input_description", ""),
                    process_description=scenario.get("process_description", ""),
                    expected_outcome=scenario.get("expected_outcome", ""),
                    success_criteria=scenario.get("success_criteria", ""),
                    humanitarian_impact=scenario.get("humanitarian_impact", "")
                ))
            
            return testing_scenarios
            
        except Exception as e:
            logger.error(f"Failed to generate testing scenarios: {e}")
            return []
    
    async def _generate_example_scenarios(
        self, 
        project: Project, 
        generated_project_data: Dict[str, Any], 
        selected_solution: Dict[str, Any],
        custom_scenarios: Optional[List[str]] = None
    ) -> List[ExampleScenario]:
        
        custom_context = ""
        if custom_scenarios:
            custom_context = f"\nUSER SCENARIOS: {', '.join(custom_scenarios)}\nIncorporate these scenarios."
        
        prompt = f"""
        Generate example scenarios showing how this AI solution works:
        
        PROJECT: {project.title} - {project.description}
        SOLUTION: {selected_solution.get('title', 'AI Solution')}
        AI TECHNIQUE: {selected_solution.get('ai_technique', 'classification')}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        {custom_context}
        
        Generate 4 realistic examples:
        {{
            "examples": [
                {{
                    "scenario_name": "Humanitarian scenario name",
                    "input_description": "Specific humanitarian input example",
                    "process_description": "How the AI processes this input",
                    "expected_output": "Detailed expected output example",
                    "humanitarian_impact": "Specific humanitarian benefit"
                }}
            ]
        }}
        
        Respond with ONLY valid JSON.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            data = self._extract_json_from_response(response)
            examples = data.get("examples", [])
            
            scenarios = []
            for example in examples:
                scenarios.append(ExampleScenario(
                    scenario_name=example.get("scenario_name", ""),
                    input_description=example.get("input_description", ""),
                    process_description=example.get("process_description", ""),
                    expected_output=example.get("expected_output", ""),
                    humanitarian_impact=example.get("humanitarian_impact", "")
                ))
            
            return scenarios
        except Exception as e:
            logger.error(f"Failed to parse example scenarios: {e}")
            return []
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        
        content = content.strip()
        
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        json_start = -1
        json_end = -1
        
        for i, char in enumerate(content):
            if char == '{':
                json_start = i
                break
        
        if json_start != -1:
            brace_count = 0
            for i in range(json_start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
        
        if json_start != -1 and json_end != -1:
            json_content = content[json_start:json_end]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract valid JSON from response: {content[:200]}...")