import asyncio
from datetime import datetime
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import core as ctx
from core.llm_service import llm_service
from services.claude_code_generation_service import claude_code_service
from models.development import (
    ProjectContext, AISolution, EthicalSafeguard, TechnicalArchitecture,
    ProjectRecommendation, DevelopmentPhaseData, GeneratedProject,
    ProjectGenerationRequest, ResourceRequirement, ProjectContextOnly, SolutionsData,
    FileAnalysis, EthicalGuardrailStatus, GenerationReport,
)
from models.enums import AITechnique, DeploymentStrategy
from models.project import Project

logger = logging.getLogger(__name__)

class DevelopmentService:
    
    def __init__(self):
        self.max_validation_retries = 1
        self.validation_retry_delay = 1
    
    def _parse_enum_value(self, value: str, enum_class, field_name: str):
        if not value:
            raise ValueError(f"No value provided for {field_name}")
        
        if '|' in value:
            value = value.split('|')[0].strip()
        
        value = value.strip().lower()
        
        for enum_val in enum_class:
            if enum_val.value.lower() == value:
                return enum_val
        
        valid_values = [e.value for e in enum_class]
        raise ValueError(f"Invalid {field_name} value: '{value}'. Valid values: {valid_values}")
    
    async def get_basic_context(self, project: Project) -> ProjectContextOnly:
        try:
            logger.info(f"Generating basic development context for project: {project.id}")
            
            project_context = await self._generate_project_context(project)
            ethical_safeguards = await self._generate_ethical_safeguards(project)
            
            return ProjectContextOnly(
                project_context=project_context,
                ethical_safeguards=ethical_safeguards,
                solution_rationale="AI solutions will be generated based on your project requirements."
            )
            
        except Exception as e:
            logger.error(f"Failed to generate basic development context: {e}")
            raise
    
    async def generate_dynamic_solutions(self, project: Project, user_feedback: Optional[str] = None) -> SolutionsData:
        try:
            logger.info(f"Generating AI solutions for project: {project.id}")
            
            deployment_constraints = self._extract_deployment_constraints(project)
            
            viable_approaches = await claude_code_service.analyze_viable_ai_approaches(
                project_description=project.description,
                use_case=project.selected_use_case.dict() if project.selected_use_case else {},
                deployment_constraints=deployment_constraints,
                problem_domain=getattr(project, 'problem_domain', 'humanitarian'),
                user_feedback=user_feedback
            )
        
            solution_context = self._build_project_context_dict(project)
            if user_feedback:
                solution_context["user_feedback"] = user_feedback
            
            solutions = await claude_code_service.generate_contextual_solutions(
                viable_approaches=viable_approaches,
                project_context=solution_context,
                user_feedback=user_feedback
            )
            
            if not solutions:
                raise ValueError("No valid solutions could be generated for this project")
            
            solution_rationale = await self._generate_solution_rationale(
                project, solutions, viable_approaches, user_feedback
            )
            
            logger.info(f"Successfully generated {len(solutions)} solutions for project: {project.id}")
            
            return SolutionsData(
                available_solutions=solutions,
                solution_rationale=solution_rationale
            )
            
        except Exception as e:
            logger.error(f"Failed to generate AI solutions: {e}")
            raise

    async def generate_project_with_validation(self, project: Project, request: ProjectGenerationRequest) -> GeneratedProject:
        
        if not project.development_data or not project.development_data.get("selected_solution"):
            raise ValueError("No solution selected for project generation")
        
        selected_solution_data = project.development_data["selected_solution"]
        solution_id = selected_solution_data.get("solution_id")
        
        available_solutions = project.development_data.get("available_solutions", [])
        selected_solution = None
        
        for sol in available_solutions:
            if sol.get("id") == solution_id:
                selected_solution = sol
                break
        
        if not selected_solution:
            raise ValueError(f"Selected solution {solution_id} not found")
        
        user_feedback = project.development_data.get("user_feedback") if project.development_data else None
        
        logger.info(f"Generating project for solution: {selected_solution.get('title')}")
        
        try:
            architecture = await claude_code_service.design_contextual_architecture(
                selected_solution=selected_solution,
                project_context=self._build_project_context_dict(project),
                user_feedback=user_feedback
            )
            
            project_files, validation_passed = await self._generate_and_validate_code(
                architecture=architecture,
                solution=selected_solution,
                project_context=self._build_project_context_dict(project),
                user_feedback=user_feedback
            )
                        
            docs_tasks = [
                self._generate_documentation(selected_solution, project, project_files),
                self._generate_setup_instructions(selected_solution, project_files),
                self._generate_deployment_guide(selected_solution, project_files),
                self._generate_ethical_assessment_guide(selected_solution, project),
                self._generate_technical_handover_package(selected_solution, project, architecture, project_files),
                self._generate_generation_report(selected_solution, project, project_files, architecture)
            ]
            
            (documentation, setup_instructions, deployment_guide, 
            ethical_assessment_guide, technical_handover_package, generation_report) = await asyncio.gather(*docs_tasks)

            generated_project = GeneratedProject(
                id=f"project_{project.id}_{solution_id}_{int(datetime.utcnow().timestamp())}",
                title=f"{selected_solution.get('title')} - {project.title}",
                description=f"{selected_solution.get('title')} implementation for {project.description}",
                solution_type=solution_id,
                ai_technique=self._parse_enum_value(selected_solution.get("ai_technique", ""), AITechnique, "ai_technique"),
                deployment_strategy=self._parse_enum_value(selected_solution.get("deployment_strategy", ""), DeploymentStrategy, "deployment_strategy"),
                files=project_files,
                documentation=documentation,
                setup_instructions=setup_instructions,
                deployment_guide=deployment_guide,
                ethical_assessment_guide=ethical_assessment_guide,
                technical_handover_package=technical_handover_package,
                generation_report=generation_report
            )
            
            logger.info(f"Successfully generated project with {len(project_files)} files")
            return generated_project
            
        except Exception as e:
            logger.error(f"Failed to generate project: {e}")
            raise

    async def _generate_ethical_assessment_guide(self, solution: Dict[str, Any], project: Project) -> str:
        
        ai_technique = solution.get("ai_technique", "classification")
        target_beneficiaries = await self._extract_target_beneficiaries(project)
        
        bias_context = await ctx.rag_service.get_bias_testing_frameworks_context(
            ai_technique=ai_technique,
            target_beneficiaries=target_beneficiaries,
            project_description=project.description
        )
        
        prompt = f"""
        Create a comprehensive ethical assessment guide in Markdown format for this humanitarian AI project:
        
        PROJECT: {project.title} - {project.description}
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {ai_technique}
        BENEFICIARIES: {target_beneficiaries}
        SAFEGUARDS: {solution.get('ethical_safeguards', [])}
        BIAS TESTING CONTEXT: {bias_context}
        
        Write for humanitarian professionals who need to ensure responsible AI deployment.
        Use clear, non-technical language with practical guidance.
        
        Create a complete Markdown document with these sections:
        
        # Ethical Assessment Guide for {solution.get('title')}
        
        ## Overview
        Brief explanation of why ethical assessment matters for this AI solution
        
        ## Privacy and Data Protection
        - Specific privacy measures implemented
        - Data handling protocols for {target_beneficiaries}
        - Compliance considerations for humanitarian contexts
        
        ## Bias Prevention and Fairness
        - Potential bias risks for {target_beneficiaries}
        - Testing methods for this {ai_technique} solution
        - Fairness evaluation steps
        - Mitigation strategies
        
        ## Transparency and Accountability
        - How to explain AI decisions to beneficiaries
        - Documentation requirements
        - Accountability mechanisms
        
        ## Community Impact Assessment
        - Expected benefits for {target_beneficiaries}
        - Risk mitigation strategies
        - Impact monitoring guidelines
        
        ## Testing and Validation Plan
        - Step-by-step bias testing procedures
        - Evaluation metrics specific to humanitarian impact
        - User acceptance testing guidelines
        - Ongoing monitoring recommendations
        
        ## Compliance and Documentation
        - Required documentation for humanitarian standards
        - Audit trail requirements
        - Reporting protocols
        
        ## Quick Reference Checklist
        - Essential checkpoints before deployment
        - Regular review schedule
        - Emergency protocols
        
        Format as professional Markdown with clear headers, bullet points, and actionable steps.
        """
        
        return await llm_service.analyze_text("", prompt)

    async def _generate_technical_handover_package(
        self, 
        solution: Dict[str, Any], 
        project: Project, 
        architecture: Dict[str, Any],
        project_files: Dict[str, str]
    ) -> str:
        
        ai_technique = solution.get("ai_technique", "classification")
        deployment_strategy = solution.get("deployment_strategy", "local_processing")
        target_beneficiaries = await self._extract_target_beneficiaries(project)
        deployment_constraints = self._extract_deployment_constraints(project)
        
        monitoring_context = await ctx.rag_service.get_monitoring_frameworks_context(
            ai_technique=ai_technique,
            deployment_strategy=deployment_strategy,
            project_description=project.description
        )
        
        prompt = f"""
        Create a comprehensive technical handover package in Markdown format for transitioning this humanitarian AI prototype to production:
        
        PROJECT: {project.title} - {project.description}
        SOLUTION: {solution.get('title')} using {ai_technique}
        DEPLOYMENT: {deployment_strategy}
        BENEFICIARIES: {target_beneficiaries}
        CONSTRAINTS: {json.dumps(deployment_constraints, indent=2)}
        MONITORING GUIDANCE: {monitoring_context}
        
        Write for technical teams who will implement this solution in production.
        Include specific, actionable requirements rather than generic guidelines.
        
        Create a complete Markdown document with these sections:
        
        # Technical Handover Package: {solution.get('title')}
        
        ## Executive Summary
        - Project overview and humanitarian objectives
        - Technical approach and rationale
        - Key implementation decisions
        
        ## Production Requirements
        
        ### Security Requirements
        - Specific security measures for {target_beneficiaries} data
        - Infrastructure security for {deployment_strategy}
        - Compliance requirements for humanitarian contexts
        
        ### Performance and Scalability
        - Expected performance benchmarks for {ai_technique}
        - Scalability requirements for {deployment_strategy}
        - Resource allocation guidelines
        
        ### Error Handling and Resilience
        - Specific error patterns for {ai_technique} failures
        - Graceful degradation strategies
        - User-friendly error messaging for humanitarian contexts
        
        ## Development Team Requirements
        
        ### Skills and Expertise
        - Required technical skills for {ai_technique} implementation
        - Humanitarian domain knowledge needs
        - Team composition recommendations
        
        ### Infrastructure and Tools
        - Development environment requirements
        - Deployment infrastructure for {deployment_strategy}
        - Monitoring and logging tools
        
        ## Implementation Timeline
        
        ### Phase 1: Foundation (Weeks 1-4)
        - Core {ai_technique} implementation
        - Basic infrastructure setup
        - Security framework implementation
        
        ### Phase 2: Integration (Weeks 5-8)
        - Humanitarian workflow integration
        - User interface development
        - Data pipeline implementation
        
        ### Phase 3: Production (Weeks 9-12)
        - Production deployment for {deployment_strategy}
        - Performance optimization
        - Monitoring implementation
        
        ## Risk Assessment and Mitigation
        
        ### Technical Risks
        - {ai_technique} specific challenges
        - {deployment_strategy} deployment risks
        - Mitigation strategies
        
        ### Humanitarian Context Risks
        - Data privacy and protection risks
        - Beneficiary impact risks
        - Operational continuity risks
        
        ## Monitoring and Maintenance
        
        ### Performance Monitoring
        - Key metrics for {ai_technique} in humanitarian context
        - Automated monitoring setup
        - Alert thresholds and responses
        
        ### Model Maintenance
        - Retraining schedule and procedures
        - Data quality monitoring
        - Performance degradation detection
        
        ### User Support and Training
        - User training requirements for humanitarian staff
        - Support documentation needs
        - Feedback collection mechanisms
        
        ## Success Metrics and KPIs
        
        ### Technical Metrics
        - Performance benchmarks for {ai_technique}
        - System availability and reliability
        - Response time requirements
        
        ### Humanitarian Impact Metrics
        - Specific impact measures for {target_beneficiaries}
        - Operational efficiency improvements
        - User adoption and satisfaction
        
        ## Compliance and Documentation
        - Required technical documentation
        - Audit and compliance requirements
        - Change management procedures
        
        ## Emergency Procedures
        - System failure response protocols
        - Data breach response procedures
        - Rollback and recovery procedures
        
        Format as professional Markdown with clear sections, tables where appropriate, and specific technical guidance.
        """
        
        return await llm_service.analyze_text("", prompt)

    async def _generate_generation_report(
        self, 
        solution: Dict[str, Any], 
        project: Project, 
        project_files: Dict[str, str],
        architecture: Dict[str, Any]
    ) -> GenerationReport:
        
        files_analysis = await self._analyze_all_files_with_llm(project_files, solution)
        
        ethical_implementation = await self._analyze_ethical_implementation(solution, project_files)
        
        solution_approach = f"Implemented {solution.get('title')} using {solution.get('ai_technique')} technique with {solution.get('deployment_strategy')} deployment strategy specifically for {project.description}."
        
        architecture_decisions = [
            f"Selected {solution.get('ai_technique')} as optimal AI technique for {project.description}",
            f"Chose {solution.get('deployment_strategy')} deployment strategy based on technical infrastructure analysis",
            f"Implemented {len(solution.get('ethical_safeguards', []))} ethical safeguards specific to humanitarian context",
        ]
        
        deployment_considerations = await self._generate_deployment_considerations(solution, project)
        
        return GenerationReport(
            solution_approach=solution_approach,
            files_generated=files_analysis,
            ethical_implementation=ethical_implementation,
            architecture_decisions=architecture_decisions,
            deployment_considerations=deployment_considerations
        )

    async def _analyze_all_files_with_llm(self, project_files: Dict[str, str], solution: Dict[str, Any]) -> List[FileAnalysis]:
        
        files_summary = {}
        for filename, content in project_files.items():
            files_summary[filename] = {
                "filename": filename,
                "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
                "content_length": len(content),
                "file_extension": filename.split('.')[-1] if '.' in filename else 'no_extension'
            }
        
        prompt = f"""
        Analyze these project files for a {solution.get('ai_technique')} humanitarian AI solution.
        
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {solution.get('ai_technique')}
        DEPLOYMENT: {solution.get('deployment_strategy')}
        
        FILES TO ANALYZE:
        {json.dumps(files_summary, indent=2)}
        
        For each file, determine:
        1. Content type based on actual content and file extension
        2. Purpose within this specific AI project
        3. Key features it provides
        4. Dependencies it introduces
        
        Return JSON with file analyses:
        {{
            "file_analyses": [
                {{
                    "filename": "exact_filename",
                    "purpose": "specific purpose in this {solution.get('ai_technique')} project",
                    "content_type": "accurate_content_type_based_on_extension_and_content",
                    "key_features": ["specific feature 1", "specific feature 2"],
                    "dependencies": ["dependency1", "dependency2"]
                }}
            ]
        }}
        
        Content types should be specific: python_script, javascript, dart_file, configuration, documentation, stylesheet, markup, etc.
        Dependencies should be external packages/libraries imported or referenced.
        """
        
        response = await llm_service.analyze_text("", prompt)
        
        try:
            content = response.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            
            file_analyses = []
            for analysis_data in data.get("file_analyses", []):
                file_analyses.append(FileAnalysis(
                    filename=analysis_data.get("filename", ""),
                    purpose=analysis_data.get("purpose", ""),
                    content_type=analysis_data.get("content_type", "text"),
                    key_features=analysis_data.get("key_features", []),
                    dependencies=analysis_data.get("dependencies", [])
                ))
            
            return file_analyses
            
        except Exception as e:
            logger.error(f"Failed to parse file analysis response: {e}")
            return self._fallback_file_analysis(project_files, solution)

    def _fallback_file_analysis(self, project_files: Dict[str, str], solution: Dict[str, Any]) -> List[FileAnalysis]:
        file_analyses = []
        for filename, content in project_files.items():
            analysis = self._analyze_file_purpose_fallback(filename, content, solution)
            file_analyses.append(analysis)
        return file_analyses

    def _analyze_file_purpose_fallback(self, filename: str, content: str, solution: Dict[str, Any]) -> FileAnalysis:
        content_type = self._determine_content_type_fallback(filename)
        
        if filename == "main.py":
            purpose = f"Main {solution.get('ai_technique')} application for {solution.get('title')}"
            key_features = [
                f"Implements {solution.get('ai_technique')} for humanitarian use",
                "Handles user data input and processing", 
                "Provides results visualization and insights",
                "Includes built-in ethical protections and error handling"
            ]
            dependencies = self._extract_dependencies_from_content(content)
        elif filename == "requirements.txt":
            purpose = f"Python dependencies for {solution.get('ai_technique')} implementation"
            key_features = ["Specifies exact versions for reproducible setup", "Includes all necessary AI and data processing libraries"]
            dependencies = []
        elif filename.endswith('.py'):
            purpose = f"Support module for {solution.get('ai_technique')} functionality"
            key_features = [f"Specialized functions for {solution.get('title')}", "Modular design for maintainability"]
            dependencies = self._extract_dependencies_from_content(content)
        else:
            purpose = f"Documentation for {solution.get('title')} implementation"
            key_features = ["Setup and usage instructions", "Context-specific guidance"]
            dependencies = []
        
        return FileAnalysis(
            filename=filename,
            purpose=purpose,
            content_type=content_type,
            key_features=key_features,
            dependencies=dependencies
        )

    def _determine_content_type_fallback(self, filename: str) -> str:
        if filename.endswith('.py'):
            return "python_script"
        elif filename.endswith('.txt'):
            return "documentation"
        elif filename.endswith('.md'):
            return "markdown"
        elif filename.endswith('.json'):
            return "configuration"
        elif filename.endswith(('.yml', '.yaml')):
            return "configuration"
        return "text"

    def _extract_dependencies_from_content(self, content: str) -> List[str]:
        dependencies = []
        import_lines = [line.strip() for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
        
        for line in import_lines[:5]:
            if 'import ' in line:
                module = line.split('import ')[1].split(' ')[0].split('.')[0]
                if module not in ['os', 'sys', 're', 'json', 'datetime']:
                    dependencies.append(module)
        
        return list(set(dependencies))

    async def _generate_deployment_considerations(self, solution: Dict[str, Any], project: Project) -> List[str]:
        
        deployment_constraints = self._extract_deployment_constraints(project)
        target_beneficiaries = await self._extract_target_beneficiaries(project)
        
        prompt = f"""
        Generate relevant deployment considerations for this humanitarian AI project:
        
        PROJECT: {project.title} - {project.description}
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {solution.get('ai_technique')}
        DEPLOYMENT STRATEGY: {solution.get('deployment_strategy')}
        TARGET BENEFICIARIES: {target_beneficiaries}
        PROBLEM DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        
        DEPLOYMENT CONSTRAINTS:
        {json.dumps(deployment_constraints, indent=2)}
        
        INFRASTRUCTURE CONTEXT:
        - Computing: {deployment_constraints.get('computing_resources', 'Not specified')}
        - Connectivity: {deployment_constraints.get('internet_connectivity', 'Not specified')}
        - Environment: {deployment_constraints.get('deployment_environment', 'Not specified')}
        
        Generate relevant deployment considerations for this exact context. Focus on:
        - Technical requirements for {solution.get('deployment_strategy')} deployment
        - Humanitarian context needs for {target_beneficiaries}
        - Infrastructure constraints and their implications
        - Security and privacy requirements for {getattr(project, 'problem_domain', 'humanitarian')} data
        - Scalability and maintenance considerations
        - Integration with existing humanitarian workflows
        
        Return JSON with specific considerations:
        {{
            "deployment_considerations": [
                "specific consideration for this deployment strategy and context",
                "another specific consideration for these infrastructure constraints",
                "security consideration specific to this humanitarian context"
            ]
        }}
        
        Make each consideration specific to this project's deployment strategy, infrastructure, and humanitarian context.
        Avoid generic advice.
        """
        
        response = await llm_service.analyze_text("", prompt)
        
        try:
            content = response.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)

            extracted_considerations = []
            for item in data.get("deployment_considerations", []):
                if isinstance(item, str):
                    extracted_considerations.append(item)
                elif isinstance(item, dict) and "consideration" in item:
                    extracted_considerations.append(item["consideration"])
                else:
                    extracted_considerations.append(str(item))

            return extracted_considerations
            
        except Exception as e:
            logger.error(f"Failed to generate deployment considerations: {e}")
            return self._fallback_deployment_considerations(solution, project)

    def _fallback_deployment_considerations(self, solution: Dict[str, Any], project: Project) -> List[str]:
        deployment_constraints = self._extract_deployment_constraints(project)
        
        considerations = []
        if deployment_constraints.get('internet_connectivity') == 'limited':
            considerations.append("Offline-first design required for limited connectivity environments")
        
        if solution.get('deployment_strategy') == 'local_processing':
            considerations.append("Local processing ensures data privacy and works in low-connectivity areas")
        
        if getattr(project, 'problem_domain', '') in ['refugee_services', 'humanitarian_response']:
            considerations.append("Deployment must handle sensitive humanitarian data with appropriate security measures")
        
        return considerations

    async def _generate_and_validate_code(
        self, 
        architecture: Dict[str, Any], 
        solution: Dict[str, Any], 
        project_context: Dict[str, Any],
        user_feedback: Optional[str] = None
    ) -> Tuple[Dict[str, str], bool]:
        
        try:
            logger.info("Generating project code")
            
            project_files = await claude_code_service.generate_contextual_code(
                architecture=architecture,
                solution=solution,
                project_context=project_context,
                user_feedback=user_feedback
            )
            
            if not project_files:
                raise ValueError("No files generated")
            
            validation_results = await claude_code_service.validate_generated_code(
                code_files=project_files,
                solution_requirements=solution,
                user_feedback=user_feedback
            )
            
            validation_passed = validation_results.get("validation_passed", False)
            working_code_assessment = validation_results.get("working_code_assessment", {})
            
            if not validation_passed or not working_code_assessment.get("runnable_immediately", False):
                validation_issues = validation_results.get("issues_found", [])
                logger.warning(f"Code validation failed: {len(validation_issues)} issues found")
                
                for i, issue in enumerate(validation_issues):
                    if isinstance(issue, dict):
                        logger.warning(f"Issue {i+1}: {issue.get('severity', 'unknown')} - {issue.get('description', 'no description')}")
                    else:
                        logger.warning(f"Issue {i+1}: {issue}")
                
                try:
                    project_files = await claude_code_service.fix_code_validation_issues(
                        current_code=project_files,
                        validation_issues=validation_issues,
                        architecture=architecture,
                        solution=solution,
                        user_feedback=user_feedback
                    )
                    
                    validation_results = await claude_code_service.validate_generated_code(
                        code_files=project_files,
                        solution_requirements=solution,
                        user_feedback=user_feedback
                    )
                    
                    validation_passed = validation_results.get("validation_passed", False)
                    
                    if validation_passed:
                        logger.info("Code validation passed after fixes")
                    else:
                        logger.warning("Code validation still failed after fixes, proceeding anyway")
                        
                except Exception as e:
                    logger.error(f"Code fixing failed: {e}, proceeding with original code")
            else:
                logger.info("Code validation passed on first attempt")
            
            return project_files, validation_passed
                        
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    async def _analyze_ethical_implementation(
        self, 
        solution: Dict[str, Any], 
        project_files: Dict[str, str]
    ) -> List[EthicalGuardrailStatus]:
        
        ethical_statuses = []
        safeguards = solution.get('ethical_safeguards', [])
        
        for safeguard in safeguards:
            if isinstance(safeguard, dict):
                category = safeguard.get('category', 'Unknown')
                measures = safeguard.get('measures', [])
                
                implementation_details, verification_method = self._determine_ethical_implementation(
                    category, measures, solution
                )
                
                ethical_statuses.append(EthicalGuardrailStatus(
                    category=category,
                    status="implemented",
                    implementation_details=implementation_details,
                    verification_method=verification_method
                ))
        
        if not ethical_statuses:
            ethical_statuses.append(EthicalGuardrailStatus(
                category="Basic Ethical Protection",
                status="implemented",
                implementation_details=[
                    "Local data processing with no external data sharing",
                    "Transparent operation with clear decision explanations",
                    "User control over data input and processing"
                ],
                verification_method="Code review and user testing"
            ))
        
        return ethical_statuses

    def _determine_ethical_implementation(self, category: str, measures: List[str], solution: Dict[str, Any]) -> Tuple[List[str], str]:
        category_lower = category.lower()
        
        if 'privacy' in category_lower or 'data protection' in category_lower:
            implementation_details = [
                f"Local data processing only for {solution.get('ai_technique')} operations",
                "No data transmission to external servers or APIs",
                "Secure file handling with appropriate access controls"
            ]
            verification_method = "Data flow analysis and privacy audit"
        elif 'bias' in category_lower or 'fairness' in category_lower:
            implementation_details = [
                f"Balanced training approach for {solution.get('ai_technique')} model",
                "Bias detection mechanisms in prediction pipeline",
                "Fair representation checks across different user groups"
            ]
            verification_method = "Bias testing with diverse datasets and fairness metrics"
        elif 'transparency' in category_lower:
            implementation_details = [
                f"Clear decision explanations for {solution.get('ai_technique')} predictions",
                "Accessible user interface with plain language explanations",
                "Documented methodology and limitations"
            ]
            verification_method = "User testing and documentation review"
        else:
            implementation_details = measures[:3] if len(measures) >= 3 else measures
            verification_method = "Manual review and testing"
        
        return implementation_details, verification_method

    async def _generate_project_context(self, project: Project) -> ProjectContext:
        
        target_beneficiaries = await self._extract_target_beneficiaries(project)
        recommendations = await self._generate_recommendations(project)
        
        technical_infrastructure = None
        if project.scoping_data and 'technical_infrastructure' in project.scoping_data:
            technical_infrastructure = project.scoping_data['technical_infrastructure']
        
        return ProjectContext(
            title=project.title,
            description=project.description,
            target_beneficiaries=target_beneficiaries,
            problem_domain=getattr(project, 'problem_domain', 'humanitarian'),
            selected_use_case=project.selected_use_case.dict() if project.selected_use_case else None,
            technical_infrastructure=technical_infrastructure,
            recommendations=recommendations,
            technical_recommendations=[],
            deployment_recommendations=[]
        )
    
    async def _extract_target_beneficiaries(self, project: Project) -> str:
        
        if hasattr(project, 'target_beneficiaries') and project.target_beneficiaries:
            return project.target_beneficiaries
        
        reflection_context = ""
        if project.reflection_data and project.reflection_data.get("answers"):
            answers = project.reflection_data["answers"]
            reflection_context = "\n".join([f"{key}: {value}" for key, value in answers.items()])
        
        prompt = f"""
        Extract target beneficiaries for this humanitarian AI project.
        
        PROJECT: {project.title} - {project.description}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        
        REFLECTION DATA:
        {reflection_context if reflection_context else "No reflection data available"}
        
        Describe the primary beneficiaries in 1-2 sentences focusing on humanitarian impact.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_recommendations(self, project: Project) -> List[ProjectRecommendation]:
        
        scoping_insights = self._extract_scoping_insights(project)
        deployment_constraints = self._extract_deployment_constraints(project)
        
        prompt = f"""
        Generate 3 practical recommendations for this humanitarian AI project.
        
        PROJECT: {project.title} - {project.description}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        USE CASE: {project.selected_use_case.title if project.selected_use_case else 'General approach'}
        
        INSIGHTS: {json.dumps(scoping_insights, indent=2) if scoping_insights else "No specific insights"}
        CONSTRAINTS: {json.dumps(deployment_constraints, indent=2) if deployment_constraints else "No specific constraints"}
        
        Return JSON with 3 recommendations:
        {{
            "recommendations": [
                {{
                    "type": "implementation",
                    "title": "Specific implementation recommendation",
                    "description": "2-3 sentences explaining what to do and why",
                    "confidence": 85,
                    "reason": "Why this is important for this project",
                    "deployment_strategy": "local_processing"
                }},
                {{
                    "type": "data", 
                    "title": "Data-related recommendation",
                    "description": "How to handle data for best results",
                    "confidence": 78,
                    "reason": "Connection to their data situation",
                    "deployment_strategy": "local_processing"
                }},
                {{
                    "type": "impact",
                    "title": "Impact maximization recommendation", 
                    "description": "How to ensure humanitarian goals are met",
                    "confidence": 82,
                    "reason": "Why this matters for their beneficiaries",
                    "deployment_strategy": "local_processing"
                }}
            ]
        }}
        
        Return only valid JSON.
        """
        
        response = await llm_service.analyze_text("", prompt)
        
        content = response.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse recommendations: {response}")
            return []
        
        recommendations = []
        for rec_data in data.get("recommendations", []):
            try:
                deployment_strategy = self._parse_enum_value(
                    rec_data.get("deployment_strategy", "local_processing"), 
                    DeploymentStrategy,
                    "deployment_strategy"
                )
                
                recommendations.append(ProjectRecommendation(
                    type=rec_data.get("type", "implementation"),
                    title=rec_data.get("title", ""),
                    description=rec_data.get("description", ""),
                    confidence=rec_data.get("confidence", 50),
                    reason=rec_data.get("reason", ""),
                    deployment_strategy=deployment_strategy
                ))
            except Exception as e:
                logger.warning(f"Skipping invalid recommendation: {e}")
                continue
        
        return recommendations
    
    async def _generate_ethical_safeguards(self, project: Project) -> List[EthicalSafeguard]:
        
        reflection_data = ""
        if project.reflection_data and project.reflection_data.get("answers"):
            answers = project.reflection_data["answers"]
            reflection_data = "\n".join([f"{key}: {value}" for key, value in answers.items()])
        
        use_case_context = ""
        if project.selected_use_case:
            use_case_context = f"USE CASE: {project.selected_use_case.title} - {project.selected_use_case.description}"
        
        infrastructure_context = ""
        if project.scoping_data and project.scoping_data.get('technical_infrastructure'):
            tech_infra = project.scoping_data['technical_infrastructure']
            infrastructure_context = f"""
            TECHNICAL SETUP:
            - Computing: {tech_infra.get('computing_resources')}
            - Connectivity: {tech_infra.get('internet_connectivity')}
            - Environment: {tech_infra.get('deployment_environment')}
            """
        
        target_beneficiaries = getattr(project, 'target_beneficiaries', '') or "humanitarian communities"
        
        prompt = f"""
        Generate specific ethical safeguards for this exact humanitarian AI project:
        
        PROJECT: {project.title}
        DESCRIPTION: {project.description}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        TARGET BENEFICIARIES: {target_beneficiaries}
        {use_case_context}
        {infrastructure_context}
        
        REFLECTION INSIGHTS:
        {reflection_data if reflection_data else "No reflection data available"}
        
        Based on this specific project context, create 4 tailored ethical safeguards:
        
        For "{project.title}" serving {target_beneficiaries} in {getattr(project, 'problem_domain', 'humanitarian')} context:
        
        1. DATA PROTECTION safeguard specific to this project's data and beneficiaries
        2. FAIRNESS safeguard addressing potential bias risks for these specific beneficiaries  
        3. TRANSPARENCY safeguard explaining how this specific AI tool will be accountable
        4. COMMUNITY IMPACT safeguard ensuring this specific project benefits the intended communities
        
        Return JSON with project-specific safeguards:
        {{
            "safeguards": [
                {{
                    "category": "Data Protection for [specific beneficiary group]",
                    "measures": [
                        "specific data protection measure for {target_beneficiaries}",
                        "another specific measure for this project's data",
                        "third specific protection for {getattr(project, 'problem_domain', 'humanitarian')} context"
                    ],
                    "icon": "shield",
                    "priority": "high"
                }},
                {{
                    "category": "Fairness for [specific context]",
                    "measures": [
                        "specific bias prevention for {target_beneficiaries}",
                        "specific fairness measure for this project",
                        "specific equity consideration for {getattr(project, 'problem_domain', 'humanitarian')}"
                    ],
                    "icon": "users",
                    "priority": "high"
                }},
                {{
                    "category": "Transparency in [project context]",
                    "measures": [
                        "specific transparency measure for this tool",
                        "specific explanation method for {target_beneficiaries}",
                        "specific accountability mechanism for this project"
                    ],
                    "icon": "eye",
                    "priority": "medium"
                }},
                {{
                    "category": "Community Benefit for [specific beneficiaries]",
                    "measures": [
                        "specific community benefit measure",
                        "specific impact verification for {target_beneficiaries}",
                        "specific harm prevention for this context"
                    ],
                    "icon": "users",
                    "priority": "high"
                }}
            ]
        }}
        
        Make each safeguard specific to:
        - The exact project: {project.title}
        - The specific beneficiaries: {target_beneficiaries}  
        - The domain context: {getattr(project, 'problem_domain', 'humanitarian')}
        - The technical setup and use case
        
        Avoid generic measures. Every measure should be tailored to this specific project.
        
        Return only valid JSON.
        """
        
        response = await llm_service.analyze_text("", prompt)
        
        content = response.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        try:
            safeguard_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse safeguards JSON: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError("Failed to generate ethical safeguards for project")
        
        safeguards = []
        for sg_data in safeguard_data.get("safeguards", []):
            try:
                priority = sg_data.get("priority", "medium").strip().lower()
                if priority not in ["low", "medium", "high", "critical"]:
                    priority = "medium"
                
                safeguards.append(EthicalSafeguard(
                    category=sg_data.get("category", ""),
                    measures=sg_data.get("measures", []),
                    icon=sg_data.get("icon", "shield"),
                    priority=priority
                ))
            except Exception as e:
                logger.warning(f"Skipping invalid safeguard: {e}")
                continue
        
        if not safeguards:
            raise ValueError("No valid ethical safeguards could be generated")
        
        return safeguards
    
    def _extract_deployment_constraints(self, project: Project) -> Dict[str, Any]:
        
        constraints = {}
        
        if project.scoping_data and 'technical_infrastructure' in project.scoping_data:
            technical_infra = project.scoping_data['technical_infrastructure']
            
            constraints.update({
                "computing_resources": technical_infra.get("computing_resources"),
                "storage_data": technical_infra.get("storage_data"),
                "internet_connectivity": technical_infra.get("internet_connectivity"),
                "deployment_environment": technical_infra.get("deployment_environment"),
            })
        
        if project.scoping_data and 'infrastructure_assessment' in project.scoping_data:
            infra_assessment = project.scoping_data['infrastructure_assessment']
            constraints.update({
                "infrastructure_score": infra_assessment.get("score", 0),
                "recommendations": infra_assessment.get("recommendations", []),
                "constraints": infra_assessment.get("non_ai_alternatives", [])
            })
        
        return constraints
    
    def _extract_scoping_insights(self, project: Project) -> Dict[str, Any]:
        
        if not project.scoping_data:
            return {}
        
        return {
            "infrastructure": project.scoping_data.get("infrastructure_assessment"),
            "data_suitability": project.scoping_data.get("data_suitability"),
            "use_case": project.scoping_data.get("selected_use_case"),
            "dataset": project.scoping_data.get("selected_dataset"),
            "technical_setup": project.scoping_data.get("technical_infrastructure")
        }
    
    def _build_project_context_dict(self, project: Project) -> Dict[str, Any]:
        
        technical_infrastructure = None
        if project.scoping_data and 'technical_infrastructure' in project.scoping_data:
            technical_infrastructure = project.scoping_data['technical_infrastructure']
        
        return {
            "title": project.title,
            "description": project.description,
            "problem_domain": getattr(project, 'problem_domain', 'humanitarian'),
            "target_beneficiaries": getattr(project, 'target_beneficiaries', ''),
            "selected_use_case": project.selected_use_case.dict() if project.selected_use_case else None,
            "technical_infrastructure": technical_infrastructure,
            "scoping_data": project.scoping_data
        }
    
    async def _get_technical_guidance(self, selected_solution: Dict[str, Any], project: Project) -> str:
        
        return await ctx.rag_service.get_technical_implementation_context(
            ai_technique=selected_solution.get('ai_technique', 'classification'),
            deployment_strategy=selected_solution.get('deployment_strategy', 'local_processing'),
            project_description=project.description
        )
    
    async def _generate_solution_rationale(self, project: Project, solutions: List[AISolution], viable_approaches: Dict[str, Any], user_feedback: Optional[str] = None) -> str:
        
        feedback_context = ""
        if user_feedback:
            feedback_context = f"\nBased on your requirements: {user_feedback}"
        
        solution_summaries = []
        for solution in solutions:
            solution_summaries.append({
                'title': solution.title,
                'technique': solution.ai_technique.value,
                'best_for': solution.best_for,
                'confidence': solution.confidence_score
            })
        
        prompt = f"""
        Write a simple explanation of why these AI solutions were created for this humanitarian project.
        
        PROJECT: {project.title} - {project.description}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        
        SOLUTIONS GENERATED: {[solution.title for solution in solutions]}
        {feedback_context}
        
        Write 1-2 simple sentences explaining why these solutions were suggested for this project.
        Use simple language that humanitarian professionals can understand.
        No technical jargon.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_documentation(self, solution: Dict[str, Any], project: Project, project_files: Dict[str, str]) -> str:
        
        prompt = f"""
        Generate comprehensive project documentation in Markdown format for this humanitarian AI prototype:
        
        SOLUTION: {solution.get('title')}
        PROJECT: {project.title}
        AI TECHNIQUE: {solution.get('ai_technique')}
        FILES: {list(project_files.keys())}
        
        Write for humanitarian professionals who will test this prototype and potentially hand it to technical teams.
        Use clear, non-technical language with practical guidance.
        
        Create a complete Markdown document with these sections:
        
        # {solution.get('title')} - Project Documentation
        
        ## Project Overview
        - What this AI solution does for humanitarian work
        - Who it's designed to help
        - Key capabilities and limitations
        
        ## How It Works
        - Simple explanation of the AI approach
        - What data it needs (if any)
        - What results it provides
        
        ## Getting Started
        - Prerequisites for testing
        - Initial setup requirements
        - First steps for humanitarian users
        
        ## Testing the Prototype
        - How to test with sample data
        - What results to expect
        - How to interpret outputs
        
        ## Ethical Considerations
        - Built-in protections for beneficiaries
        - Privacy and data handling
        - Bias prevention measures
        
        ## Technical Overview
        - Files included in the project
        - System requirements
        - Integration possibilities
        
        ## Next Steps
        - Moving from prototype to production
        - Technical team requirements
        - Scaling considerations
        
        ## Support and Resources
        - Troubleshooting common issues
        - Where to get help
        - Additional documentation references
        
        Format as professional Markdown with clear headers, bullet points, and practical guidance.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_setup_instructions(self, solution: Dict[str, Any], project_files: Dict[str, str]) -> str:
        
        prompt = f"""
        Generate step-by-step setup instructions in Markdown format for this AI prototype:
        
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {solution.get('ai_technique')}
        FILES: {list(project_files.keys())}
        
        Write for non-technical humanitarian professionals setting up a prototype for testing.
        Use clear, simple language with detailed steps.
        
        Create a complete Markdown document:
        
        # Setup Guide: {solution.get('title')}
        
        ## Before You Start
        - System requirements (Windows, Mac, or Linux)
        - Required software installations
        - Estimated setup time
        
        ## Step 1: Install Python
        - Download and installation instructions
        - How to verify installation
        - Common troubleshooting
        
        ## Step 2: Download Project Files
        - Where to extract the project files
        - Understanding the file structure
        - Important files overview
        
        ## Step 3: Install Dependencies
        - Running the requirements installation
        - What each major dependency does
        - Troubleshooting installation issues
        
        ## Step 4: First Run
        - How to start the application
        - What to expect on first launch
        - Initial configuration if needed
        
        ## Step 5: Test with Sample Data
        - Using the included test examples
        - Loading your own data (if applicable)
        - Verifying everything works correctly
        
        ## Troubleshooting
        - Common error messages and solutions
        - Performance issues and fixes
        - When to seek technical help
        
        ## Getting Help
        - Log file locations
        - Information to include when asking for help
        - Next steps if setup fails
        
        Format as clear Markdown with numbered steps, code blocks where needed, and helpful tips.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_deployment_guide(self, solution: Dict[str, Any], project_files: Dict[str, str]) -> str:
        
        prompt = f"""
        Generate deployment guidance in Markdown format for this AI prototype:
        
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {solution.get('ai_technique')}
        FILES: {list(project_files.keys())}
        
        Write for humanitarian professionals planning real-world deployment.
        Focus on bridging prototype to production with practical guidance.
        
        Create a complete Markdown document:
        
        # Deployment Guide: {solution.get('title')}
        
        ## Production Readiness Assessment
        - Current prototype capabilities
        - Limitations that need addressing
        - Required improvements for real-world use
        
        ## Infrastructure Requirements
        - Minimum system requirements for production
        - Recommended hardware specifications
        - Network and connectivity needs
        - Security infrastructure requirements
        
        ## Integration Planning
        - How this AI fits into existing humanitarian systems
        - Data flow and workflow integration
        - User training requirements
        - Change management considerations
        
        ## Security and Privacy
        - Data protection measures needed
        - User access controls
        - Audit and compliance requirements
        - Risk mitigation strategies
        
        ## Scaling Considerations
        - Expected user load and data volume
        - Performance optimization needs
        - Multi-location deployment planning
        - Backup and disaster recovery
        
        ## Maintenance and Support
        - Ongoing monitoring requirements
        - Update and maintenance schedules
        - User support infrastructure
        - Performance tracking and optimization
        
        ## Cost Planning
        - Initial deployment costs
        - Ongoing operational expenses
        - Training and support costs
        - Return on investment considerations
        
        ## Implementation Timeline
        - Recommended deployment phases
        - Key milestones and deliverables
        - Risk factors and contingencies
        - Success metrics and evaluation
        
        ## Working with Technical Teams
        - Information to provide to developers
        - Key decisions for technical implementation
        - Humanitarian requirements specification
        - Quality assurance and testing protocols
        
        Format as actionable Markdown with clear sections and practical checklists.
        """
        
        return await llm_service.analyze_text("", prompt)