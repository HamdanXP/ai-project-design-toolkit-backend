from datetime import datetime
import json
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from core.llm_service import llm_service
from services.rag_service import rag_service
from models.development import (
    ProjectContext, AISolution, EthicalSafeguard, TechnicalArchitecture,
    ProjectRecommendation, DevelopmentPhaseData, GeneratedProject,
    ProjectGenerationRequest, AITechnique, DeploymentStrategy, 
    ComplexityLevel, ResourceRequirement, ProjectContextOnly, SolutionsData
)
from models.project import Project

logger = logging.getLogger(__name__)

class DevelopmentService:
    """Enhanced development service with guided flexibility for truly dynamic project generation"""
    
    def __init__(self):
        # Template mappings for different solution types (kept for reference but not used as constraints)
        self.ai_technique_reference = {
            AITechnique.CLASSIFICATION: "classification_reference",
            AITechnique.COMPUTER_VISION: "computer_vision_reference",
            AITechnique.NATURAL_LANGUAGE_PROCESSING: "nlp_reference",
            AITechnique.LARGE_LANGUAGE_MODEL: "llm_reference",
            AITechnique.TIME_SERIES_ANALYSIS: "time_series_reference",
            AITechnique.RECOMMENDATION_SYSTEM: "recommendation_reference",
            AITechnique.ANOMALY_DETECTION: "anomaly_detection_reference",
            AITechnique.REGRESSION: "regression_reference",
            AITechnique.CLUSTERING: "clustering_reference",
            AITechnique.OPTIMIZATION: "optimization_reference",
            AITechnique.MULTI_MODAL: "multi_modal_reference",
            AITechnique.REINFORCEMENT_LEARNING: "rl_reference"
        }
    
    def _parse_enum_value(self, value: str, enum_class, field_name: str):
        """Parse and validate enum values, handling multiple values or invalid responses"""
        if not value:
            return list(enum_class)[0]
        
        if '|' in value:
            value = value.split('|')[0].strip()
        
        value = value.strip().lower()
        
        for enum_val in enum_class:
            if enum_val.value.lower() == value:
                return enum_val
        
        for enum_val in enum_class:
            if value in enum_val.value.lower() or enum_val.value.lower() in value:
                logger.warning(f"Partial match for {field_name}: '{value}' -> '{enum_val.value}'")
                return enum_val
        
        logger.warning(f"Invalid {field_name} value: '{value}'. Using fallback: '{list(enum_class)[0].value}'")
        return list(enum_class)[0]
    
    async def get_basic_context(self, project: Project) -> ProjectContextOnly:
        """Get basic development context (FAST) - no AI solution generation"""
        try:
            logger.info(f"Generating basic development context for project: {project.id}")
            
            use_case_analysis = await self._analyze_use_case_with_llm(project)
            deployment_analysis = await self._analyze_deployment_with_llm(project)
            
            project_context = await self._generate_dynamic_project_context(
                project, use_case_analysis, deployment_analysis
            )
            
            ethical_safeguards = await self._generate_basic_ethical_safeguards(project)
            
            return ProjectContextOnly(
                project_context=project_context,
                ethical_safeguards=ethical_safeguards,
                solution_rationale="AI solutions will be generated when you proceed to the next step."
            )
            
        except Exception as e:
            logger.error(f"Failed to generate basic development context: {e}")
            raise
    
    async def generate_solutions(self, project: Project) -> SolutionsData:
        """Generate AI solutions (SLOW) - called only when user needs them"""
        try:
            logger.info(f"Generating AI solutions for project: {project.id}")
            
            use_case_analysis = await self._analyze_use_case_with_llm(project)
            deployment_analysis = await self._analyze_deployment_with_llm(project)
            
            solutions = await self._generate_dynamic_solutions(
                project, use_case_analysis, deployment_analysis
            )
            
            solution_rationale = await self._generate_dynamic_solution_rationale(
                project, solutions, use_case_analysis, deployment_analysis
            )
            
            return SolutionsData(
                available_solutions=solutions,
                solution_rationale=solution_rationale
            )
            
        except Exception as e:
            logger.error(f"Failed to generate AI solutions: {e}")
            raise
    
    async def get_development_context(self, project: Project) -> DevelopmentPhaseData:
        """Legacy method for backward compatibility - generates everything at once"""
        try:
            logger.info(f"Generating full development context for project: {project.id}")
            
            basic_context = await self.get_basic_context(project)
            solutions_data = await self.generate_solutions(project)
            
            return DevelopmentPhaseData(
                project_context=basic_context.project_context,
                available_solutions=solutions_data.available_solutions,
                ethical_safeguards=basic_context.ethical_safeguards,
                solution_rationale=solutions_data.solution_rationale
            )
            
        except Exception as e:
            logger.error(f"Failed to generate full development context: {e}")
            raise

    async def generate_project(self, project: Project, request: ProjectGenerationRequest) -> GeneratedProject:
        """ENHANCED: Generate project with guided flexibility - dynamic choices within proven patterns"""
        
        if not project.development_data or not project.development_data.get("selected_solution"):
            raise ValueError("No solution selected for project generation")
        
        selected_solution_data = project.development_data["selected_solution"]
        solution_id = selected_solution_data.get("solution_id")
        
        # Get the full solution details
        available_solutions = project.development_data.get("available_solutions", [])
        selected_solution = None
        
        for sol in available_solutions:
            if sol.get("id") == solution_id:
                selected_solution = sol
                break
        
        if not selected_solution:
            raise ValueError(f"Selected solution {solution_id} not found in available solutions")
        
        logger.info(f"Generating project with guided flexibility for solution: {selected_solution.get('title')}")
        
        generation_steps = []
        
        try:
            # Step 1: Make architectural decisions with guided flexibility
            generation_steps.append("Making informed architectural decisions for solution")
            project_structure = await self._generate_solution_specific_structure(selected_solution, project)
            
            # Step 2: Validate architectural feasibility
            generation_steps.append("Validating architectural decisions for feasibility")
            structure_validation = await self._validate_project_structure(project_structure, selected_solution, project)
            
            if not structure_validation["valid"]:
                logger.warning(f"Architecture validation issues: {structure_validation['issues']}")
                generation_steps.append(f"Architecture concerns noted: {len(structure_validation['issues'])} issues")
            
            # Step 3: Generate coordinated implementation
            generation_steps.append("Generating coordinated project implementation")
            project_files = await self._generate_coordinated_project_files(selected_solution, project, request, project_structure)
            
            # Step 4: Generate documentation reflecting actual implementation
            generation_steps.append("Creating implementation-specific documentation")
            documentation = await self._generate_implementation_documentation(selected_solution, project, project_files, project_structure)
            setup_instructions = await self._generate_setup_instructions(selected_solution, project, project_files)
            deployment_guide = await self._generate_deployment_guide(selected_solution, project, project_files)
            
            # Step 5: Generate ethical compliance reports
            generation_steps.append("Generating ethical audit and monitoring reports")
            ethical_audit_report = await self._generate_ethical_audit_report(selected_solution, project, project_files)
            bias_testing_plan = await self._generate_bias_testing_plan(selected_solution, project)
            monitoring_recommendations = await self._generate_monitoring_recommendations(selected_solution, project)
            
            # Step 6: Final implementation validation
            generation_steps.append("Validating complete implementation")
            validation_results = await self._validate_generated_project(selected_solution, project_files, project)
            
            if not validation_results["valid"]:
                logger.warning(f"Implementation validation issues: {validation_results['issues']}")
                documentation += f"\n\n## Implementation Notes\n{validation_results.get('report', 'See validation results')}"
            
            # Create the generated project
            generated_project = GeneratedProject(
                id=f"guided_{project.id}_{solution_id}_{int(datetime.utcnow().timestamp())}",
                title=f"{selected_solution.get('title')} - {project.title}",
                description=f"Intelligently generated {selected_solution.get('title')} implementation for {project.description}",
                solution_type=solution_id,
                ai_technique=self._parse_enum_value(selected_solution.get("ai_technique", ""), AITechnique, "ai_technique"),
                deployment_strategy=self._parse_enum_value(selected_solution.get("deployment_strategy", ""), DeploymentStrategy, "deployment_strategy"),
                files=project_files,
                documentation=documentation,
                setup_instructions=setup_instructions,
                deployment_guide=deployment_guide,
                ethical_audit_report=ethical_audit_report,
                bias_testing_plan=bias_testing_plan,
                monitoring_recommendations=monitoring_recommendations
            )
            
            tech_choices = project_structure.get("technology_decisions", {})
            logger.info(f"Successfully generated project with {len(project_files)} files using: {tech_choices.get('frontend_framework', 'default frontend')}, {tech_choices.get('backend_framework', 'default backend')}")
            return generated_project
            
        except Exception as e:
            generation_steps.append(f"Error during guided generation: {str(e)}")
            logger.error(f"Failed to generate project with guided flexibility: {e}")
            raise ValueError(f"Guided project generation failed: {e}")

    async def _generate_solution_specific_structure(
        self, 
        solution: Dict[str, Any], 
        project: Project
    ) -> Dict[str, Any]:
        """Generate project structure with guided flexibility - dynamic technology choices within proven patterns"""
        
        # Get comprehensive context for informed decisions
        technical_context = await rag_service.get_technical_implementation_context(
            ai_technique=solution.get("ai_technique", "classification"),
            deployment_strategy=solution.get("deployment_strategy", "cloud_native"),
            complexity_level=solution.get("complexity_level", "moderate"),
            project_description=project.description
        )
        
        case_studies_context = await rag_service.get_real_world_case_studies_context(
            ai_technique=solution.get("ai_technique", "classification"),
            problem_domain=getattr(project, 'problem_domain', 'humanitarian'),
            project_description=project.description
        )
        
        prompt = f"""
        Design an optimal project structure for this AI solution using guided flexibility.
        
        SOLUTION TO IMPLEMENT:
        - Title: {solution.get('title')}
        - AI Technique: {solution.get('ai_technique')}
        - Capabilities: {solution.get('capabilities', [])}
        - Deployment Strategy: {solution.get('deployment_strategy')}
        - Complexity: {solution.get('complexity_level')}
        
        PROJECT CONTEXT:
        - Title: {project.title}
        - Description: {project.description}
        - Domain: {getattr(project, 'problem_domain', 'humanitarian')}
        
        TECHNICAL GUIDANCE:
        {technical_context}
        
        REAL-WORLD EXAMPLES:
        {case_studies_context}
        
        DESIGN REQUIREMENTS:
        1. Choose optimal technology stack for this specific solution
        2. Determine necessary project components (not files yet)
        3. Consider humanitarian context and constraints
        4. Balance innovation with proven approaches
        
        Return your architectural decisions:
        {{
            "technology_decisions": {{
                "frontend_framework": "chosen framework and rationale",
                "backend_framework": "chosen framework and rationale", 
                "ai_implementation": "chosen approach for {solution.get('ai_technique')} and rationale",
                "data_processing": "chosen approach and rationale",
                "deployment_approach": "chosen approach for {solution.get('deployment_strategy')} and rationale",
                "database_choice": "chosen database type and rationale",
                "monitoring_tools": "chosen monitoring approach and rationale"
            }},
            "component_architecture": {{
                "core_components": [
                    {{
                        "name": "component_name",
                        "purpose": "what this component does for the solution",
                        "technology": "technology chosen for this component",
                        "critical": true/false,
                        "files_needed": ["general description of files, not specific paths"]
                    }}
                ],
                "integration_strategy": "how components work together",
                "data_flow": "how data flows through the system"
            }},
            "project_structure_approach": {{
                "organization_strategy": "how to organize the codebase",
                "separation_of_concerns": "how to separate different aspects",
                "configuration_approach": "how to handle configuration",
                "testing_strategy": "how to structure testing",
                "documentation_approach": "how to organize documentation"
            }},
            "implementation_priorities": [
                "priority 1 - most critical component",
                "priority 2 - second most important",
                "priority 3 - supporting components"
            ],
            "architecture_rationale": "detailed explanation of why these choices fit this specific solution"
        }}
        
        Focus on making informed technology choices rather than following templates.
        Consider the specific requirements of {solution.get('ai_technique')} implementation.
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            structure = json.loads(response)
            structure["solution_id"] = solution.get("id")
            structure["generated_at"] = datetime.utcnow().isoformat()
            return structure
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structure response: {response}")
            raise ValueError(f"Invalid JSON response for project structure: {e}")

    async def _validate_project_structure(
        self, 
        structure: Dict[str, Any], 
        solution: Dict[str, Any], 
        project: Project
    ) -> Dict[str, Any]:
        """Validate architectural decisions for feasibility and appropriateness"""
        
        tech_decisions = structure.get("technology_decisions", {})
        components = structure.get("component_architecture", {}).get("core_components", [])
        
        validation_issues = []
        validation_checks = []
        
        # Validate technology choices
        ai_technique = solution.get("ai_technique", "classification")
        deployment_strategy = solution.get("deployment_strategy", "cloud_native")
        
        prompt = f"""
        Validate these architectural decisions for a {ai_technique} solution:
        
        TECHNOLOGY DECISIONS:
        {json.dumps(tech_decisions, indent=2)}
        
        CORE COMPONENTS:
        {json.dumps([{"name": c["name"], "purpose": c["purpose"], "technology": c["technology"]} for c in components], indent=2)}
        
        SOLUTION REQUIREMENTS:
        - AI Technique: {ai_technique}
        - Deployment: {deployment_strategy}
        - Capabilities: {solution.get('capabilities', [])}
        
        Validate feasibility and appropriateness:
        {{
            "technology_compatibility": true/false,
            "ai_technique_support": true/false,
            "deployment_feasibility": true/false,
            "component_completeness": true/false,
            "integration_viability": true/false,
            "humanitarian_suitability": true/false,
            "potential_issues": ["issue1", "issue2"],
            "missing_components": ["component1", "component2"],
            "technology_concerns": ["concern1", "concern2"],
            "overall_viability": true/false,
            "confidence_score": 0.85
        }}
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            validation_data = json.loads(response)
            
            if not validation_data.get("ai_technique_support", False):
                validation_issues.append(f"Technology choices may not properly support {ai_technique}")
            else:
                validation_checks.append(f"✓ Technology stack supports {ai_technique}")
            
            if not validation_data.get("deployment_feasibility", False):
                validation_issues.append(f"Architecture may not support {deployment_strategy} deployment")
            else:
                validation_checks.append(f"✓ Architecture supports {deployment_strategy}")
            
            missing_components = validation_data.get("missing_components", [])
            validation_issues.extend([f"Missing component: {comp}" for comp in missing_components])
            
            potential_issues = validation_data.get("potential_issues", [])
            validation_issues.extend([f"Potential issue: {issue}" for issue in potential_issues])
            
            is_valid = validation_data.get("overall_viability", False) and len(validation_issues) <= 2
            
            return {
                "valid": is_valid,
                "issues": validation_issues,
                "checks_passed": validation_checks,
                "validation_data": validation_data,
                "total_components": len(components),
                "confidence_score": validation_data.get("confidence_score", 0.0)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation response: {response}")
            return {"valid": False, "issues": ["Could not validate architecture"], "checks_passed": []}

    async def _generate_coordinated_project_files(
        self, 
        solution: Dict[str, Any], 
        project: Project, 
        request: ProjectGenerationRequest,
        project_structure: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate all project files with coordination but within logical categories"""
        
        tech_decisions = project_structure.get("technology_decisions", {})
        components = project_structure.get("component_architecture", {}).get("core_components", [])
        structure_approach = project_structure.get("project_structure_approach", {})
        
        prompt = f"""
        Generate a complete, coordinated project implementation:
        
        SOLUTION TO IMPLEMENT:
        - Title: {solution.get('title')}
        - AI Technique: {solution.get('ai_technique')}
        - Capabilities: {solution.get('capabilities', [])}
        
        ARCHITECTURAL DECISIONS:
        - Frontend: {tech_decisions.get('frontend_framework', 'Not specified')}
        - Backend: {tech_decisions.get('backend_framework', 'Not specified')}
        - AI Implementation: {tech_decisions.get('ai_implementation', 'Not specified')}
        - Deployment: {tech_decisions.get('deployment_approach', 'Not specified')}
        
        REQUIRED COMPONENTS:
        {json.dumps([{"name": c["name"], "purpose": c["purpose"], "technology": c["technology"]} for c in components], indent=2)}
        
        ORGANIZATION APPROACH:
        {json.dumps(structure_approach, indent=2)}
        
        PROJECT CONTEXT:
        - Title: {project.title}
        - Description: {project.description}
        - Domain: {getattr(project, 'problem_domain', 'humanitarian')}
        
        Generate complete implementation with logical file organization:
        
        REQUIREMENTS:
        1. Use the chosen technology stack consistently
        2. Implement each required component properly
        3. Follow the organization strategy
        4. Ensure components integrate well
        5. Include proper configuration and setup
        6. Add appropriate testing structure
        7. Include deployment configuration
        8. Consider humanitarian context in implementation
        
        Organize files logically and generate complete content:
        {{
            "frontend/main_component.ext": "Complete frontend implementation using chosen framework",
            "backend/api_server.ext": "Complete backend implementation using chosen framework",
            "ai/model_implementation.ext": "Complete AI implementation for chosen technique",
            "config/settings.ext": "Complete configuration using chosen approach",
            "deployment/deploy_config.ext": "Complete deployment config for chosen strategy",
            "tests/test_main.ext": "Complete test implementation",
            "docs/README.md": "Complete project documentation",
            "requirements.txt": "Complete dependency list"
        }}
        
        IMPORTANT:
        - Generate working, production-ready code
        - Use specific file extensions appropriate for chosen technologies
        - Ensure all files work together as a cohesive system
        - Implement the actual AI technique and capabilities
        - Don't use placeholder code or TODOs
        
        Respond ONLY with valid JSON where keys are file paths and values are complete file content.
        No markdown formatting, no explanations.
        """
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = await llm_service.analyze_text("", prompt)
                generated_files = json.loads(response)
                
                # Validate the coordinated implementation
                validation_result = await self._validate_coordinated_implementation(
                    generated_files, components, solution, tech_decisions
                )
                
                if validation_result["valid"] or attempt == max_retries:
                    if not validation_result["valid"]:
                        logger.warning(f"Implementation validation failed on final attempt: {validation_result['issues']}")
                    return generated_files
                else:
                    logger.info(f"Implementation validation failed (attempt {attempt + 1}), retrying...")
                    prompt += f"\n\nPREVIOUS ATTEMPT ISSUES:\n{chr(10).join(validation_result['issues'])}\nPlease address these issues."
                    
            except json.JSONDecodeError as e:
                if attempt == max_retries:
                    logger.error(f"Failed to parse implementation response after {max_retries} retries")
                    return {"README.md": f"# {solution.get('title')} - Generation failed"}
                else:
                    logger.warning(f"JSON parsing failed (attempt {attempt + 1}), retrying...")
                    prompt += "\n\nIMPORTANT: Respond with VALID JSON only. Check your JSON syntax."
        
        return {}

    async def _validate_coordinated_implementation(
        self, 
        generated_files: Dict[str, str], 
        components: List[Dict], 
        solution: Dict[str, Any],
        tech_decisions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that the implementation properly coordinates and implements requirements"""
        
        issues = []
        checks = []
        
        # Check for component implementation
        component_names = [c["name"].lower() for c in components]
        files_content = " ".join(generated_files.values()).lower()
        
        implemented_components = 0
        for comp in components:
            comp_name = comp["name"].lower()
            if comp_name in files_content or any(comp_name in fname.lower() for fname in generated_files.keys()):
                implemented_components += 1
                checks.append(f"✓ {comp['name']} component implemented")
            else:
                issues.append(f"Missing {comp['name']} component implementation")
        
        # Check for technology consistency
        frontend_tech = tech_decisions.get("frontend_framework", "").lower()
        backend_tech = tech_decisions.get("backend_framework", "").lower()
        
        if frontend_tech and not any(frontend_tech.split()[0] in content.lower() for content in generated_files.values()):
            issues.append(f"Frontend technology ({frontend_tech}) not properly used")
        else:
            checks.append("✓ Frontend technology implemented")
        
        if backend_tech and not any(backend_tech.split()[0] in content.lower() for content in generated_files.values()):
            issues.append(f"Backend technology ({backend_tech}) not properly used")
        else:
            checks.append("✓ Backend technology implemented")
        
        # Check for AI technique implementation
        ai_technique = solution.get("ai_technique", "").lower()
        if not any(ai_technique in content.lower() for content in generated_files.values()):
            issues.append(f"AI technique ({ai_technique}) not implemented")
        else:
            checks.append(f"✓ {ai_technique} implementation found")
        
        # Check for basic project structure
        essential_files = ["readme", "requirements", "config"]
        found_essential = sum(1 for essential in essential_files 
                             if any(essential in fname.lower() for fname in generated_files.keys()))
        
        if found_essential < 2:
            issues.append("Missing essential project files (README, requirements, config)")
        else:
            checks.append("✓ Essential project files present")
        
        component_implementation_rate = implemented_components / len(components) if components else 0
        
        return {
            "valid": len(issues) <= 2 and component_implementation_rate >= 0.7,
            "issues": issues,
            "checks": checks,
            "component_implementation_rate": component_implementation_rate,
            "total_files": len(generated_files)
        }

    async def _generate_implementation_documentation(
        self, 
        solution: Dict[str, Any], 
        project: Project, 
        project_files: Dict[str, str],
        project_structure: Dict[str, Any]
    ) -> str:
        """Generate documentation that reflects the actual architectural choices and implementation"""
        
        tech_decisions = project_structure.get("technology_decisions", {})
        components = project_structure.get("component_architecture", {}).get("core_components", [])
        
        prompt = f"""
        Generate comprehensive documentation for this implemented project:
        
        SOLUTION: {solution.get('title')}
        PROJECT: {project.title}
        
        ARCHITECTURAL DECISIONS MADE:
        - Frontend: {tech_decisions.get('frontend_framework', 'Not specified')}
        - Backend: {tech_decisions.get('backend_framework', 'Not specified')}
        - AI Implementation: {tech_decisions.get('ai_implementation', 'Not specified')}
        - Data Processing: {tech_decisions.get('data_processing', 'Not specified')}
        - Deployment: {tech_decisions.get('deployment_approach', 'Not specified')}
        
        IMPLEMENTED COMPONENTS:
        {json.dumps([{"name": c["name"], "purpose": c["purpose"]} for c in components], indent=2)}
        
        ACTUAL FILES GENERATED: {list(project_files.keys())[:12]}
        
        ARCHITECTURE RATIONALE:
        {project_structure.get('architecture_rationale', 'No rationale provided')}
        
        Create comprehensive documentation covering:
        
        # Project Overview
        - Purpose and humanitarian impact
        - Chosen solution approach
        - Key capabilities implemented
        
        # Architecture Documentation
        - Technology choices and rationale
        - Component architecture
        - System integration approach
        - Data flow design
        
        # Implementation Details
        - AI technique implementation specifics
        - Key algorithms and models used
        - Integration patterns
        - Configuration approach
        
        # Usage Guide
        - How to run the system
        - API endpoints and interfaces
        - Input/output specifications
        - Example usage scenarios
        
        # Ethical Implementation
        - Built-in safeguards
        - Bias mitigation measures
        - Transparency features
        - Accountability mechanisms
        
        # Technical Considerations
        - Performance characteristics
        - Scalability design
        - Security measures
        - Monitoring capabilities
        
        Write in clear markdown that reflects the actual implementation choices made.
        """
        
        return await llm_service.analyze_text("", prompt)

    async def _validate_generated_project(
        self, 
        solution: Dict[str, Any], 
        project_files: Dict[str, str], 
        project: Project
    ) -> Dict[str, Any]:
        """Validate the complete project against solution requirements"""
        
        ai_technique = solution.get("ai_technique", "classification")
        
        # Get ethical frameworks context for validation
        ethical_context = await rag_service.get_ethical_frameworks_context(
            ai_technique=ai_technique,
            project_description=project.description,
            target_beneficiaries=await self._extract_target_beneficiaries_dynamically(project)
        )
        
        prompt = f"""
        Validate this complete project implementation:
        
        SOLUTION REQUIREMENTS:
        - Title: {solution.get('title')}
        - AI Technique: {ai_technique}
        - Capabilities: {solution.get('capabilities', [])}
        - Key Features: {solution.get('key_features', [])}
        
        IMPLEMENTATION ANALYSIS:
        - Total Files: {len(project_files)}
        - File Types: {list(set([f.split('.')[-1] for f in project_files.keys() if '.' in f]))}
        - Average File Size: {sum(len(content) for content in project_files.values()) // len(project_files) if project_files else 0} characters
        
        ETHICAL FRAMEWORKS CONTEXT:
        {ethical_context}
        
        Comprehensive validation:
        {{
            "solution_requirements_met": true/false,
            "ai_technique_properly_implemented": true/false,
            "capabilities_functional": true/false,
            "ethical_safeguards_present": true/false,
            "deployment_ready": true/false,
            "framework_compliance": true/false,
            "code_quality_score": 0.85,
            "implementation_completeness": 0.90,
            "missing_critical_elements": ["element1", "element2"],
            "implementation_strengths": ["strength1", "strength2"],
            "areas_for_improvement": ["improvement1", "improvement2"],
            "humanitarian_alignment": true/false,
            "overall_success": true/false,
            "confidence_level": 0.88,
            "validation_summary": "Brief summary of validation results"
        }}
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            validation_data = json.loads(response)
            
            issues = []
            if not validation_data.get("solution_requirements_met", False):
                issues.append("Solution requirements not fully met")
            if not validation_data.get("ai_technique_properly_implemented", False):
                issues.append(f"{ai_technique} not properly implemented")
            if not validation_data.get("ethical_safeguards_present", False):
                issues.append("Ethical safeguards missing or insufficient")
            if not validation_data.get("framework_compliance", False):
                issues.append("Framework compliance issues identified")
            
            missing_elements = validation_data.get("missing_critical_elements", [])
            issues.extend([f"Missing: {element}" for element in missing_elements])
            
            return {
                "valid": validation_data.get("overall_success", False) and len(issues) <= 1,
                "issues": issues,
                "validation_data": validation_data,
                "report": validation_data.get("validation_summary", "Validation completed"),
                "humanitarian_alignment": validation_data.get("humanitarian_alignment", False)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation response: {response}")
            return {
                "valid": False,
                "issues": ["Could not validate implementation"],
                "report": "Validation failed due to parsing error"
            }
    
    # Enhanced helper methods
    async def _analyze_use_case_with_llm(self, project: Project) -> Dict[str, Any]:
        """Use LLM to completely analyze the use case and determine requirements"""
        if not project.selected_use_case:
            raise ValueError("No use case selected - cannot generate development context")
        
        use_case = project.selected_use_case
        
        prompt = f"""
        You are an expert AI consultant analyzing a humanitarian AI project's use case.
        
        PROJECT CONTEXT:
        - Title: {project.title}
        - Description: {project.description}
        - Domain: {getattr(project, 'problem_domain', 'humanitarian')}
        
        SELECTED USE CASE:
        - Title: {use_case.title}
        - Description: {use_case.description}
        - Category: {use_case.category}
        - Source: {use_case.source if hasattr(use_case, 'source') else 'Unknown'}
        
        Analyze this use case and determine the most appropriate technical approach.
        
        AVAILABLE AI TECHNIQUES (choose ONE):
        - classification
        - computer_vision  
        - nlp
        - llm
        - time_series
        - recommendation
        - anomaly_detection
        - regression
        - clustering
        - optimization
        - multi_modal
        - reinforcement_learning
        
        COMPLEXITY LEVELS (choose ONE):
        - simple
        - moderate
        - advanced
        - enterprise
        
        Return detailed JSON analysis:
        {{
            "primary_ai_technique": "ONE_TECHNIQUE_FROM_LIST_ABOVE",
            "secondary_techniques": ["technique1", "technique2"],
            "complexity_level": "ONE_LEVEL_FROM_LIST_ABOVE",
            "data_types": ["text", "images", "time_series", "tabular", "audio", "sensor"],
            "processing_requirements": ["requirement1", "requirement2"],
            "technical_challenges": ["challenge1", "challenge2"],
            "success_factors": ["factor1", "factor2"],
            "humanitarian_considerations": ["consideration1", "consideration2"],
            "performance_requirements": ["requirement1", "requirement2"],
            "scalability_needs": ["need1", "need2"],
            "integration_requirements": ["requirement1", "requirement2"]
        }}

        CRITICAL: For primary_ai_technique and complexity_level, return EXACTLY ONE value from the provided lists.
        Do not use pipes or multiple values. Choose the single most appropriate option.
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
    
    async def _analyze_deployment_with_llm(self, project: Project) -> Dict[str, Any]:
        """Use LLM to analyze deployment constraints and generate smart strategies"""
        if not project.deployment_environment:
            raise ValueError("No deployment environment data - cannot generate development context")
        
        env = project.deployment_environment.dict()
        
        prompt = f"""
        You are an expert deployment strategist analyzing a humanitarian AI project's constraints.
        
        PROJECT: {project.title} - {project.description}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        
        DEPLOYMENT ENVIRONMENT:
        - Computing Resources: {env.get('computing_resources', 'unknown')}
        - Budget: {env.get('project_budget', 'unknown')}
        - Team Size: {env.get('team_size', 'unknown')}
        - Internet Connectivity: {env.get('reliable_internet_connection', 'unknown')}
        - Technical Skills: {env.get('technical_skills', 'unknown')}
        - AI Experience: {env.get('ai_ml_experience', 'unknown')}
        - Infrastructure: {env.get('local_technology_setup', 'unknown')}
        
        AVAILABLE DEPLOYMENT STRATEGIES (choose from these):
        - api_integration
        - cloud_native
        - edge_computing
        - hybrid_approach
        - serverless
        - federated_learning
        - local_processing
        
        Analyze these constraints and recommend intelligent deployment strategies.
        For each recommended strategy, choose EXACTLY ONE strategy from the list above.
        
        Return detailed strategic analysis:
        {{
            "recommended_strategies": [
                {{
                    "strategy": "ONE_STRATEGY_FROM_LIST_ABOVE",
                    "rationale": "specific reason why this strategy fits these exact constraints",
                    "confidence": 85,
                    "trade_offs": ["tradeoff1", "tradeoff2"]
                }}
            ],
            "constraint_optimizations": [
                {{
                    "constraint": "limited_compute",
                    "optimization": "specific optimization for this constraint",
                    "impact": "how this helps the project"
                }}
            ],
            "resource_recommendations": ["recommendation1", "recommendation2"],
            "cost_considerations": ["consideration1", "consideration2"],
            "technical_recommendations": ["recommendation1", "recommendation2"],
            "risk_mitigations": ["mitigation1", "mitigation2"]
        }}

        CRITICAL: For strategy fields, use EXACTLY ONE value from the deployment strategies list.
        Do not use pipes or multiple values.
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
    
    async def _generate_dynamic_project_context(self, project: Project, use_case_analysis: Dict[str, Any], deployment_analysis: Dict[str, Any]) -> ProjectContext:
        target_beneficiaries = await self._extract_target_beneficiaries_dynamically(project)
        recommendations = await self._generate_intelligent_recommendations_dynamically(project, use_case_analysis, deployment_analysis)
        
        return ProjectContext(
            title=project.title,
            description=project.description,
            target_beneficiaries=target_beneficiaries,
            problem_domain=getattr(project, 'problem_domain', 'humanitarian'),
            selected_use_case=project.selected_use_case.dict() if project.selected_use_case else None,
            use_case_analysis=use_case_analysis,
            deployment_environment=project.deployment_environment.dict() if project.deployment_environment else None,
            deployment_analysis=deployment_analysis,
            recommendations=recommendations,
            technical_recommendations=deployment_analysis.get("technical_recommendations", []),
            deployment_recommendations=[opt["optimization"] for opt in deployment_analysis.get("constraint_optimizations", [])]
        )
        
    async def _extract_target_beneficiaries_dynamically(self, project: Project) -> str:
        reflection_context = ""
        if project.reflection_data and project.reflection_data.get("answers"):
            answers = project.reflection_data["answers"]
            reflection_context = "\n".join([f"{key}: {value}" for key, value in answers.items()])
        
        prompt = f"""
        Extract and describe the target beneficiaries for this humanitarian AI project.
        
        PROJECT: {project.title} - {project.description}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        
        REFLECTION DATA:
        {reflection_context if reflection_context else "No reflection data available"}
        
        Based on this information, describe who the primary beneficiaries are and how they will benefit.
        Be specific about the humanitarian impact.
        
        Return 1-2 sentences describing the target beneficiaries.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_intelligent_recommendations_dynamically(
        self, 
        project: Project, 
        use_case_analysis: Dict[str, Any], 
        deployment_analysis: Dict[str, Any]
    ) -> List[ProjectRecommendation]:
        """Generate completely dynamic, intelligent recommendations"""
        
        reflection_context = ""
        if project.reflection_data and project.reflection_data.get("answers"):
            reflection_context = "\n".join([f"- {key}: {value}" for key, value in project.reflection_data["answers"].items()])
        
        deployment_constraints = deployment_analysis.get('constraint_optimizations', [])
        primary_strategy = deployment_analysis.get('recommended_strategies', [{}])[0].get('strategy', 'cloud_native')
        
        prompt = f"""
        Generate 3 highly specific, actionable recommendations for this humanitarian AI project. 
        
        PROJECT DETAILS:
        - Title: {project.title}
        - Description: {project.description}
        - Domain: {getattr(project, 'problem_domain', 'humanitarian')}
        - Selected Use Case: {project.selected_use_case.title if project.selected_use_case else 'General approach'}
        
        PROJECT REFLECTION INSIGHTS:
        {reflection_context or "No specific reflection data available"}
        
        TECHNICAL ANALYSIS:
        - Primary AI Technique Needed: {use_case_analysis.get('primary_ai_technique')}
        - Project Complexity: {use_case_analysis.get('complexity_level')}
        - Key Technical Challenges: {', '.join(use_case_analysis.get('technical_challenges', []))}
        - Best Deployment Strategy: {primary_strategy}
        
        DEPLOYMENT CONSTRAINTS:
        {json.dumps(deployment_constraints, indent=2) if deployment_constraints else "No specific constraints identified"}
        
        Return JSON with 3 strategic recommendations:
        {{
            "recommendations": [
                {{
                    "type": "technical",
                    "title": "Specific recommendation about the AI technique or implementation",
                    "description": "2-3 sentences explaining exactly what to do and why it matters for this project",
                    "confidence": 85,
                    "reason": "1-2 sentences explaining why this recommendation is tailored to this specific project context",
                    "deployment_strategy": "{primary_strategy}"
                }},
                {{
                    "type": "deployment", 
                    "title": "Specific recommendation about deployment or scaling",
                    "description": "2-3 sentences about deployment strategy tailored to their constraints",
                    "confidence": 78,
                    "reason": "1-2 sentences connecting this to their specific deployment environment",
                    "deployment_strategy": "{primary_strategy}"
                }},
                {{
                    "type": "impact",
                    "title": "Specific recommendation about maximizing humanitarian impact", 
                    "description": "2-3 sentences about ensuring the project achieves its humanitarian goals",
                    "confidence": 82,
                    "reason": "1-2 sentences explaining why this matters for their target beneficiaries",
                    "deployment_strategy": "{primary_strategy}"
                }}
            ]
        }}
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for recommendations: {response}")
            return []
        
        recommendations = []
        for rec_data in data.get("recommendations", []):
            try:
                deployment_strategy = self._parse_enum_value(
                    rec_data.get("deployment_strategy", primary_strategy), 
                    DeploymentStrategy,
                    "deployment_strategy"
                )
                
                recommendations.append(ProjectRecommendation(
                    type=rec_data.get("type", "technical"),
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
    
    async def _generate_basic_ethical_safeguards(self, project: Project) -> List[EthicalSafeguard]:
        """Generate basic ethical safeguards (lightweight operation)"""
        
        prompt = f"""
        Generate 3-4 basic ethical safeguard categories for this humanitarian AI project:
        
        PROJECT:
        - Title: {project.title}
        - Domain: {getattr(project, 'problem_domain', 'humanitarian')}
        - Description: {project.description}
        
        Generate fundamental ethical safeguards:
        {{
            "safeguards": [
                {{
                    "category": "category name",
                    "measures": ["measure1", "measure2"],
                    "icon": "icon_name",
                    "priority": "medium"
                }}
            ]
        }}

        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            safeguard_data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            safeguard_data = {"safeguards": []}
        
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
        
        return safeguards
    
    async def _generate_dynamic_solutions(
        self, 
        project: Project, 
        use_case_analysis: Dict[str, Any], 
        deployment_analysis: Dict[str, Any]
    ) -> List[AISolution]:
        """Generate 5 completely dynamic AI solutions using LLM + RAG context"""
        
        logger.info("Starting RAG-grounded AI solutions generation...")
        
        # GET HUMANITARIAN AI CONTEXT from RAG
        primary_ai_technique = use_case_analysis.get('primary_ai_technique', 'classification')
        primary_deployment = deployment_analysis.get('recommended_strategies', [{}])[0].get('strategy', 'cloud_native')
        target_beneficiaries = await self._extract_target_beneficiaries_dynamically(project)
        
        # Get comprehensive context from humanitarian AI knowledge base
        rag_context = await rag_service.get_comprehensive_development_context(
            project_description=project.description,
            ai_technique=primary_ai_technique,
            deployment_strategy=primary_deployment,
            complexity_level=use_case_analysis.get('complexity_level', 'moderate'),
            target_beneficiaries=target_beneficiaries,
            resource_constraints=deployment_analysis
        )
        
        # Get real-world case studies
        case_studies_context = await rag_service.get_real_world_case_studies_context(
            ai_technique=primary_ai_technique,
            problem_domain=getattr(project, 'problem_domain', 'humanitarian'),
            project_description=project.description
        )
        
        prompt = f"""
        You are an expert AI solution architect with access to humanitarian AI best practices and frameworks.
        
        PROJECT CONTEXT:
        - Title: {project.title}
        - Description: {project.description}
        - Domain: {getattr(project, 'problem_domain', 'humanitarian')}
        - Target Beneficiaries: {target_beneficiaries}
        
        USE CASE ANALYSIS:
        {json.dumps(use_case_analysis, indent=2)}
        
        DEPLOYMENT ANALYSIS:
        {json.dumps(deployment_analysis, indent=2)}
        
        HUMANITARIAN AI KNOWLEDGE BASE CONTEXT:
        
        SOLUTION GENERATION GUIDANCE:
        {rag_context.get('solution_generation', 'No specific guidance available')}
        
        ETHICAL FRAMEWORKS AND GUIDELINES AVAILABLE:
        {rag_context.get('ethical_frameworks', 'No specific frameworks available')}
        
        TECHNICAL IMPLEMENTATION PATTERNS:
        {rag_context.get('technical_implementation', 'No specific patterns available')}
        
        DEPLOYMENT BEST PRACTICES:
        {rag_context.get('deployment_best_practices', 'No specific practices available')}
        
        REAL-WORLD CASE STUDIES:
        {case_studies_context}
        
        Based on this humanitarian AI knowledge base and available frameworks/guidelines, generate 5 distinct AI solutions.
        
        IMPORTANT: 
        - Ground your solutions in the provided humanitarian AI best practices
        - Reference whatever frameworks/guidelines are mentioned in the context above
        - Don't assume specific frameworks - only use what's available in the context
        - Mark 1-2 solutions as "recommended": true based on alignment with available guidelines
        
        Generate 5 solutions that leverage available humanitarian AI knowledge:
        {{
            "solutions": [
                {{
                    "id": "unique_solution_id",
                    "title": "Solution title grounded in available humanitarian AI practices",
                    "description": "Description referencing best practices and frameworks from the provided context",
                    "ai_technique": "classification",
                    "complexity_level": "moderate", 
                    "deployment_strategy": "cloud_native",
                    "confidence_score": 85,
                    "capabilities": ["capability1 based on available guidance", "capability2"],
                    "key_features": ["feature1 from available best practices", "feature2"],
                    "best_for": "specific humanitarian use case from available knowledge",
                    "use_case_alignment": "alignment with available frameworks/guidelines",
                    "deployment_considerations": ["consideration1 from available best practices"],
                    "implementation_timeline": "timeline based on available case studies",
                    "maintenance_requirements": ["requirement1 from available frameworks"],
                    "external_apis": ["api1"],
                    "integration_complexity": "simple",
                    "recommended": false,
                    "humanitarian_alignment": "How this aligns with humanitarian AI principles from available knowledge",
                    "framework_references": ["Only frameworks/guidelines actually found in the provided context"]
                }}
            ]
        }}
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        
        solutions = []
        for solution_data in data.get("solutions", []):
            try:
                ai_technique = self._parse_enum_value(
                    solution_data.get("ai_technique", ""), 
                    AITechnique,
                    "ai_technique"
                )
                complexity_level = self._parse_enum_value(
                    solution_data.get("complexity_level", ""), 
                    ComplexityLevel,
                    "complexity_level"
                )
                deployment_strategy = self._parse_enum_value(
                    solution_data.get("deployment_strategy", ""), 
                    DeploymentStrategy,
                    "deployment_strategy"
                )
                
                tech_arch = await self._generate_dynamic_technical_architecture(
                    solution_data, use_case_analysis, deployment_analysis
                )
                
                resource_req = await self._generate_dynamic_resource_requirements(
                    solution_data, deployment_analysis
                )
                
                ethical_safeguards = await self._generate_solution_specific_ethical_safeguards(
                    solution_data, project, use_case_analysis
                )
                
                solution = AISolution(
                    id=solution_data.get("id", f"solution_{len(solutions)}"),
                    title=solution_data.get("title", ""),
                    description=solution_data.get("description", ""),
                    ai_technique=ai_technique,
                    complexity_level=complexity_level,
                    deployment_strategy=deployment_strategy,
                    recommended=solution_data.get("recommended", False),
                    confidence_score=solution_data.get("confidence_score", 50),
                    capabilities=solution_data.get("capabilities", []),
                    key_features=solution_data.get("key_features", []),
                    technical_architecture=tech_arch,
                    resource_requirements=resource_req,
                    best_for=solution_data.get("best_for", ""),
                    use_case_alignment=solution_data.get("use_case_alignment", ""),
                    deployment_considerations=solution_data.get("deployment_considerations", []),
                    ethical_safeguards=ethical_safeguards,
                    implementation_timeline=solution_data.get("implementation_timeline", ""),
                    maintenance_requirements=solution_data.get("maintenance_requirements", []),
                    external_apis=solution_data.get("external_apis", []),
                    integration_complexity=solution_data.get("integration_complexity", "moderate")
                )
                solutions.append(solution)
                
            except Exception as e:
                logger.warning(f"Skipping invalid solution: {e}")
                continue
        
        logger.info(f"Successfully generated {len(solutions)} AI solutions")
        return solutions
    
    async def _generate_dynamic_technical_architecture(
        self, 
        solution_data: Dict[str, Any], 
        use_case_analysis: Dict[str, Any], 
        deployment_analysis: Dict[str, Any]
    ) -> TechnicalArchitecture:
        """Generate technical architecture dynamically based on solution context"""
        
        prompt = f"""
        Generate technical architecture for this AI solution:
        
        SOLUTION:
        - Title: {solution_data.get('title', '')}
        - AI Technique: {solution_data.get('ai_technique', '')}
        - Deployment Strategy: {solution_data.get('deployment_strategy', '')}
        
        Generate architecture components:
        {{
            "frontend": "Frontend technology description",
            "backend": "Backend architecture description", 
            "ai_components": ["component1", "component2"],
            "data_processing": "Data processing description",
            "deployment": "Deployment architecture description",
            "monitoring": "Monitoring setup description"
        }}
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            arch_data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            arch_data = {}
        
        def ensure_string(value):
            if isinstance(value, dict):
                return ", ".join([f"{k}: {v}" for k, v in value.items()])
            elif isinstance(value, list):
                return ", ".join(map(str, value))
            elif value is None:
                return "Not specified"
            else:
                return str(value)
        
        ai_technique = self._parse_enum_value(
            solution_data.get("ai_technique", ""), 
            AITechnique,
            "ai_technique"
        )
        deployment_strategy = self._parse_enum_value(
            solution_data.get("deployment_strategy", ""), 
            DeploymentStrategy,
            "deployment_strategy"
        )
        
        return TechnicalArchitecture(
            ai_technique=ai_technique,
            deployment_strategy=deployment_strategy,
            frontend=ensure_string(arch_data.get("frontend", "Frontend not specified")),
            backend=ensure_string(arch_data.get("backend", "Backend not specified")),
            ai_components=arch_data.get("ai_components", []),
            data_processing=ensure_string(arch_data.get("data_processing", "Data processing not specified")),
            deployment=ensure_string(arch_data.get("deployment", "Deployment not specified")),
            monitoring=ensure_string(arch_data.get("monitoring", "Monitoring not specified"))
        )
    
    async def _generate_dynamic_resource_requirements(
        self, 
        solution_data: Dict[str, Any], 
        deployment_analysis: Dict[str, Any]
    ) -> ResourceRequirement:
        """Generate resource requirements dynamically"""
        
        prompt = f"""
        Determine resource requirements for this AI solution:
        
        SOLUTION:
        - Title: {solution_data.get('title', '')}
        - Complexity: {solution_data.get('complexity_level', '')}
        - Deployment: {solution_data.get('deployment_strategy', '')}
        
        Return requirements (choose exactly one value for each):
        {{
            "computing_power": "low",
            "storage_needs": "moderate", 
            "internet_dependency": "continuous",
            "technical_expertise": "intermediate",
            "budget_estimate": "medium"
        }}
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            req_data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            req_data = {}
        
        def validate_option(value, valid_options, field_name):
            if not value:
                return valid_options[0]
            value = value.strip().lower()
            if value in valid_options:
                return value
            for option in valid_options:
                if value in option or option in value:
                    return option
            logger.warning(f"Invalid {field_name} value: '{value}'. Using fallback: '{valid_options[0]}'")
            return valid_options[0]
        
        return ResourceRequirement(
            computing_power=validate_option(
                req_data.get("computing_power", ""), 
                ["low", "medium", "high"], 
                "computing_power"
            ),
            storage_needs=validate_option(
                req_data.get("storage_needs", ""), 
                ["minimal", "moderate", "extensive"], 
                "storage_needs"
            ),
            internet_dependency=validate_option(
                req_data.get("internet_dependency", ""), 
                ["offline", "periodic", "continuous"], 
                "internet_dependency"
            ),
            technical_expertise=validate_option(
                req_data.get("technical_expertise", ""), 
                ["basic", "intermediate", "advanced"], 
                "technical_expertise"
            ),
            budget_estimate=validate_option(
                req_data.get("budget_estimate", ""), 
                ["low", "medium", "high"], 
                "budget_estimate"
            )
        )
    
    async def _generate_solution_specific_ethical_safeguards(
        self, 
        solution_data: Dict[str, Any], 
        project: Project, 
        use_case_analysis: Dict[str, Any]
    ) -> List[EthicalSafeguard]:
        """Generate ethical safeguards grounded in AVAILABLE humanitarian AI frameworks"""
        
        ai_technique = solution_data.get('ai_technique', 'classification')
        target_beneficiaries = await self._extract_target_beneficiaries_dynamically(project)
        
        # GET ETHICAL FRAMEWORKS CONTEXT from RAG
        ethical_context = await rag_service.get_ethical_frameworks_context(
            ai_technique=ai_technique,
            project_description=project.description,
            target_beneficiaries=target_beneficiaries
        )
        
        prompt = f"""
        Generate specific ethical safeguards grounded in available humanitarian AI frameworks:
        
        PROJECT:
        - Domain: {getattr(project, 'problem_domain', 'humanitarian')}
        - Description: {project.description}
        - Target Beneficiaries: {target_beneficiaries}
        
        SOLUTION:
        - Title: {solution_data.get('title', '')}
        - AI Technique: {ai_technique}
        - Deployment: {solution_data.get('deployment_strategy', '')}
        
        AVAILABLE HUMANITARIAN AI ETHICAL FRAMEWORKS AND GUIDELINES:
        {ethical_context}
        
        USE CASE CONTEXT:
        - Humanitarian Considerations: {use_case_analysis.get('humanitarian_considerations', [])}
        
        Based on the available frameworks and guidelines mentioned in the context above, 
        generate 3-4 specific ethical safeguards:
        
        {{
            "safeguards": [
                {{
                    "category": "Category based on available frameworks (reference what's actually in the context)",
                    "measures": ["Specific measure based on available frameworks", "Another framework-based measure"],
                    "icon": "icon_name",
                    "priority": "medium",
                    "framework_reference": "Reference to frameworks actually mentioned in the context above"
                }}
            ]
        }}

        IMPORTANT: 
        - Only reference frameworks and guidelines that are actually mentioned in the context above
        - If no specific frameworks are available, focus on general humanitarian AI principles
        - Don't assume specific frameworks exist - use what's available in the provided context
        
        Respond ONLY with valid JSON. No markdown formatting, no explanations.
        """
        
        response = await llm_service.analyze_text("", prompt)
        try:
            safeguard_data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            safeguard_data = {"safeguards": []}
        
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
        
        return safeguards
    
    async def _generate_dynamic_solution_rationale(
        self, 
        project: Project, 
        solutions: List[AISolution], 
        use_case_analysis: Dict[str, Any], 
        deployment_analysis: Dict[str, Any]
    ) -> str:
        """Generate rationale for why these specific solutions were chosen"""
        
        prompt = f"""
        Create a compelling rationale explaining why these AI solutions were generated:
        
        PROJECT: {project.title} - {project.description}
        DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        
        KEY ANALYSIS INSIGHTS:
        - Primary AI Technique: {use_case_analysis.get('primary_ai_technique')}
        - Key Challenges: {use_case_analysis.get('technical_challenges', [])}
        - Deployment Constraints: {[opt.get('constraint') for opt in deployment_analysis.get('constraint_optimizations', [])]}
        
        GENERATED SOLUTIONS:
        {[{'title': s.title, 'technique': s.ai_technique.value, 'strategy': s.deployment_strategy.value} for s in solutions]}
        
        Write 2-3 sentences explaining why this specific set of solutions addresses the project's unique needs, constraints, and opportunities.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    # Documentation and Setup Generation Methods
    async def _generate_setup_instructions(
        self, 
        solution: Dict[str, Any], 
        project: Project, 
        project_files: Dict[str, str]
    ) -> str:
        """Generate detailed setup instructions based on actual implementation"""
        
        prompt = f"""
        Generate step-by-step setup instructions for this AI project:
        
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {solution.get('ai_technique')}
        DEPLOYMENT: {solution.get('deployment_strategy')}
        
        FILES INCLUDED: {list(project_files.keys())[:10]}
        
        Create detailed setup instructions including:
        1. Prerequisites and requirements
        2. Installation steps
        3. Configuration
        4. Initial setup and testing
        5. Common issues and solutions
        
        Write clear, step-by-step instructions based on the actual files generated.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_deployment_guide(
        self, 
        solution: Dict[str, Any], 
        project: Project, 
        project_files: Dict[str, str]
    ) -> str:
        """Generate deployment guide based on actual implementation"""
        
        deployment_files = {k: v for k, v in project_files.items() if 'deploy' in k.lower() or 'docker' in k.lower()}
        
        prompt = f"""
        Generate deployment guide for this AI solution:
        
        SOLUTION: {solution.get('title')}
        DEPLOYMENT STRATEGY: {solution.get('deployment_strategy')}
        
        DEPLOYMENT FILES: {list(deployment_files.keys())}
        ALL FILES: {list(project_files.keys())[:12]}
        
        Create deployment guide covering:
        1. Deployment prerequisites
        2. Environment setup
        3. Deployment steps
        4. Configuration management
        5. Monitoring and maintenance
        6. Scaling considerations
        
        Provide specific instructions for the deployment strategy based on actual generated files.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_ethical_audit_report(
        self, 
        solution: Dict[str, Any], 
        project: Project, 
        project_files: Dict[str, str]
    ) -> str:
        """Generate ethical audit report based on actual implementation"""
        
        ethical_files = {k: v for k, v in project_files.items() if 'ethic' in k.lower() or 'bias' in k.lower()}
        
        prompt = f"""
        Generate ethical audit report for this AI solution:
        
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {solution.get('ai_technique')}
        PROJECT DOMAIN: {getattr(project, 'problem_domain', 'humanitarian')}
        
        ETHICAL SAFEGUARDS: {solution.get('ethical_safeguards', [])}
        ETHICAL FILES GENERATED: {list(ethical_files.keys())}
        
        Create comprehensive ethical audit report covering:
        1. Ethical framework assessment
        2. Bias analysis and mitigation
        3. Privacy protection measures
        4. Transparency and explainability
        5. Fairness evaluation
        6. Risk assessment
        7. Compliance checklist
        8. Recommendations
        
        Provide detailed analysis and recommendations based on the actual implementation.
        """
        
        return await llm_service.analyze_text("", prompt)
    
    async def _generate_bias_testing_plan(
        self, 
        solution: Dict[str, Any], 
        project: Project
    ) -> str:
        """Generate bias testing plan grounded in ethical AI frameworks"""
        
        ai_technique = solution.get("ai_technique", "classification")
        target_beneficiaries = await self._extract_target_beneficiaries_dynamically(project)
        
        # GET BIAS TESTING FRAMEWORKS from RAG
        bias_testing_context = await rag_service.get_bias_testing_frameworks_context(
            ai_technique=ai_technique,
            target_beneficiaries=target_beneficiaries,
            project_description=project.description
        )
        
        prompt = f"""
        Generate comprehensive bias testing plan grounded in humanitarian AI frameworks:
        
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {ai_technique}
        PROJECT: {project.title}
        TARGET BENEFICIARIES: {target_beneficiaries}
        
        BIAS TESTING FRAMEWORKS FROM HUMANITARIAN AI KNOWLEDGE BASE:
        {bias_testing_context}
        
        Based on established frameworks, create comprehensive bias testing plan including:
        
        1. Framework-specific bias types to test for {ai_technique}
        2. Testing methodologies from established humanitarian AI guidelines
        3. Evaluation metrics recommended by frameworks
        4. Bias mitigation strategies from knowledge base
        5. Continuous monitoring aligned with humanitarian AI best practices
        6. Reporting procedures following framework requirements
        
        Reference specific frameworks and guidelines from the humanitarian AI knowledge base.
        Include specific metrics, tools, and methodologies proven in humanitarian AI applications.
        """
        
        return await llm_service.analyze_text("", prompt)

    async def _generate_monitoring_recommendations(
        self, 
        solution: Dict[str, Any], 
        project: Project
    ) -> str:
        """Generate monitoring recommendations grounded in AI governance frameworks"""
        
        ai_technique = solution.get("ai_technique", "classification")
        deployment_strategy = solution.get("deployment_strategy", "cloud_native")
        
        # GET MONITORING FRAMEWORKS from RAG
        monitoring_context = await rag_service.get_monitoring_frameworks_context(
            ai_technique=ai_technique,
            deployment_strategy=deployment_strategy,
            project_description=project.description
        )
        
        prompt = f"""
        Generate monitoring recommendations grounded in humanitarian AI governance frameworks:
        
        SOLUTION: {solution.get('title')}
        AI TECHNIQUE: {ai_technique}
        DEPLOYMENT: {deployment_strategy}
        
        HUMANITARIAN AI MONITORING FRAMEWORKS:
        {monitoring_context}
        
        Based on established AI governance frameworks and humanitarian AI guidelines, create monitoring 
        recommendations covering:
        
        1. Performance monitoring metrics from frameworks
        2. Model drift detection following best practices
        3. Ethical compliance monitoring per humanitarian AI guidelines
        4. System health monitoring for {deployment_strategy}
        5. User feedback collection aligned with frameworks
        6. Alerting and governance following established patterns
        7. Periodic review processes from humanitarian AI case studies
        
        Reference specific frameworks, tools, and metrics proven effective in humanitarian AI monitoring.
        Provide concrete implementation guidance based on knowledge base best practices.
        """
        
        return await llm_service.analyze_text("", prompt)