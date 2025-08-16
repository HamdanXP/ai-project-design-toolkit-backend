from anthropic import AsyncAnthropic
from config.settings import settings
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from models.development import (
    AISolution, LLMRequirements, NLPRequirements, RequiredFeature, TabularDataRequirements, TechnicalArchitecture, ResourceRequirement, EthicalSafeguard
)
from models.enums import AITechnique, DeploymentStrategy

logger = logging.getLogger(__name__)

class ClaudeCodeGenerationService:
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.claude_api_key)
        self.max_retries = 3
        
    def _format_validation_issue(self, issue) -> str:
        if isinstance(issue, dict):
            severity = issue.get('severity', 'medium').upper()
            description = issue.get('description', 'Unknown issue')
            return f"{severity}: {description}"
        elif isinstance(issue, str):
            return f"ISSUE: {issue}"
        else:
            return f"UNKNOWN: {str(issue)}"

    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        if '"""' in content:
            import re
            content = re.sub(r'"""([^"]*?)"""', lambda m: json.dumps(m.group(1)), content, flags=re.DOTALL)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_content = content[json_start:json_end + 1]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract valid JSON from response: {content[:200]}...")
    
    def _validate_enum_values(self, data: Dict[str, Any], field_mappings: Dict[str, List[str]]) -> bool:
        
        for field, valid_values in field_mappings.items():
            if field in data:
                value = str(data[field]).strip().lower()
                if value not in valid_values:
                    logger.error(f"Invalid {field} value: '{value}'. Valid values: {valid_values}")
                    return False
        
        return True

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

    async def analyze_viable_ai_approaches(self, project_description: str, use_case: Dict[str, Any], 
                                            deployment_constraints: Dict[str, Any], problem_domain: str,
                                            user_feedback: Optional[str] = None) -> Dict[str, Any]:
            
        infrastructure_context = ""
        constraint_guidance = ""
        if deployment_constraints:
            infrastructure_context = f"""
            Available Infrastructure:
            - Computing: {deployment_constraints.get('computing_resources', 'Not specified')}
            - Storage: {deployment_constraints.get('storage_data', 'Not specified')}
            - Connectivity: {deployment_constraints.get('internet_connectivity', 'Not specified')}
            - Environment: {deployment_constraints.get('deployment_environment', 'Not specified')}
            """
            
            constraint_guidance = """
            IMPORTANT CONSTRAINT COMPATIBILITY:
            - cloud_native/api_integration require stable internet (stable_broadband/satellite_internet) and cloud storage
            - shared_community/intermittent_connection internet cannot reliably support cloud_native but may support hybrid_approach or limited api_integration
            - basic_digital storage limits local model size and complexity, affecting deployment options
            - basic_hardware computing limits deployment to local_processing, edge_computing, or offline_first
            - paper_based storage prevents AI deployment entirely
            - no_internet/limited_connectivity requires local_processing, edge_computing, or offline_first only
            - mobile_optimized requires mobile_devices as primary computing resource
            - offline_first is designed for no_internet/limited_connectivity scenarios with local storage
            
            Only suggest deployments that are technically feasible given these constraints.
            """
            
        feedback_context = ""
        if user_feedback:
            feedback_context = f"""
            
            USER REQUIREMENTS:
            {user_feedback}
            
            Incorporate these requirements when analyzing viable AI approaches and ensure suggested techniques align with user needs.
            """
        
        valid_techniques = [
            "classification", "regression", "computer_vision", "nlp", "llm", 
            "time_series", "recommendation", "anomaly_detection", "clustering", 
            "optimization"
        ]
        
        valid_deployment = [
            "local_processing", "cloud_native", "api_integration", "hybrid_approach",
            "edge_computing", "offline_first", "mobile_optimized"
        ]
        
        user_prompt = f"""
        Analyze this humanitarian AI project and identify the most relevant AI approaches.

        PROJECT: {project_description}
        DOMAIN: {problem_domain}
        USE CASE: {use_case.get('title', '')} - {use_case.get('description', '')}
        {infrastructure_context}{constraint_guidance}{feedback_context}

        Respond with ONLY valid JSON. No explanatory text before or after.

        Analyze what AI techniques could address this humanitarian problem, specifically considering the selected use case.
        Tailor the AI techniques to the specific requirements and constraints of: {use_case.get('title', '')}.
        Consider how this particular use case affects data requirements, processing needs, and output formats.

        Valid AI techniques: {', '.join(valid_techniques)}
        Valid deployment strategies: {', '.join(valid_deployment)}

        {{
            "viable_techniques": [
                "technique_that_addresses_primary_problem",
                "technique_that_addresses_different_aspect"
            ],
            "technique_rationales": {{
                "technique1": "why this technique specifically addresses the problem",
                "technique2": "why this different technique addresses another aspect"
            }},
            "suitable_deployments": [
                "deployment_strategy_that_fits_constraints"
            ],
            "complexity_assessment": {{
                "data_complexity": "simple|moderate|advanced",
                "problem_complexity": "simple|moderate|advanced",
                "recommended_complexity": "simple|moderate|advanced"
            }},
            "key_requirements": [
                "specific functional requirement based on problem"
            ],
            "analysis_summary": "Brief explanation of why these techniques were identified"
        }}

        Only include techniques that genuinely address different aspects of the problem.
        Base decisions on the specific humanitarian context and actual problem requirements.
        Verify each suggested deployment is compatible with the infrastructure constraints listed above.
        
        """

        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=settings.claude_model,
                    max_tokens=settings.claude_max_tokens,
                    temperature=settings.claude_temperature,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                
                content = response.content[0].text
                result = self._extract_json_from_response(content)
                
                viable_techniques = result.get("viable_techniques", [])
                if not viable_techniques:
                    logger.warning(f"Attempt {attempt + 1}: No viable techniques identified")
                    continue
                
                if all(technique in valid_techniques for technique in viable_techniques):
                    return result
                else:
                    logger.warning(f"Attempt {attempt + 1}: Invalid techniques in viable approaches")
                    if attempt == self.max_retries - 1:
                        raise ValueError("Failed to generate valid viable approaches after multiple attempts")
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to analyze viable AI approaches: {e}")
        
    async def generate_contextual_solutions(self, viable_approaches: Dict[str, Any], 
                                    project_context: Dict[str, Any], user_feedback: Optional[str] = None) -> List[AISolution]:
        
        system_prompt = """Generate diverse AI solutions for humanitarian problems with accurate dataset requirements. Each solution should clearly specify if it needs data and what type. The targeted users are non-technical humanitarian professionals"""
        
        feedback_context = ""
        if user_feedback:
            feedback_context = f"""
            
            USER REQUIREMENTS:
            {user_feedback}
            
            Incorporate these requirements into solutions that address the problem effectively.
            """
        
        viable_techniques = viable_approaches.get("viable_techniques", [])
        suitable_deployments = viable_approaches.get("suitable_deployments", [])
        technique_rationales = viable_approaches.get("technique_rationales", {})

        user_prompt = f"""Generate 3 relevant AI solutions for this humanitarian problem.

    VIABLE TECHNIQUES: {viable_techniques}
    RATIONALES: {json.dumps(technique_rationales, indent=2)}
    PROJECT: {json.dumps(project_context, indent=2)}{feedback_context}

    For each solution, determine:
    1. Does this solution need input data? (vs being purely conversational/rule-based)
    2. If yes, what type of data? (tabular, text, image, audio, video)
    3. If tabular data, what specific features/columns are required?

    All ai_technique values must be exactly one of: {', '.join(viable_techniques)}.
    All deployment_strategy values must be exactly one of: {', '.join(suitable_deployments)}.

    {{
        "solutions": [
            {{
                "id": "solution_1",
                "title": "Specific Solution Name",
                "description": "How this approach specifically addresses the humanitarian problem",
                "ai_technique": "relevant technique from viable list",
                "deployment_strategy": "suitable deployment",
                "recommended": appropriate boolean value if the most appropriate solution among the available solutions,
                "confidence_score": appropriate integer value for confidence score of how well this solution addresses the problem,
                "needs_dataset": Does the solution's ai technique require a dataset? (appropriate boolean value),
                "dataset_type": "The required dataset type for the solution's ai technique (tabular|text|image|audio|video|none)",
                "tabular_requirements (only if dataset_type is tabular)": {{
                    "required_features": [
                        {{
                            "name": "feature_name",
                            "description": "What this feature represents",
                            "data_type": "numeric|categorical|datetime|text",
                            "humanitarian_purpose": "Why this is needed for humanitarian impact"
                        }}
                    ],
                    "optional_features": [
                        {{
                            "name": "optional_feature_name", 
                            "description": "Additional helpful feature",
                            "data_type": "numeric|categorical|datetime|text",
                            "humanitarian_purpose": "How this enhances the solution"
                        }}
                    ],
                    "minimum_rows": minimum_data_points_needed,
                    "data_types": {{"column_name": "expected_type"}}
                }},
                "llm_requirements (only if ai_technique is llm)": {{
                    "system_prompt": "Complete, production-ready system prompt specifically designed for this humanitarian use case and target beneficiaries",
                    "suggested_model": "Appropriate model based on solution complexity and deployment constraints",
                    "key_parameters": {{
                        "temperature": appropriate_value_for_this_use_case,
                        "max_tokens": appropriate_value_for_expected_output_length,
                        "additional_parameters": "as_needed_for_this_solution"
                    }}
                }},
                "nlp_requirements (only if ai_technique is nlp)": {{
                    "preprocessing_steps": ["specific preprocessing step for this humanitarian context"],
                    "processing_approach": "Specific NLP approach suitable for this problem and deployment context",
                    "feature_extraction": "Feature extraction method appropriate for this humanitarian use case",
                    "expected_input_format": "Description of expected input format for this solution"
                }},
                "capabilities": ["specific capability this approach provides"],
                "key_features": ["main feature of this approach"],
                "technical_architecture": {{
                    "implementation": "interface appropriate for this approach",
                    "ai_component": "specific AI implementation", 
                    "data_input": "input method appropriate for this approach",
                    "output_format": "output appropriate for humanitarian context",
                    "user_interface": "the most appropriate interface type for this specific solution, ai_technique, target users, and deployment environment",
                    "deployment_method": "deployment appropriate for constraints"
                }},
                "resource_requirements": {{
                    "computing_power": "requirement level for this approach",
                    "storage_needs": "storage needs for this approach",
                    "internet_dependency": "connectivity needs for this approach", 
                    "technical_expertise": "expertise level for users",
                    "setup_time": "estimated setup time"
                }},
                "best_for": "specific humanitarian scenario where this approach excels",
                "use_case_alignment": "how this specifically addresses the project problem",
                "implementation_notes": ["practical consideration for this approach"],
                "expected_setup_time": "expected setup time for humanitarian professionals to get this prototype running. It's expected to be very short, on the order of minutes to a few hours.",
                "maintenance_requirements": ["specific maintenance need"],
                "data_requirements": ["input this approach needs"],
                "output_examples": ["specific example of what this will produce"],
                "ethical_safeguards": [
                    {{
                        "category": "specific protection for this context",
                        "measures": ["protection measure"],
                        "icon": "appropriate icon",
                        "priority": "appropriate priority level from low|medium|high"
                    }}
                ]
            }}
        ]
    }}

    CRITICAL REQUIREMENTS:
    - Only include tabular_requirements if needs_dataset=true AND dataset_type="tabular"
    - Only include llm_requirements if ai_technique="llm" 
    - Only include nlp_requirements if ai_technique="nlp"
    - For other combinations, set unused requirement fields to null
    - LLM system prompts must be complete and specific to the humanitarian problem and target users
    - NLP processing approaches must be specific to the data type and humanitarian context
    - All requirements must be directly implementable in generated code"""

        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=settings.claude_model,
                    max_tokens=settings.claude_max_tokens,
                    temperature=settings.claude_temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                
                content = response.content[0].text
                solutions_data = self._extract_json_from_response(content)
                solutions_list = solutions_data.get("solutions", [])
                
                if not solutions_list:
                    logger.warning(f"Attempt {attempt + 1}: No solutions returned")
                    continue
                
                valid_solutions = []
                used_techniques = set()
                
                for solution_dict in solutions_list:
                    if isinstance(solution_dict, dict):
                        technique = solution_dict.get("ai_technique", "").lower()
                        
                        if technique in used_techniques:
                            logger.warning(f"Skipping duplicate technique: {technique}")
                            continue
                            
                        field_mappings = {
                            "ai_technique": viable_techniques,
                            "deployment_strategy": suitable_deployments
                        }
                        
                        if self._validate_enum_values(solution_dict, field_mappings):
                            try:                                
                                solution_obj = self._convert_solution_dict_to_object(solution_dict)
                                valid_solutions.append(solution_obj)
                                used_techniques.add(technique)
                            except Exception as e:
                                logger.warning(f"Failed to convert solution to object: {e}")
                                continue
                        else:
                            logger.warning(f"Skipping solution with invalid enum values: {solution_dict.get('title', 'Unknown')}")
                
                if valid_solutions:
                    logger.info(f"Generated {len(valid_solutions)} valid solutions using different techniques")
                    return valid_solutions
                else:
                    logger.warning(f"Attempt {attempt + 1}: No valid solutions generated")
                    if attempt == self.max_retries - 1:
                        raise ValueError("Failed to generate valid solutions after multiple attempts")
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to generate contextual solutions: {e}")

    def _convert_solution_dict_to_object(self, solution_data: Dict[str, Any]) -> AISolution:
        try:
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
            
            tech_arch_data = solution_data.get("technical_architecture", {})
            if not tech_arch_data:
                raise ValueError("Missing technical architecture in solution data")
            
            tech_arch = TechnicalArchitecture(
                ai_technique=ai_technique,
                deployment_strategy=deployment_strategy,
                implementation=tech_arch_data.get("implementation", ""),
                ai_component=tech_arch_data.get("ai_component", ""),
                data_input=tech_arch_data.get("data_input", ""),
                output_format=tech_arch_data.get("output_format", ""),
                user_interface=tech_arch_data.get("user_interface", ""),
                deployment_method=tech_arch_data.get("deployment_method", "")
            )
            
            req_data = solution_data.get("resource_requirements", {})
            if not req_data:
                raise ValueError("Missing resource requirements in solution data")
            
            resource_req = ResourceRequirement(
                computing_power=req_data.get("computing_power", ""),
                storage_needs=req_data.get("storage_needs", ""),
                internet_dependency=req_data.get("internet_dependency", ""),
                technical_expertise=req_data.get("technical_expertise", ""),
                setup_time=req_data.get("setup_time", "")
            )
            
            tabular_requirements = None
            if (solution_data.get("needs_dataset") and 
                solution_data.get("dataset_type") == "tabular" and
                solution_data.get("tabular_requirements")):
                
                tabular_req_data = solution_data["tabular_requirements"]
                
                required_features = []
                for feature_data in tabular_req_data.get("required_features", []):
                    required_features.append(RequiredFeature(
                        name=feature_data.get("name", ""),
                        description=feature_data.get("description", ""),
                        data_type=feature_data.get("data_type", ""),
                        humanitarian_purpose=feature_data.get("humanitarian_purpose", "")
                    ))
                
                optional_features = []
                for feature_data in tabular_req_data.get("optional_features", []):
                    optional_features.append(RequiredFeature(
                        name=feature_data.get("name", ""),
                        description=feature_data.get("description", ""),
                        data_type=feature_data.get("data_type", ""),
                        humanitarian_purpose=feature_data.get("humanitarian_purpose", "")
                    ))
                    
                tabular_requirements = TabularDataRequirements(
                    required_features=required_features,
                    optional_features=optional_features,
                    minimum_rows=tabular_req_data.get("minimum_rows", 100),
                    data_types=tabular_req_data.get("data_types", {})
                )
            
            llm_requirements = None
            if solution_data.get("llm_requirements"):
                llm_req_data = solution_data["llm_requirements"]
                if llm_req_data.get("system_prompt"):
                    llm_requirements = LLMRequirements(**llm_req_data)
            
            nlp_requirements = None
            if solution_data.get("nlp_requirements"):
                nlp_req_data = solution_data["nlp_requirements"]
                if nlp_req_data.get("processing_approach"):
                    nlp_requirements = NLPRequirements(**nlp_req_data)

            ethical_safeguards = []
            safeguard_data = solution_data.get("ethical_safeguards", [])
            for sg_data in safeguard_data:
                if isinstance(sg_data, dict):
                    priority = sg_data.get("priority", "medium").strip().lower()
                    if priority not in ["low", "medium", "high", "critical"]:
                        priority = "medium"
                    
                    ethical_safeguards.append(EthicalSafeguard(
                        category=sg_data.get("category", "Protection"),
                        measures=sg_data.get("measures", []),
                        icon=sg_data.get("icon", "shield"),
                        priority=priority
                    ))
            
            return AISolution(
                id=solution_data.get("id", ""),
                title=solution_data.get("title", ""),
                description=solution_data.get("description", ""),
                ai_technique=ai_technique,
                deployment_strategy=deployment_strategy,
                recommended=solution_data.get("recommended", False),
                confidence_score=min(max(solution_data.get("confidence_score", 70), 0), 100),
                needs_dataset=solution_data.get("needs_dataset", False),
                dataset_type=solution_data.get("dataset_type"),
                tabular_requirements=tabular_requirements,
                llm_requirements=llm_requirements,
                nlp_requirements=nlp_requirements,
                capabilities=solution_data.get("capabilities", []),
                key_features=solution_data.get("key_features", []),
                technical_architecture=tech_arch,
                resource_requirements=resource_req,
                best_for=solution_data.get("best_for", ""),
                use_case_alignment=solution_data.get("use_case_alignment", ""),
                implementation_notes=solution_data.get("implementation_notes", []),
                ethical_safeguards=ethical_safeguards,
                estimated_setup_time=solution_data.get("estimated_setup_time", ""),
                maintenance_requirements=solution_data.get("maintenance_requirements", []),
                data_requirements=solution_data.get("data_requirements", []),
                output_examples=solution_data.get("output_examples", [])
            )
            
        except Exception as e:
            logger.error(f"Failed to convert solution data to AISolution object: {e}")
            raise

    async def design_contextual_architecture(self, selected_solution: Dict[str, Any], 
                                        project_context: Dict[str, Any], 
                                        user_feedback: Optional[str] = None) -> Dict[str, Any]:
        
        system_prompt = """Design implementation architecture based on the already-determined AI technique, project context, and technical decisions. Focus on file structure, dependencies, and code organization rather than re-deciding architectural choices."""
        
        existing_architecture = selected_solution.get('technical_architecture', {})
        ai_technique = selected_solution.get('ai_technique', 'classification')
        deployment_strategy = selected_solution.get('deployment_strategy', 'local_processing')
        llm_requirements = selected_solution.get('llm_requirements')
        nlp_requirements = selected_solution.get('nlp_requirements')
    
        requirements_context = ""
        if llm_requirements:
            requirements_context = f"""
            LLM IMPLEMENTATION REQUIREMENTS:
            System Prompt: {llm_requirements['system_prompt']}
            Model: {llm_requirements['suggested_model']}
            Parameters: {json.dumps(llm_requirements['key_parameters'])}
            """
        elif nlp_requirements:
            requirements_context = f"""
            NLP IMPLEMENTATION REQUIREMENTS:
            Processing: {nlp_requirements['processing_approach']}
            Preprocessing: {', '.join(nlp_requirements['preprocessing_steps'])}
            Feature Extraction: {nlp_requirements['feature_extraction']}
            Input Format: {nlp_requirements['expected_input_format']}
            """
        feedback_context = ""
        if user_feedback:
            feedback_context = f"""
            
            ADDITIONAL USER REQUIREMENTS:
            {user_feedback}
            
            Ensure the implementation architecture addresses these specific requirements while maintaining the existing architectural decisions.
            """
        
        user_prompt = f"""Design implementation architecture for this specific AI solution using the existing architectural decisions.

    SOLUTION: {json.dumps(selected_solution, indent=2)}
    PROJECT: {json.dumps(project_context, indent=2)}{feedback_context}
    REQUIREMENTS: {requirements_context}

    EXISTING ARCHITECTURAL DECISIONS (DO NOT CHANGE):
    - AI Technique: {ai_technique}
    - Deployment Strategy: {deployment_strategy}
    - User Interface: {existing_architecture.get('user_interface', '')}
    - AI Component: {existing_architecture.get('ai_component', '')}
    - Data Input Method: {existing_architecture.get('data_input', '')}
    - Output Format: {existing_architecture.get('output_format', '')}
    - Deployment Method: {existing_architecture.get('deployment_method', '')}

    Respond with ONLY valid JSON. No explanatory text.

    Focus on IMPLEMENTATION DETAILS that support these existing decisions:
    - What files, folders, and directory structure are needed for this exact solution
    - What dependencies and setup are actually required for these specific architectural choices  
    - What data processing steps implement the {ai_technique} technique with the chosen {existing_architecture.get('data_input', 'data input method')}
    - What output generation process creates the chosen {existing_architecture.get('output_format', 'output format')}

    IMPORTANT: 
        - Do not use generate or use R programming language - avoid .r, .R, .rmd, .Rmd files
        
    {{
        "implementation_plan": {{
            "main_file": "appropriate_main_file_for_{ai_technique}",
            "file_structure": [
                {{
                    "filename": "appropriate_filename",
                    "purpose": "specific purpose for implementing {existing_architecture.get('ai_component', '')}",
                    "type": "file_type",
                    "content_requirements": "what this file should contain to implement the chosen {ai_technique} approach"
                }}
            ],
            "folder_structure": [
                {{
                    "folder_name": "folder_if_needed",
                    "purpose": "what this folder contains for the {deployment_strategy} deployment",
                    "contents": ["files_or_subfolders_in_this_folder"]
                }}
            ],
            "dependencies": [
                "packages_needed_for_{ai_technique}_with_{deployment_strategy}"
            ],
            "setup_requirements": [
                "installation steps for implementing the chosen architecture"
            ]
        }},
        "technical_implementation": {{
            "data_processing": "how to process data from {existing_architecture.get('data_input', '')} for {ai_technique}",
            "output_generation": "how to generate {existing_architecture.get('output_format', '')} from {ai_technique} results"
        }},
        "technical_implementation": {{
            "data_processing": "how to process data from {existing_architecture.get('data_input', '')} for {ai_technique}",
            "output_generation": "how to generate {existing_architecture.get('output_format', '')} from {ai_technique} results"
        }},
        "code_structure": {{
            "main_functions": [
                "functions_needed_for_{ai_technique}_implementation"
            ],
            "user_inputs": ["inputs for the chosen {existing_architecture.get('data_input', '')}"],
            "processing_steps": ["steps for {ai_technique} with {existing_architecture.get('ai_component', '')}"],
            "outputs": ["outputs in {existing_architecture.get('output_format', '')}"]
        }},
        "integration_requirements": {{
            "external_apis": "apis_if_needed_for_{deployment_strategy}",
            "local_models": "models needed for {ai_technique}",
            "data_formats": "formats for {existing_architecture.get('data_input', '')}",
            "configuration_needs": "config for {existing_architecture.get('deployment_method', '')}"
        }}
    }}"""

        try:
            response = await self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=settings.claude_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            content = response.content[0].text
            result = self._extract_json_from_response(content)
            
            result["existing_architecture"] = existing_architecture
            
            if "issues_found" in result and isinstance(result["issues_found"], list):
                structured_issues = []
                for issue in result["issues_found"]:
                    if isinstance(issue, dict):
                        structured_issue = {
                            "severity": issue.get("severity", "medium"),
                            "description": issue.get("description", "Unknown issue"),
                            "category": issue.get("category", "general")
                        }
                        structured_issues.append(structured_issue)
                    elif isinstance(issue, str):
                        structured_issues.append({
                            "severity": "medium",
                            "description": issue,
                            "category": "general"
                        })
                
                result["issues_found"] = structured_issues
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to design contextual architecture: {e}")
            raise

    async def generate_contextual_code(self, architecture: Dict[str, Any], solution: Dict[str, Any], 
                                project_context: Dict[str, Any],
                                user_feedback: Optional[str] = None) -> Dict[str, str]:
        
        system_prompt = f"""Generate working Python code for this humanitarian AI solution. Create functional code that implements the specific AI technique with appropriate interfaces for humanitarian users."""
        
        llm_requirements = solution.get('llm_requirements')
        nlp_requirements = solution.get('nlp_requirements')
    
        requirements_context = ""
        if llm_requirements:
            requirements_context = f"""
            LLM IMPLEMENTATION REQUIREMENTS:
            System Prompt: {llm_requirements['system_prompt']}
            Model: {llm_requirements['suggested_model']}
            Parameters: {json.dumps(llm_requirements['key_parameters'])}
            """
        elif nlp_requirements:
            requirements_context = f"""
            NLP IMPLEMENTATION REQUIREMENTS:
            Processing: {nlp_requirements['processing_approach']}
            Preprocessing: {', '.join(nlp_requirements['preprocessing_steps'])}
            Feature Extraction: {nlp_requirements['feature_extraction']}
            Input Format: {nlp_requirements['expected_input_format']}
            """
        feedback_context = ""
        if user_feedback:
            feedback_context = f"""
            
            ADDITIONAL USER REQUIREMENTS:
            {user_feedback}
            
            Ensure the generated code addresses these specific requirements while implementing the {solution.get('ai_technique')} solution.
            """
                
        user_prompt = f"""Generate complete working code for this AI solution.

ARCHITECTURE: {json.dumps(architecture, indent=2)}
SOLUTION: {json.dumps(solution, indent=2)}
PROJECT: {json.dumps(project_context, indent=2)}{feedback_context}
{requirements_context}

Respond with ONLY valid JSON mapping filenames to complete file contents. No explanatory text.

Requirements based on this specific solution:
1. Implement the specific AI technique for this humanitarian context
2. Create appropriate interface for the users and use case described
3. Handle data input appropriate for this specific solution
4. Include clear guidance for users through code comments and interface
5. Generate error handling appropriate for this technique and context
6. Create any file types needed (code, config, documentation, data files, etc.)

If this solution works with datasets:
- Generate dynamic code that adapts to any dataset structure
- Include user-friendly ways to explore and select columns
- Provide clear guidance on choosing target variables and features
- Include data visualization and exploration capabilities
- Add comments explaining how to work with different data types

If this solution doesn't work with datasets:
- Design appropriate input method for the specific use case
- Focus on the right interaction pattern for this humanitarian problem
- Create interface suitable for the described users and context

Return complete, working files as valid JSON with proper escaping:
{{
    "filename.extension": "complete_file_content_with_proper_escaping"
}}"""

        try:
            response = await self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=settings.claude_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            content = response.content[0].text.strip()
            return self._extract_json_from_response(content)
            
        except Exception as e:
            logger.error(f"Failed to generate contextual code: {e}")
            raise

    async def validate_generated_code(self, code_files: Dict[str, str], 
                                    solution_requirements: Dict[str, Any],
                                    user_feedback: Optional[str] = None) -> Dict[str, Any]:
        
        system_prompt = """Validate that generated files implement the AI solution correctly and are appropriate for humanitarian users. Check all files for functionality, completeness, and usability."""
        
        user_requirements_context = ""
        if user_feedback:
            user_requirements_context = f"""
            
            USER REQUIREMENTS TO VERIFY:
            {user_feedback}
            
            Validate that the generated code addresses these specific user requirements.
            """
        
        file_summary = {}
        for filename, content in code_files.items():
            file_extension = filename.split('.')[-1] if '.' in filename else 'unknown'
            file_summary[filename] = {
                "file_type": file_extension,
                "content_length": len(content),
                "has_content": bool(content.strip())
            }

        user_prompt = f"""Validate this humanitarian AI solution for {solution_requirements.get('ai_technique')}.

SOLUTION REQUIREMENTS: {json.dumps(solution_requirements, indent=2)}
FILES GENERATED: {json.dumps(file_summary, indent=2)}{user_requirements_context}

Analyze all generated files and check:
1. Do the files implement {solution_requirements.get('ai_technique')} technique appropriately?
2. Are the files suitable for humanitarian users described in the context?
3. Do the files work together to create a functional solution?
4. Is error handling appropriate for this technique and use case?
5. Are dependencies and setup requirements complete?
6. Can users actually run and use this solution?

Respond with ONLY valid JSON.

{{
    "validation_passed": true,
    "code_completeness_score": 0.9,
    "issues_found": [],
    "working_code_assessment": {{
        "has_main_functionality": true,
        "implements_ai_technique": true,
        "handles_user_data": true,
        "produces_outputs": true,
        "error_handling_present": true,
        "dependencies_complete": true,
        "runnable_immediately": true
    }},
    "recommendations": []
}}"""

        try:
            response = await self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=settings.claude_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            content = response.content[0].text
            return self._extract_json_from_response(content)
            
        except Exception as e:
            logger.error(f"Failed to validate generated code: {e}")
            return {
                "validation_passed": False,
                "code_completeness_score": 0.0,
                "issues_found": [
                    {
                        "severity": "critical", 
                        "description": f"Validation failed: {str(e)}",
                        "category": "validation_error"
                    }
                ],
                "working_code_assessment": {
                    "has_main_functionality": False,
                    "implements_ai_technique": False,
                    "handles_user_data": False,
                    "produces_outputs": False,
                    "error_handling_present": False,
                    "dependencies_complete": False,
                    "runnable_immediately": False
                },
                "recommendations": ["Manual code review required"]
            }

    async def fix_code_validation_issues(self, current_code: Dict[str, str], 
                                        validation_issues: List[Any],
                                        architecture: Dict[str, Any],
                                        solution: Dict[str, Any],
                                        user_feedback: Optional[str] = None) -> Dict[str, str]:
        
        if not validation_issues:
            return current_code
        
        system_prompt = f"""Fix specific issues in this humanitarian AI tool code. Focus on making the AI technique implementation work correctly for the intended users and context."""
        
        user_context = ""
        if user_feedback:
            user_context = f"""
            
            USER REQUIREMENTS TO PRESERVE:
            {user_feedback}
            
            While fixing issues, ensure these user requirements remain implemented.
            """
        
        issues_summary = "\n".join([
            f"- {self._format_validation_issue(issue)}"
            for issue in validation_issues
        ])

        user_prompt = f"""Fix these issues in the {solution.get('ai_technique')} implementation:

ISSUES TO FIX:
{issues_summary}

CURRENT CODE:
{json.dumps(current_code, indent=2)}{user_context}

AI TECHNIQUE: {solution.get('ai_technique')}
ARCHITECTURE: {json.dumps(architecture.get('technical_implementation', {}), indent=2)}

Fix issues while maintaining the {solution.get('ai_technique')} implementation for this humanitarian context.

Respond with ONLY valid JSON containing the corrected code.

{json.dumps({filename: f"corrected implementation" for filename in current_code.keys()}, indent=2)}"""

        try:
            response = await self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=settings.claude_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            content = response.content[0].text.strip()
            fixed_code = self._extract_json_from_response(content)
            
            for original_file in current_code.keys():
                if original_file not in fixed_code:
                    logger.warning(f"Fixed code missing file: {original_file}, keeping original")
                    fixed_code[original_file] = current_code[original_file]
            
            return fixed_code
            
        except Exception as e:
            logger.error(f"Failed to fix code validation issues: {e}")
            return current_code

claude_code_service = ClaudeCodeGenerationService()