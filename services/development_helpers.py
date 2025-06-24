"""
Helper methods for the Development Service
This file contains all the supporting methods needed for the enhanced development service
"""

import logging
from typing import List, Dict, Any, Optional
from models.development import (
    AITechnique, DeploymentStrategy, ComplexityLevel, ResourceRequirement,
    EthicalSafeguard, TechnicalArchitecture, AISolution
)
from models.project import Project

logger = logging.getLogger(__name__)

class DevelopmentServiceHelpers:
    """Helper methods for the Development Service"""
    
    @staticmethod
    def calculate_technique_confidence(use_case: Dict[str, Any], technique: str) -> int:
        """Calculate confidence score for AI technique selection"""
        base_confidence = 70
        
        # Boost confidence based on keyword matches
        title = use_case.get('title', '').lower()
        description = use_case.get('description', '').lower()
        text_content = f"{title} {description}"
        
        technique_keywords = {
            "classification": ["classify", "categorize", "detect", "identify", "diagnosis"],
            "computer_vision": ["image", "photo", "visual", "picture", "scan", "camera"],
            "nlp": ["text", "language", "document", "report", "translation"],
            "llm": ["chatbot", "assistant", "generate", "summarize", "conversation"],
            "time_series": ["trend", "forecast", "predict", "time series", "monitoring"],
            "recommendation": ["recommend", "suggest", "optimize", "allocate"]
        }
        
        keywords = technique_keywords.get(technique, [])
        matches = sum(1 for keyword in keywords if keyword in text_content)
        
        # Increase confidence based on matches
        confidence_boost = min(matches * 5, 25)
        return min(base_confidence + confidence_boost, 95)
    
    @staticmethod
    def get_implementation_notes(technique: str) -> List[str]:
        """Get implementation notes for specific AI techniques"""
        notes_map = {
            "classification": [
                "Consider class imbalance in humanitarian data",
                "Implement confidence thresholds for safety",
                "Plan for multi-language support"
            ],
            "computer_vision": [
                "Ensure diverse image training data",
                "Plan for varying image quality in field conditions",
                "Consider privacy implications of image data"
            ],
            "nlp": [
                "Support multiple languages and dialects",
                "Handle domain-specific terminology",
                "Consider cultural context in text processing"
            ],
            "llm": [
                "Implement content filtering and safety measures",
                "Plan for API cost management",
                "Consider offline fallback options"
            ]
        }
        return notes_map.get(technique, ["Standard AI implementation considerations"])
    
    @staticmethod
    async def extract_target_beneficiaries(project: Project) -> str:
        """Extract target beneficiaries from project data"""
        if project.reflection_data and project.reflection_data.get("answers"):
            answers = project.reflection_data["answers"]
            for key, answer in answers.items():
                if any(term in key.lower() for term in ["beneficiar", "target", "user", "community"]):
                    if answer and len(answer.strip()) > 10:
                        return answer.strip()[:200]
        
        # Fallback based on domain
        domain_beneficiaries = {
            "health": "healthcare workers, patients, and community health volunteers",
            "education": "teachers, students, and education coordinators",
            "agriculture": "farmers, agricultural extension workers, and rural communities",
            "disaster_response": "emergency responders, affected populations, and coordination teams",
            "protection": "protection officers, case workers, and vulnerable populations"
        }
        
        domain = getattr(project, 'problem_domain', 'general_humanitarian')
        return domain_beneficiaries.get(domain, "humanitarian workers and affected communities")
    
    @staticmethod
    def generate_technique_capabilities(technique: AITechnique, use_case_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific capabilities based on AI technique"""
        base_capabilities = {
            AITechnique.CLASSIFICATION: [
                "Automated categorization with confidence scores",
                "Multi-class and multi-label classification support",
                "Real-time prediction capabilities",
                "Batch processing for large datasets",
                "Explainable predictions with reasoning"
            ],
            AITechnique.COMPUTER_VISION: [
                "Image classification and object detection",
                "Visual quality assessment and anomaly detection",
                "Multi-format image processing (JPEG, PNG, TIFF)",
                "Batch image analysis capabilities",
                "Visual similarity search and matching"
            ],
            AITechnique.LARGE_LANGUAGE_MODEL: [
                "Natural language understanding and generation",
                "Document summarization and analysis",
                "Multi-language conversation support",
                "Content generation and editing assistance",
                "Question-answering capabilities"
            ],
            AITechnique.TIME_SERIES_ANALYSIS: [
                "Trend analysis and pattern recognition",
                "Forecasting with confidence intervals",
                "Anomaly detection and alerting",
                "Seasonal pattern analysis",
                "Real-time monitoring and updates"
            ],
            AITechnique.RECOMMENDATION_SYSTEM: [
                "Personalized recommendations",
                "Resource allocation optimization",
                "Similarity-based matching",
                "Collaborative and content-based filtering",
                "Recommendation explanation and reasoning"
            ]
        }
        
        capabilities = base_capabilities.get(technique, base_capabilities[AITechnique.CLASSIFICATION])
        
        # Add use case specific capabilities
        if use_case_analysis.get("data_types"):
            data_types = use_case_analysis["data_types"]
            if "text" in data_types:
                capabilities.append("Multi-language text processing support")
            if "images" in data_types:
                capabilities.append("Advanced image preprocessing and enhancement")
        
        return capabilities
    
    @staticmethod
    def generate_technique_features(technique: AITechnique) -> List[str]:
        """Generate key features for specific AI techniques"""
        features_map = {
            AITechnique.CLASSIFICATION: [
                "Pre-trained model fine-tuning",
                "Custom feature engineering",
                "Model interpretability tools",
                "Performance monitoring dashboard"
            ],
            AITechnique.COMPUTER_VISION: [
                "Advanced CNN architectures",
                "Image augmentation pipeline",
                "Visual attention mechanisms",
                "Edge-optimized model variants"
            ],
            AITechnique.LARGE_LANGUAGE_MODEL: [
                "State-of-the-art transformer models",
                "Custom prompt engineering",
                "Content safety filtering",
                "API cost optimization"
            ],
            AITechnique.TIME_SERIES_ANALYSIS: [
                "Multiple forecasting algorithms",
                "Automatic seasonality detection",
                "Real-time data ingestion",
                "Interactive visualization tools"
            ]
        }
        
        return features_map.get(technique, [
            "Machine learning pipeline",
            "Data preprocessing tools",
            "Model monitoring",
            "User-friendly interface"
        ])
    
    @staticmethod
    def create_technical_architecture(technique: AITechnique, deployment_strategy: DeploymentStrategy) -> TechnicalArchitecture:
        """Create technical architecture based on technique and deployment strategy"""
        
        # Base architectures by deployment strategy
        deployment_configs = {
            DeploymentStrategy.API_INTEGRATION: {
                "frontend": "Progressive Web App with API integration",
                "backend": "API gateway with cloud AI service integration",
                "deployment": "Serverless functions with API management",
                "monitoring": "API usage monitoring and cost tracking"
            },
            DeploymentStrategy.CLOUD_NATIVE: {
                "frontend": "Scalable web application",
                "backend": "Microservices architecture with container orchestration",
                "deployment": "Kubernetes cluster with auto-scaling",
                "monitoring": "Comprehensive application and infrastructure monitoring"
            },
            DeploymentStrategy.EDGE_COMPUTING: {
                "frontend": "Offline-capable progressive web app",
                "backend": "Edge computing runtime with local processing",
                "deployment": "Edge devices with local model serving",
                "monitoring": "Local monitoring with periodic cloud sync"
            },
            DeploymentStrategy.HYBRID_APPROACH: {
                "frontend": "Multi-platform application (web + mobile)",
                "backend": "Hybrid cloud-edge architecture",
                "deployment": "Flexible deployment across cloud and edge",
                "monitoring": "Unified monitoring across deployment targets"
            }
        }
        
        # AI components by technique
        ai_components_map = {
            AITechnique.CLASSIFICATION: [
                "Scikit-learn/XGBoost models",
                "Feature engineering pipeline",
                "Model serving API",
                "Prediction confidence scoring"
            ],
            AITechnique.COMPUTER_VISION: [
                "TensorFlow/PyTorch CNN models",
                "OpenCV image processing",
                "Model optimization (TensorRT/ONNX)",
                "Batch inference pipeline"
            ],
            AITechnique.LARGE_LANGUAGE_MODEL: [
                "Transformer model integration",
                "Text preprocessing pipeline",
                "Prompt engineering framework",
                "Response generation and filtering"
            ],
            AITechnique.TIME_SERIES_ANALYSIS: [
                "Time series forecasting models",
                "Data streaming pipeline",
                "Anomaly detection algorithms",
                "Visualization and reporting tools"
            ]
        }
        
        config = deployment_configs.get(deployment_strategy, deployment_configs[DeploymentStrategy.CLOUD_NATIVE])
        ai_components = ai_components_map.get(technique, ai_components_map[AITechnique.CLASSIFICATION])
        
        return TechnicalArchitecture(
            ai_technique=technique,
            deployment_strategy=deployment_strategy,
            frontend=config["frontend"],
            backend=config["backend"],
            ai_components=ai_components,
            data_processing="Secure data pipeline with preprocessing and validation",
            deployment=config["deployment"],
            monitoring=config["monitoring"]
        )
    
    @staticmethod
    def calculate_resource_requirements(complexity: ComplexityLevel, deployment_strategy: DeploymentStrategy) -> ResourceRequirement:
        """Calculate resource requirements based on complexity and deployment"""
        
        # Base requirements by complexity
        complexity_requirements = {
            ComplexityLevel.SIMPLE: {
                "computing_power": "low",
                "storage_needs": "minimal",
                "technical_expertise": "basic"
            },
            ComplexityLevel.MODERATE: {
                "computing_power": "medium",
                "storage_needs": "moderate",
                "technical_expertise": "intermediate"
            },
            ComplexityLevel.ADVANCED: {
                "computing_power": "high",
                "storage_needs": "extensive",
                "technical_expertise": "advanced"
            },
            ComplexityLevel.ENTERPRISE: {
                "computing_power": "high",
                "storage_needs": "extensive",
                "technical_expertise": "advanced"
            }
        }
        
        # Adjust based on deployment strategy
        deployment_adjustments = {
            DeploymentStrategy.API_INTEGRATION: {
                "computing_power": "low",  # Reduced because processing is in cloud
                "internet_dependency": "continuous",
                "budget_estimate": "medium"  # API costs
            },
            DeploymentStrategy.EDGE_COMPUTING: {
                "computing_power": "medium",  # Need local processing power
                "internet_dependency": "periodic",
                "budget_estimate": "high"  # Hardware costs
            },
            DeploymentStrategy.CLOUD_NATIVE: {
                "internet_dependency": "continuous",
                "budget_estimate": "medium"
            }
        }
        
        base_req = complexity_requirements.get(complexity, complexity_requirements[ComplexityLevel.MODERATE])
        adjustments = deployment_adjustments.get(deployment_strategy, {})
        
        return ResourceRequirement(
            computing_power=adjustments.get("computing_power", base_req["computing_power"]),
            storage_needs=base_req["storage_needs"],
            internet_dependency=adjustments.get("internet_dependency", "continuous"),
            technical_expertise=base_req["technical_expertise"],
            budget_estimate=adjustments.get("budget_estimate", "medium")
        )
    
    @staticmethod
    def get_deployment_considerations(deployment_strategy: DeploymentStrategy) -> List[str]:
        """Get deployment considerations for different strategies"""
        considerations_map = {
            DeploymentStrategy.API_INTEGRATION: [
                "Requires stable internet connectivity",
                "API usage costs scale with volume",
                "Data privacy considerations for cloud processing",
                "Vendor dependency and lock-in risks"
            ],
            DeploymentStrategy.CLOUD_NATIVE: [
                "Requires cloud infrastructure setup",
                "Need for DevOps and monitoring capabilities",
                "Data residency and compliance considerations",
                "Ongoing cloud operational costs"
            ],
            DeploymentStrategy.EDGE_COMPUTING: [
                "Requires edge hardware procurement and setup",
                "Local technical support and maintenance needed",
                "Model updates and synchronization complexity",
                "Higher upfront hardware costs"
            ],
            DeploymentStrategy.HYBRID_APPROACH: [
                "Complex architecture requiring advanced technical skills",
                "Multiple deployment targets increase maintenance overhead",
                "Data synchronization and consistency challenges",
                "Higher development and testing complexity"
            ]
        }
        
        return considerations_map.get(deployment_strategy, [
            "Standard deployment considerations apply",
            "Plan for monitoring and maintenance",
            "Consider security and compliance requirements"
        ])
    
    @staticmethod
    def estimate_implementation_timeline(complexity: ComplexityLevel) -> str:
        """Estimate implementation timeline based on complexity"""
        timeline_map = {
            ComplexityLevel.SIMPLE: "2-4 weeks",
            ComplexityLevel.MODERATE: "4-8 weeks",
            ComplexityLevel.ADVANCED: "8-12 weeks",
            ComplexityLevel.ENTERPRISE: "12-20 weeks"
        }
        return timeline_map.get(complexity, "6-10 weeks")
    
    @staticmethod
    def get_maintenance_requirements(complexity: ComplexityLevel) -> List[str]:
        """Get maintenance requirements based on complexity"""
        requirements_map = {
            ComplexityLevel.SIMPLE: [
                "Basic monitoring and health checks",
                "Periodic model retraining",
                "User support and documentation updates"
            ],
            ComplexityLevel.MODERATE: [
                "Model performance monitoring",
                "Data pipeline maintenance",
                "Security updates and patches",
                "User training and support"
            ],
            ComplexityLevel.ADVANCED: [
                "Advanced monitoring and alerting",
                "Model drift detection and retraining",
                "Performance optimization",
                "Security audits and compliance",
                "Advanced user training"
            ],
            ComplexityLevel.ENTERPRISE: [
                "Enterprise monitoring and SLA management",
                "MLOps pipeline maintenance",
                "Security and compliance auditing",
                "Advanced analytics and reporting",
                "Enterprise support and training"
            ]
        }
        
        return requirements_map.get(complexity, [
            "Regular monitoring and maintenance",
            "Model updates and improvements",
            "User support and training"
        ])
    
    @staticmethod
    def mark_recommended_solutions(
        solutions: List[AISolution], 
        use_case_analysis: Dict[str, Any], 
        deployment_analysis: Dict[str, Any]
    ):
        """Mark the most suitable solutions as recommended"""
        if not solutions:
            return
        
        # Reset all recommendations
        for solution in solutions:
            solution.recommended = False
        
        # Primary recommendation: technique-specific solution
        primary_technique = use_case_analysis.get("primary_ai_technique")
        for solution in solutions:
            if solution.ai_technique.value == primary_technique:
                solution.recommended = True
                break
        
        # Secondary recommendation: best deployment strategy match
        recommended_strategies = deployment_analysis.get("recommended_strategies", [])
        if recommended_strategies:
            for solution in solutions:
                if solution.deployment_strategy.value in recommended_strategies:
                    solution.recommended = True
                    break
        
        # Ensure at least one solution is recommended
        if not any(solution.recommended for solution in solutions):
            solutions[0].recommended = True