from enum import Enum

class AITechnique(str, Enum):
    """Comprehensive AI technique categories"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    COMPUTER_VISION = "computer_vision"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    LARGE_LANGUAGE_MODEL = "llm"
    TIME_SERIES_ANALYSIS = "time_series"
    RECOMMENDATION_SYSTEM = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    OPTIMIZATION = "optimization"
    MULTI_MODAL = "multi_modal"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class DeploymentStrategy(str, Enum):
    """Deployment strategy options"""
    LOCAL_PROCESSING = "local_processing"
    CLOUD_NATIVE = "cloud_native"
    API_INTEGRATION = "api_integration"
    HYBRID_APPROACH = "hybrid_approach"
    EDGE_COMPUTING = "edge_computing"
    FEDERATED_LEARNING = "federated_learning"
    SERVERLESS = "serverless"

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"
