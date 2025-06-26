from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Application
    app_name: str = "AI Project Design Toolkit for Humanitarians"
    app_version: str = "1.0.0"
    debug: bool = False
   
    # API
    api_v1_prefix: str = "/api/v1"
   
    # MongoDB
    mongodb_url: str
    mongodb_database: str = "ADPT_DB"
   
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
   
    # Google Cloud Storage - Simplified folder-based organization
    google_application_credentials: str
    google_creds_b64: Optional[str] = None
    gcp_bucket_name: str  # Main bucket with domain folders
    gcp_use_cases_bucket_name: str  # Separate bucket for use cases
    gcp_indexes_bucket_name: str  # Separate bucket for storing vector indexes
    
    # Domain-specific folder paths within main bucket
    ai_ethics_folder_path: str = "ai-ethics/"
    humanitarian_context_folder_path: str = "humanitarian-context/"
    ai_technical_folder_path: str = "ai-technical/"
    
    # LLM Settings
    llm_temperature: float = 0.1
    embedding_batch_size: int = 10
    chunk_size: int = 1200
    chunk_overlap: int = 128
    
    # RAG Settings
    rag_enabled: bool = True
    index_refresh_hours: int = 24
    index_storage_path: str = "indexes/"
    
    # Domain-specific chunk limits for targeted retrieval
    max_ai_ethics_chunks: int = 12
    max_humanitarian_context_chunks: int = 10
    max_ai_technical_chunks: int = 12
    max_use_cases_chunks: int = 8
    
    # Similarity thresholds per domain
    ai_ethics_similarity_threshold: float = 0.75
    humanitarian_context_similarity_threshold: float = 0.70
    ai_technical_similarity_threshold: float = 0.72
    use_cases_similarity_threshold: float = 0.68
    
    # Enable/disable domain-specific indexes
    enable_ai_ethics_index: bool = True
    enable_humanitarian_context_index: bool = True
    enable_ai_technical_index: bool = True
    enable_use_cases_bucket: bool = True
   
    # External Data Source APIs
    semantic_scholar_api_key: Optional[str] = None
    reliefweb_api_url: str = "https://api.reliefweb.int/v1"
    reliefweb_app_name: str = "humanitarian-ai-toolkit"
    hdx_api_url: str = "https://data.humdata.org/api/3/action"
    arxiv_api_url: str = "http://export.arxiv.org/api/query"
    
    # Enable/disable external sources
    enable_arxiv: bool = True
    enable_reliefweb: bool = True
    enable_hdx: bool = False
    enable_semantic_scholar: bool = False
    
    # API Performance Settings
    api_request_timeout: int = 45
    api_connect_timeout: int = 15
    api_ssl_timeout: int = 45
    max_concurrent_requests: int = 2
    request_retry_count: int = 3
    request_retry_delay: float = 2.0
   
    # Search Result Limits
    max_results_per_source: int = 50
    max_total_search_results: int = 100
    max_use_cases_returned: int = 6
    max_use_cases_for_enrichment: int = 50
    minimum_relevance_score: float = 0.3
    
    # Search Relevance Weights
    domain_search_weight: float = 0.4
    ai_keywords_weight: float = 0.3
    project_keywords_weight: float = 0.2
    quality_weight: float = 0.1
    
    # Educational Content Settings
    include_educational_content: bool = True
    enable_humanitarian_educational_focus: bool = True
    max_educational_examples: int = 3
    include_real_world_impact: bool = True
    include_decision_guidance: bool = True
    include_similarity_analysis: bool = True
   
    # Fallback Behavior
    use_fallback_on_api_failure: bool = False
    fallback_use_case_count: int = 0
   
    # File Upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: str = ".pdf,.txt,.docx,.csv,.xlsx"
   
    @field_validator('allowed_file_types')
    @classmethod
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',')]
        return v
    
    @field_validator('semantic_scholar_api_key', mode='before')
    @classmethod
    def get_semantic_scholar_key(cls, v):
        return v or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    
    @field_validator('enable_semantic_scholar', mode='before')
    @classmethod
    def set_semantic_scholar_enabled(cls, v):
        # Auto-enable if API key is provided
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")) if v is None else v
   
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validation
def validate_settings():
    """Validate required settings on startup"""
    required_vars = [
        "mongodb_url",
        "openai_api_key", 
        "gcp_bucket_name",
        "gcp_use_cases_bucket_name",
        "gcp_indexes_bucket_name"
    ]
   
    missing_vars = []
    for var in required_vars:
        value = getattr(settings, var, None)
        if not value or (isinstance(value, str) and value.strip() == ""):
            missing_vars.append(var.upper())
   
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate folder paths end with '/'
    folder_paths = [
        settings.ai_ethics_folder_path,
        settings.humanitarian_context_folder_path,
        settings.ai_technical_folder_path
    ]
    
    for i, path in enumerate(folder_paths):
        if path and not path.endswith('/'):
            folder_names = ['ai_ethics_folder_path', 'humanitarian_context_folder_path', 'ai_technical_folder_path']
            print(f"Warning: {folder_names[i]} should end with '/'. Auto-correcting...")
            if i == 0:
                settings.ai_ethics_folder_path = path + '/'
            elif i == 1:
                settings.humanitarian_context_folder_path = path + '/'
            elif i == 2:
                settings.ai_technical_folder_path = path + '/'

def get_domain_chunk_limits():
    """Get chunk limits for all domains"""
    return {
        "ai_ethics": settings.max_ai_ethics_chunks,
        "humanitarian_context": settings.max_humanitarian_context_chunks,
        "ai_technical": settings.max_ai_technical_chunks,
        "use_cases": settings.max_use_cases_chunks
    }

def get_domain_similarity_thresholds():
    """Get similarity thresholds for all domains"""
    return {
        "ai_ethics": settings.ai_ethics_similarity_threshold,
        "humanitarian_context": settings.humanitarian_context_similarity_threshold,
        "ai_technical": settings.ai_technical_similarity_threshold,
        "use_cases": settings.use_cases_similarity_threshold
    }

def get_enabled_domains():
    """Get list of enabled domain indexes"""
    enabled = []
    if settings.enable_ai_ethics_index:
        enabled.append("ai_ethics")
    if settings.enable_humanitarian_context_index:
        enabled.append("humanitarian_context") 
    if settings.enable_ai_technical_index:
        enabled.append("ai_technical")
    if settings.enable_use_cases_bucket:
        enabled.append("use_cases")
    return enabled

def get_enabled_external_sources():
    """Get list of enabled external data sources"""
    enabled = []
    if settings.enable_arxiv:
        enabled.append("arxiv")
    if settings.enable_reliefweb:
        enabled.append("reliefweb")
    if settings.enable_hdx:
        enabled.append("hdx")
    if settings.enable_semantic_scholar:
        enabled.append("semantic_scholar")
    return enabled