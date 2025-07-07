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
    llm_temperature: float = 0.1
   
    # Google Cloud Storage
    google_application_credentials: str
    google_creds_b64: Optional[str] = None
    gcp_bucket_name: str
    gcp_use_cases_bucket_name: str
    gcp_indexes_bucket_name: str
    
    # Domain-specific folder paths
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
    
    # Similarity thresholds
    ai_ethics_similarity_threshold: float = 0.75
    humanitarian_context_similarity_threshold: float = 0.70
    ai_technical_similarity_threshold: float = 0.72
    use_cases_similarity_threshold: float = 0.68
    
    # Enable/disable domain indexes
    enable_ai_ethics_index: bool = True
    enable_humanitarian_context_index: bool = True
    enable_ai_technical_index: bool = True
    enable_use_cases_bucket: bool = True
   
    # Guidance Settings
    guidance_relevance_threshold: float = 0.7
    max_guidance_sources_per_question: int = 2
    enable_question_guidance: bool = True

    # Academic Sources for AI Use Cases
    semantic_scholar_api_key: Optional[str] = None
    enable_arxiv: bool = True
    enable_semantic_scholar: bool = True
    enable_openalex: bool = True
    enable_papers_with_code: bool = True
    
    # Humanitarian Sources for Datasets
    reliefweb_api_url: str = "https://api.reliefweb.int/v1"
    reliefweb_app_name: str = "humanitarian-ai-toolkit"
    hdx_api_url: str = "https://data.humdata.org/api/3/action"
    enable_reliefweb: bool = True
    enable_hdx: bool = True
    
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
   
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

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

def get_enabled_academic_sources():
    """Get list of enabled academic sources for AI use case search"""
    enabled = []
    if settings.enable_arxiv:
        enabled.append("arxiv")
    if settings.enable_semantic_scholar:
        enabled.append("semantic_scholar")
    if settings.enable_openalex:
        enabled.append("openalex")
    if settings.enable_papers_with_code:
        enabled.append("papers_with_code")
    return enabled

def get_enabled_dataset_sources():
    """Get list of enabled humanitarian sources for dataset discovery"""
    enabled = []
    if settings.enable_reliefweb:
        enabled.append("reliefweb")
    if settings.enable_hdx:
        enabled.append("hdx")
    return enabled