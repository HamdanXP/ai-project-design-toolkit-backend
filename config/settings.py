from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List, Union
import os
import base64
import tempfile

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
   
    # Google Cloud Storage
    google_application_credentials: str
    google_creds_b64: Optional[str] = None  # Optional JSON credentials path
    gcp_bucket_name: str
    gcp_use_cases_bucket_name: str
    gcp_indexes_bucket_name: str
    
    # LLM Settings
    llm_temperature: float = 0.1
    embedding_batch_size: int = 10
    chunk_size: int = 1024
    chunk_overlap: int = 128
    
    # RAG Settings
    rag_enabled: bool = True
    index_refresh_hours: int = 24  # Refresh every 24 hours
    max_context_chunks: int = 5
    context_similarity_threshold: float = 0.7
    index_storage_path: str = "indexes/"  # Path within GCP bucket
   
    # Data Source API Configuration
    semantic_scholar_api_key: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    reliefweb_api_url: str = "https://api.reliefweb.int/v1"
    reliefweb_app_name: str = "humanitarian-ai-toolkit"
    hdx_api_url: str = "https://data.humdata.org/api/3/action"
    arxiv_api_url: str = "http://export.arxiv.org/api/query"
    
    # Rate limiting and timeout settings - OPTIMIZED for better performance
    api_request_timeout: int = 45  # Increased from 30 to 45 seconds
    api_connect_timeout: int = 15  # Increased from 10 to 15 seconds
    api_ssl_timeout: int = 45     # Increased SSL handshake timeout
    max_concurrent_requests: int = 2  # Reduced to 2 to avoid overwhelming servers
    request_retry_count: int = 3   # Increased retries from 2 to 3
    request_retry_delay: float = 2.0  # Increased delay between retries
   
    # Search result limits - ENHANCED with enrichment control
    max_results_per_source: int = 50  # Increased from 4 to 50
    max_total_search_results: int = 100  # Increased from 10 to 100
    max_use_cases_returned: int = 6   # Increased from 10 to 15 (final enriched results)
    max_use_cases_for_enrichment: int = 50  # NEW: Max cases to enrich (before final filtering)
   
    # Enable/disable specific sources
    enable_arxiv: bool = True
    enable_reliefweb: bool = True
    enable_hdx: bool = False  # Disabled HDX for now
    enable_semantic_scholar: bool = os.getenv("ENABLE_SEMANTIC_SCHOLAR", "false").lower() == "true"
    enable_use_cases_bucket: bool = True  # Enable use cases bucket search
   
    # Search relevance settings - ENHANCED for better filtering
    domain_search_weight: float = 0.4    # Increased importance of domain relevance
    ai_keywords_weight: float = 0.3      # AI relevance weight
    project_keywords_weight: float = 0.2 # Project-specific relevance
    quality_weight: float = 0.1          # Content quality weight
    minimum_relevance_score: float = 0.3 # Increased from 0.1 to 0.3 for better quality
    
    # Educational content settings - ENHANCED for humanitarian focus
    include_educational_content: bool = True
    enable_humanitarian_educational_focus: bool = True  # NEW: Focus on humanitarian context
    max_educational_examples: int = 3
    include_real_world_impact: bool = True
    include_decision_guidance: bool = True  # NEW: Help users choose relevant use cases
    include_similarity_analysis: bool = True  # NEW: Analyze similarity to user project
   
    # Fallback behavior - DISABLED
    use_fallback_on_api_failure: bool = False  # Changed from True to False
    fallback_use_case_count: int = 0  # Reduced from 5 to 0
   
    # File Upload
    max_file_size: int = 50 * 1024 * 1024
    allowed_file_types: str = ".pdf,.txt,.docx,.csv,.xlsx"
   
    @field_validator('allowed_file_types')
    @classmethod
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',')]
        return v
   
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
        if not getattr(settings, var, None):
            missing_vars.append(var.upper())
   
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
   
    # Set Google Cloud credentials
    if settings.google_creds_b64:
        temp_path = os.path.join(tempfile.gettempdir(), "google-creds.json")
        decoded = base64.b64decode(settings.google_creds_b64)
        with open(temp_path, "wb") as f:
            f.write(decoded)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
    elif settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    else:
        raise EnvironmentError("Missing Google credentials (JSON or path)")