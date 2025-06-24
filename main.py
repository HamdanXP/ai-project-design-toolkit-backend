from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config.settings import settings, validate_settings
from core.database import connect_to_mongo, close_mongo_connection
from api.v1.router import api_router
from utils.humanitarian_sources import HumanitarianDataSources
from services.rag_service import get_rag_service
from core.scheduler import refresh_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global humanitarian data sources instance
humanitarian_sources = HumanitarianDataSources()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    try:
        logger.info("Starting AI Project Design Toolkit API...")
        
        validate_settings()
        await connect_to_mongo()
        
        # Initialize RAG indexes
        await get_rag_service().initialize_indexes()
        
        # Start index refresh scheduler
        await refresh_scheduler.start()
        
        logger.info("API started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down API...")
        await refresh_scheduler.stop()
        await close_mongo_connection()
        logger.info("API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A toolkit for humanitarian professionals to design ethical AI projects",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.api_v1_prefix)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Project Design Toolkit for Humanitarians",
        "version": settings.app_version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "database": "connected",  # Could add actual DB health check
        "llm_service": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )