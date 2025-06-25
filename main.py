import base64
import os
import tempfile
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config.settings import settings, validate_settings
from core.database import connect_to_mongo, close_mongo_connection
from api.v1.router import api_router
from utils.humanitarian_sources import HumanitarianDataSources
from core.scheduler import refresh_scheduler
from services.rag_service import RAGService
import core as ctx

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

        if settings.google_creds_b64:
            temp_path = os.path.join(tempfile.gettempdir(), "google-creds.json")
            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(settings.google_creds_b64))
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        elif settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        else:
            raise EnvironmentError("Missing Google credentials (JSON or path)")
        
        ctx.rag_service = RAGService()

        validate_settings()
        await connect_to_mongo()
        await ctx.rag_service.initialize_indexes()
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
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
