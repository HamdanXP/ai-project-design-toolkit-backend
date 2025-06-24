from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from config import settings
from models.response import APIResponse
from services.rag_service import rag_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/indexes/info", response_model=APIResponse[Dict[str, Any]])
async def get_indexes_info():
    """Get information about stored indexes"""
    try:
        index_info = await rag_service.get_index_info()
        return APIResponse(
            data=index_info,
            message="Index information retrieved"
        )
    except Exception as e:
        logger.error(f"Failed to get index info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/indexes/refresh", response_model=APIResponse[str])
async def force_refresh_indexes():
    """Force refresh all indexes (admin operation)"""
    try:
        await rag_service.force_refresh_indexes()
        return APIResponse(
            data="success",
            message="Indexes refreshed successfully"
        )
    except Exception as e:
        logger.error(f"Failed to refresh indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indexes/status", response_model=APIResponse[Dict[str, Any]])
async def get_rag_status():
    """Get RAG service status"""
    try:
        status = {
            "rag_enabled": settings.rag_enabled,
            "main_index_loaded": rag_service.main_index is not None,
            "use_cases_index_loaded": rag_service.use_cases_index is not None,
            "last_refresh": rag_service.last_refresh,
            "refresh_interval_hours": settings.index_refresh_hours
        }
        
        return APIResponse(
            data=status,
            message="RAG status retrieved"
        )
    except Exception as e:
        logger.error(f"Failed to get RAG status: {e}")
        raise HTTPException(status_code=500, detail=str(e))