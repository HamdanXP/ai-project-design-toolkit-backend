import asyncio
from datetime import datetime
from services.rag_service import rag_service
import logging

logger = logging.getLogger(__name__)

class IndexRefreshScheduler:
    """Background scheduler for RAG index refresh"""
    
    def __init__(self):
        self.running = False
        self.task = None
    
    async def start(self):
        """Start the background refresh scheduler"""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._refresh_loop())
            logger.info("Index refresh scheduler started")
    
    async def stop(self):
        """Stop the background refresh scheduler"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Index refresh scheduler stopped")
    
    async def _refresh_loop(self):
        """Main refresh loop"""
        while self.running:
            try:
                await rag_service.refresh_indexes_if_needed()
                # Sleep for 1 hour before checking again
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

# Global scheduler instance
refresh_scheduler = IndexRefreshScheduler()