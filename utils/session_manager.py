import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
import aiohttp
import ssl
from config.settings import settings

logger = logging.getLogger(__name__)

class SessionManager:
    """Centralized HTTP session management with proper cleanup"""
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
    
    @asynccontextmanager
    async def get_session(self):
        """Context manager for HTTP session - ensures proper cleanup"""
        session = None
        try:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = await self._create_session()
                session = self._session
            
            yield session
            
        except Exception as e:
            logger.error(f"Session error: {e}")
            # Close and recreate session on error
            if session and not session.closed:
                await session.close()
            self._session = None
            raise
        finally:
            # Session cleanup happens in close_all_sessions
            pass
    
    async def _create_session(self) -> aiohttp.ClientSession:
        """Create a new HTTP session with proper configuration"""
        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE  # For development
        
        # Configure timeouts
        timeout = aiohttp.ClientTimeout(
            total=settings.api_request_timeout,
            connect=settings.api_connect_timeout,
            sock_read=20,
            sock_connect=10
        )
        
        # Configure connector
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=10,
            limit_per_host=3,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        headers = {
            'User-Agent': 'AI-Humanitarian-Toolkit/1.0 (research purposes)',
            'Accept': 'application/json, application/xml, text/html',
            'Connection': 'close'
        }
        
        return aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=connector,
            raise_for_status=False
        )
    
    async def close_all_sessions(self):
        """Close all active sessions"""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

# Global session manager instance
session_manager = SessionManager()

# Context manager for easy use
@asynccontextmanager
async def get_http_session():
    """Easy-to-use context manager for HTTP sessions"""
    async with session_manager.get_session() as session:
        yield session
