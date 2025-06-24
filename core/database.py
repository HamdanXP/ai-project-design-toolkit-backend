from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from config.settings import settings
from models.project import Project
import logging
import certifi

logger = logging.getLogger(__name__)

class DatabaseManager:
    client: AsyncIOMotorClient = None
    database = None

database_manager = DatabaseManager()

async def connect_to_mongo():
    """Create database connection"""
    try:

        database_manager.client = AsyncIOMotorClient(
            settings.mongodb_url,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=20000,
            socketTimeoutMS=20000
        )
        database_manager.database = database_manager.client[settings.mongodb_database]
        
        # Initialize Beanie with document models
        await init_beanie(
            database=database_manager.database,
            document_models=[Project]
        )
        
        logger.info("Connected to MongoDB successfully")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    if database_manager.client:
        database_manager.client.close()
        logger.info("Disconnected from MongoDB")
