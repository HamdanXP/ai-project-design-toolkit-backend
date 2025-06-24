from typing import List, Optional, Dict, Any
from google.cloud import storage
from llama_index.core import StorageContext, load_index_from_storage
from config.settings import settings
import json
import pickle
import tempfile
import os
import shutil
import logging

logger = logging.getLogger(__name__)

class GCPIndexStorage:
    """Handles storing and loading LlamaIndex storage contexts in GCP"""
    
    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket_name = settings.gcp_indexes_bucket_name
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
    async def save_storage_context(self, storage_context: StorageContext, index_name: str) -> bool:
        """Save storage context to GCP bucket"""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Persist storage context to temp directory
                storage_context.persist(persist_dir=temp_dir)
                
                # Upload each storage file to GCP
                index_path = f"{settings.index_storage_path}{index_name}/"
                
                for file_name in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file_name)
                    blob_name = f"{index_path}{file_name}"
                    
                    blob = self.bucket.blob(blob_name)
                    blob.upload_from_filename(file_path)
                
                # Upload metadata
                metadata = {
                    "index_name": index_name,
                    "created_at": "2024-01-01T00:00:00Z",  # Use datetime.utcnow().isoformat()
                    "storage_files": os.listdir(temp_dir)
                }
                
                metadata_blob = self.bucket.blob(f"{index_path}metadata.json")
                metadata_blob.upload_from_string(json.dumps(metadata))
                
                logger.info(f"Successfully saved index '{index_name}' to GCP")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save storage context for {index_name}: {e}")
            return False
    
    async def load_storage_context(self, index_name: str) -> Optional[StorageContext]:
        """Load storage context from GCP bucket"""
        try:
            index_path = f"{settings.index_storage_path}{index_name}/"
            
            # Check if index exists
            metadata_blob = self.bucket.blob(f"{index_path}metadata.json")
            if not metadata_blob.exists():
                logger.info(f"Index '{index_name}' not found in GCP")
                return None
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download metadata
                metadata_content = metadata_blob.download_as_text()
                metadata = json.loads(metadata_content)
                
                # Download all storage files
                for file_name in metadata["storage_files"]:
                    blob_name = f"{index_path}{file_name}"
                    blob = self.bucket.blob(blob_name)
                    
                    if blob.exists():
                        file_path = os.path.join(temp_dir, file_name)
                        blob.download_to_filename(file_path)
                    else:
                        logger.warning(f"Storage file {blob_name} not found")
                
                # Load storage context from temp directory
                storage_context = StorageContext.from_defaults(persist_dir=temp_dir)
                
                logger.info(f"Successfully loaded index '{index_name}' from GCP")
                return storage_context
                
        except Exception as e:
            logger.error(f"Failed to load storage context for {index_name}: {e}")
            return None
    
    async def index_exists(self, index_name: str) -> bool:
        """Check if index exists in GCP bucket"""
        try:
            index_path = f"{settings.index_storage_path}{index_name}/"
            metadata_blob = self.bucket.blob(f"{index_path}metadata.json")
            return metadata_blob.exists()
        except Exception as e:
            logger.error(f"Failed to check if index exists: {e}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete index from GCP bucket"""
        try:
            index_path = f"{settings.index_storage_path}{index_name}/"
            
            # List all blobs with the index prefix
            blobs = list(self.bucket.list_blobs(prefix=index_path))
            
            # Delete all blobs
            for blob in blobs:
                blob.delete()
            
            logger.info(f"Successfully deleted index '{index_name}' from GCP")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False
    
    async def list_indexes(self) -> List[str]:
        """List all available indexes in GCP bucket"""
        try:
            indexes = []
            blobs = self.bucket.list_blobs(prefix=settings.index_storage_path)
            
            for blob in blobs:
                if blob.name.endswith("metadata.json"):
                    # Extract index name from path
                    path_parts = blob.name.split("/")
                    if len(path_parts) >= 2:
                        index_name = path_parts[-2]  # Second to last part
                        indexes.append(index_name)
            
            return list(set(indexes))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []
    
    async def get_index_metadata(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for specific index"""
        try:
            index_path = f"{settings.index_storage_path}{index_name}/"
            metadata_blob = self.bucket.blob(f"{index_path}metadata.json")
            
            if metadata_blob.exists():
                metadata_content = metadata_blob.download_as_text()
                return json.loads(metadata_content)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get metadata for {index_name}: {e}")
            return None