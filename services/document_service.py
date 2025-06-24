from typing import List, Optional, Dict, Any
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from config.settings import settings
from utils.document_parser import DocumentParser
from core.exceptions import DataProcessingError
import logging

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for document processing and management"""
    
    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket_name = settings.gcp_bucket_name
        self.parser = DocumentParser()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        project_id: str,
        content_type: str
    ) -> Dict[str, Any]:
        """Upload document to GCS and return metadata"""
        try:
            # Create unique blob name
            blob_name = f"projects/{project_id}/documents/{filename}"
            
            # Get bucket
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            # Upload file
            blob.upload_from_string(
                file_content,
                content_type=content_type
            )
            
            # Set metadata
            blob.metadata = {
                "project_id": project_id,
                "uploaded_at": "2024-01-01T00:00:00Z",  # Would use datetime.utcnow()
                "content_type": content_type
            }
            blob.patch()
            
            logger.info(f"Uploaded document {filename} for project {project_id}")
            
            return {
                "blob_name": blob_name,
                "size": blob.size,
                "content_type": content_type,
                "url": f"gs://{self.bucket_name}/{blob_name}"
            }
            
        except Exception as e:
            logger.error(f"Failed to upload document {filename}: {e}")
            raise DataProcessingError(f"Document upload failed: {str(e)}")
    
    async def process_document(
        self,
        blob_name: str,
        extract_text: bool = True
    ) -> Dict[str, Any]:
        """Process uploaded document and extract content"""
        try:
            # Get blob
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                raise DataProcessingError(f"Document {blob_name} not found")
            
            # Download content
            content = blob.download_as_bytes()
            
            # Extract text based on file type
            extracted_text = ""
            if extract_text:
                if blob_name.lower().endswith('.pdf'):
                    extracted_text = self.parser.parse_pdf_bytes(content)
                elif blob_name.lower().endswith(('.txt', '.md')):
                    extracted_text = self.parser.parse_text_file(content)
                else:
                    logger.warning(f"Unsupported file type for text extraction: {blob_name}")
            
            return {
                "blob_name": blob_name,
                "size": len(content),
                "content_type": blob.content_type,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "processed_at": "2024-01-01T00:00:00Z"
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {blob_name}: {e}")
            raise DataProcessingError(f"Document processing failed: {str(e)}")
    
    async def create_document_index(
        self,
        project_id: str,
        document_paths: Optional[List[str]] = None
    ) -> VectorStoreIndex:
        """Create vector index from project documents"""
        try:
            # Get all documents for project if paths not specified
            if document_paths is None:
                document_paths = await self.list_project_documents(project_id)
            
            # Process documents
            documents = []
            for doc_path in document_paths:
                try:
                    doc_data = await self.process_document(doc_path)
                    if doc_data["extracted_text"]:
                        document = Document(
                            text=doc_data["extracted_text"],
                            metadata={
                                "filename": doc_path.split("/")[-1],
                                "project_id": project_id,
                                "source": f"gs://{self.bucket_name}/{doc_path}",
                                "size": doc_data["size"]
                            }
                        )
                        documents.append(document)
                except Exception as e:
                    logger.warning(f"Failed to process document {doc_path}: {e}")
                    continue
            
            if not documents:
                raise DataProcessingError("No documents available for indexing")
            
            # Create index
            parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            nodes = parser.get_nodes_from_documents(documents)
            
            if not nodes:
                raise DataProcessingError("No nodes created from documents")
            
            index = VectorStoreIndex(nodes, show_progress=True)
            logger.info(f"Created index with {len(nodes)} nodes from {len(documents)} documents")
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to create document index: {e}")
            raise DataProcessingError(f"Index creation failed: {str(e)}")
    
    async def list_project_documents(self, project_id: str) -> List[str]:
        """List all documents for a project"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            prefix = f"projects/{project_id}/documents/"
            
            blobs = bucket.list_blobs(prefix=prefix)
            document_paths = []
            
            for blob in blobs:
                if not blob.name.endswith('/'):  # Skip directories
                    document_paths.append(blob.name)
            
            return document_paths
            
        except Exception as e:
            logger.error(f"Failed to list project documents: {e}")
            return []
    
    async def delete_document(self, blob_name: str) -> bool:
        """Delete document from GCS"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            if blob.exists():
                blob.delete()
                logger.info(f"Deleted document {blob_name}")
                return True
            else:
                logger.warning(f"Document {blob_name} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document {blob_name}: {e}")
            return False
    
    async def get_document_metadata(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                return None
            
            blob.reload()  # Refresh metadata
            
            return {
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "metadata": blob.metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get document metadata {blob_name}: {e}")
            return None
