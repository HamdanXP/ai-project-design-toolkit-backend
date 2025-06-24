from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from services.gcp_index_storage import GCPIndexStorage
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from google.cloud import storage
from config.settings import settings
from core.llm_service import llm_service
from datetime import datetime, timedelta
import os
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

class RAGService:
    """Retrieval-Augmented Generation service for grounding LLM responses"""
    
    def __init__(self):
        self.storage_client = storage.Client()
        self.main_bucket = settings.gcp_bucket_name
        self.use_cases_bucket = settings.gcp_use_cases_bucket_name
        
        # GCP Index storage
        self.index_storage = GCPIndexStorage()
        
        # Index names
        self.main_index_name = "main_knowledge_base"
        self.use_cases_index_name = "use_cases_knowledge_base"
        
        # Initialize indexes
        self.main_index = None
        self.use_cases_index = None
        self.last_refresh = {}
        
    async def initialize_indexes(self):
        """Initialize both indexes on startup"""
        try:
            await self._load_or_create_main_index()
            await self._load_or_create_use_cases_index()
            logger.info("RAG indexes initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG indexes: {e}")
            raise
    
    async def get_context_for_reflection(self, project_description: str) -> str:
        """Get context for reflection phase from main knowledge base"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            query = f"ethical considerations humanitarian AI projects reflection {project_description}"
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get reflection context: {e}")
            return ""
    
    async def get_curated_use_cases_context(self, project_description: str, domain: str) -> List[Dict[str, Any]]:
        """
        Get use cases from dedicated use cases bucket (RENAMED from get_use_cases_context)
        This searches the curated use cases repository, not general knowledge
        """
        if not settings.rag_enabled or not self.use_cases_index:
            logger.info("RAG disabled or use cases index not available")
            return []
        
        try:
            query = f"{domain} humanitarian AI use case implementation {project_description}"
            query_engine = self.use_cases_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            
            # Extract structured use cases from response
            use_cases = await self._parse_use_cases_from_context(str(response), project_description)
            logger.info(f"Retrieved {len(use_cases)} use cases from curated repository")
            return use_cases
        except Exception as e:
            logger.warning(f"Failed to get curated use cases context: {e}")
            return []
    
    async def get_context_for_feasibility(self, project_description: str, constraints: Dict) -> str:
        """Get context for feasibility assessment"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            constraint_text = " ".join([f"{k}:{v}" for k, v in constraints.items()])
            query = f"feasibility assessment humanitarian AI projects {constraint_text} {project_description}"
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get feasibility context: {e}")
            return ""
    
    async def get_context_for_solution_generation(
        self, 
        project_description: str, 
        ai_technique: str, 
        deployment_strategy: str,
        problem_domain: str
    ) -> str:
        """Get context for AI solution generation from humanitarian AI knowledge base"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            query = f"""
            humanitarian AI {ai_technique} implementation {deployment_strategy} 
            {problem_domain} best practices frameworks guidelines 
            {project_description}
            """
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get solution generation context: {e}")
            return ""
    
    async def get_ethical_frameworks_context(
        self, 
        ai_technique: str, 
        project_description: str,
        target_beneficiaries: str
    ) -> str:
        """Get ethical frameworks and guidelines context from documents like EU AI Act, NIST, etc."""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            query = f"""
            AI ethics framework guidelines {ai_technique} humanitarian applications
            {target_beneficiaries} responsible AI principles bias fairness
            transparency accountability {project_description}
            EU AI Act NIST UNICEF ethical AI implementation
            """
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get ethical frameworks context: {e}")
            return ""
    
    async def get_technical_implementation_context(
        self, 
        ai_technique: str, 
        deployment_strategy: str,
        complexity_level: str,
        project_description: str
    ) -> str:
        """Get technical implementation guidance from humanitarian AI case studies"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            query = f"""
            {ai_technique} technical implementation humanitarian {deployment_strategy}
            {complexity_level} complexity architecture patterns deployment
            technical challenges solutions {project_description}
            case studies lessons learned implementation guide
            """
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get technical implementation context: {e}")
            return ""
    
    async def get_deployment_best_practices_context(
        self, 
        deployment_strategy: str, 
        resource_constraints: Dict[str, Any],
        project_description: str
    ) -> str:
        """Get deployment best practices and patterns from humanitarian AI knowledge base"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            constraint_text = " ".join([f"{k}:{v}" for k, v in resource_constraints.items()])
            
            query = f"""
            {deployment_strategy} deployment humanitarian AI best practices
            {constraint_text} implementation patterns infrastructure
            resource optimization {project_description}
            deployment challenges solutions scalability
            """
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get deployment best practices context: {e}")
            return ""
    
    async def get_bias_testing_frameworks_context(
        self, 
        ai_technique: str, 
        target_beneficiaries: str,
        project_description: str
    ) -> str:
        """Get bias testing frameworks and methodologies from ethical AI documents"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            query = f"""
            AI bias testing {ai_technique} fairness evaluation methodologies
            {target_beneficiaries} vulnerable populations bias mitigation
            testing frameworks evaluation metrics {project_description}
            algorithmic fairness humanitarian AI ethics
            """
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get bias testing context: {e}")
            return ""
    
    async def get_monitoring_frameworks_context(
        self, 
        ai_technique: str, 
        deployment_strategy: str,
        project_description: str
    ) -> str:
        """Get AI monitoring and governance frameworks from humanitarian AI guidelines"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            query = f"""
            AI monitoring governance frameworks {ai_technique} {deployment_strategy}
            performance monitoring model drift detection humanitarian AI
            ongoing evaluation metrics {project_description}
            responsible AI governance oversight
            """
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get monitoring frameworks context: {e}")
            return ""
    
    async def get_comprehensive_development_context(
        self,
        project_description: str,
        ai_technique: str,
        deployment_strategy: str,
        complexity_level: str,
        target_beneficiaries: str,
        resource_constraints: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get comprehensive context for all aspects of development phase"""
        
        # Gather context from multiple specialized queries
        contexts = await asyncio.gather(
            self.get_context_for_solution_generation(
                project_description, ai_technique, deployment_strategy, 
                getattr(resource_constraints, 'problem_domain', 'humanitarian')
            ),
            self.get_ethical_frameworks_context(
                ai_technique, project_description, target_beneficiaries
            ),
            self.get_technical_implementation_context(
                ai_technique, deployment_strategy, complexity_level, project_description
            ),
            self.get_deployment_best_practices_context(
                deployment_strategy, resource_constraints, project_description
            ),
            self.get_bias_testing_frameworks_context(
                ai_technique, target_beneficiaries, project_description
            ),
            self.get_monitoring_frameworks_context(
                ai_technique, deployment_strategy, project_description
            ),
            return_exceptions=True
        )
        
        return {
            "solution_generation": contexts[0] if not isinstance(contexts[0], Exception) else "",
            "ethical_frameworks": contexts[1] if not isinstance(contexts[1], Exception) else "",
            "technical_implementation": contexts[2] if not isinstance(contexts[2], Exception) else "",
            "deployment_best_practices": contexts[3] if not isinstance(contexts[3], Exception) else "",
            "bias_testing_frameworks": contexts[4] if not isinstance(contexts[4], Exception) else "",
            "monitoring_frameworks": contexts[5] if not isinstance(contexts[5], Exception) else "",
        }
    
    async def get_real_world_case_studies_context(
        self,
        ai_technique: str,
        problem_domain: str,
        project_description: str
    ) -> str:
        """Get real-world case studies and implementation examples"""
        if not settings.rag_enabled or not self.main_index:
            return ""
        
        try:
            query = f"""
            {ai_technique} humanitarian AI case studies real world implementations
            {problem_domain} success stories lessons learned
            {project_description} similar projects outcomes results
            implementation examples practical applications
            """
            
            query_engine = self.main_index.as_query_engine(
                similarity_top_k=settings.max_context_chunks
            )
            
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to get case studies context: {e}")
            
    async def refresh_indexes_if_needed(self):
        """Check if indexes need refresh and update if necessary"""
        return
        # Commented out for now - manual refresh only
        # try:
        #     current_time = datetime.utcnow()
        #     refresh_interval = timedelta(hours=settings.index_refresh_hours)
            
        #     # Check main index
        #     if (self.main_index is None or 
        #         current_time - self.last_refresh.get('main', datetime.min) > refresh_interval):
        #         logger.info("Refreshing main index...")
        #         await self._refresh_main_index()
        #         self.last_refresh['main'] = current_time
            
        #     # Check use cases index
        #     if (self.use_cases_index is None or 
        #         current_time - self.last_refresh.get('use_cases', datetime.min) > refresh_interval):
        #         logger.info("Refreshing use cases index...")
        #         await self._refresh_use_cases_index()
        #         self.last_refresh['use_cases'] = current_time
                
        # except Exception as e:
        #     logger.error(f"Failed to refresh indexes: {e}")
    
    async def _load_or_create_main_index(self):
        """Load existing main index from GCP or create new one"""
        try:
            # Try to load existing index from GCP
            storage_context = await self.index_storage.load_storage_context(self.main_index_name)
            
            if storage_context:
                self.main_index = load_index_from_storage(storage_context)
                logger.info("Loaded existing main index from GCP")
            else:
                # Create new index
                await self._refresh_main_index()
        except Exception as e:
            logger.warning(f"Failed to load main index from GCP, creating new: {e}")
            await self._refresh_main_index()
    
    async def _load_or_create_use_cases_index(self):
        """Load existing use cases index from GCP or create new one"""
        try:
            # Try to load existing index from GCP
            storage_context = await self.index_storage.load_storage_context(self.use_cases_index_name)
            
            if storage_context:
                self.use_cases_index = load_index_from_storage(storage_context)
                logger.info("Loaded existing use cases index from GCP")
            else:
                # Create new index
                await self._refresh_use_cases_index()
        except Exception as e:
            logger.warning(f"Failed to load use cases index from GCP, creating new: {e}")
            await self._refresh_use_cases_index()
    
    async def _refresh_main_index(self):
        """Refresh main knowledge base index and save to GCP"""
        try:
            documents = await self._load_documents_from_bucket(self.main_bucket)
            if documents:
                self.main_index = await self._create_and_save_index(
                    documents, 
                    self.main_index_name
                )
                logger.info(f"Refreshed main index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to refresh main index: {e}")
    
    async def _refresh_use_cases_index(self):
        """Refresh use cases index and save to GCP"""
        try:
            documents = await self._load_documents_from_bucket(self.use_cases_bucket)
            if documents:
                self.use_cases_index = await self._create_and_save_index(
                    documents, 
                    self.use_cases_index_name
                )
                logger.info(f"Refreshed use cases index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to refresh use cases index: {e}")
    
    async def _load_documents_from_bucket(self, bucket_name: str) -> List[Document]:
        """Load all documents from GCP bucket"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs())
            
            documents = []
            for blob in blobs:
                if blob.name.endswith(('.pdf', '.txt', '.docx', '.md')):
                    try:
                        content = await self._extract_text_from_blob(blob)
                        if content.strip():
                            doc = Document(
                                text=content,
                                metadata={
                                    "filename": blob.name,
                                    "bucket": bucket_name,
                                    "size": blob.size,
                                    "updated": blob.updated.isoformat() if blob.updated else None
                                }
                            )
                            documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to process {blob.name}: {e}")
                        continue
            
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from {bucket_name}: {e}")
            return []
    
    async def _extract_text_from_blob(self, blob) -> str:
        """Extract text content from blob"""
        try:
            if blob.name.endswith('.pdf'):
                # Use PyMuPDF for PDF extraction
                import fitz
                content = blob.download_as_bytes()
                pdf = fitz.open(stream=content, filetype="pdf")
                text = ""
                for page in pdf:
                    text += page.get_text()
                pdf.close()
                return text
            else:
                # For text files
                return blob.download_as_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to extract text from {blob.name}: {e}")
            return ""
    
    async def _create_and_save_index(self, documents: List[Document], index_name: str) -> VectorStoreIndex:
        """Create vector index from documents and save to GCP"""
        try:
            # Configure storage
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                vector_store=SimpleVectorStore(),
                index_store=SimpleIndexStore()
            )
            
            # Parse documents into nodes
            parser = SentenceSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            nodes = parser.get_nodes_from_documents(documents)
            
            # Create index
            index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context,
                show_progress=True
            )
            
            # Save to GCP
            success = await self.index_storage.save_storage_context(
                index.storage_context, 
                index_name
            )
            
            if not success:
                logger.warning(f"Failed to save index {index_name} to GCP")
            
            return index
        except Exception as e:
            logger.error(f"Failed to create and save index: {e}")
            raise

    async def _parse_use_cases_from_context(self, context: str, project_description: str) -> List[Dict[str, Any]]:
        """Parse structured use cases from RAG context"""
        try:
            prompt = f"""
            Based on this context from humanitarian AI use cases documents:
            {context}
            
            And this project description: {project_description}
            
            Extract and return 3-5 relevant use cases in JSON format:
            [
                {{
                    "id": "unique_id",
                    "title": "Use Case Title",
                    "description": "Description from documents",
                    "source": "Use Cases Repository",
                    "complexity": "low/medium/high",
                    "category": "category_name",
                    "how_it_works": "Brief technical explanation",
                    "real_world_impact": "Specific impact example",
                    "suitability_score": 0.8
                }}
            ]
            
            Return ONLY valid JSON array. If no relevant use cases found, return [].
            """
            
            response = await llm_service.analyze_text("", prompt)
            
            # Clean and parse JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            use_cases = json.loads(cleaned_response.strip())
            
            # Validate and enhance use cases
            validated_cases = []
            for uc in use_cases:
                if uc.get("title") and uc.get("description"):
                    # Add missing fields
                    uc.setdefault("source", "Use Cases Repository")
                    uc.setdefault("source_url", "")
                    uc.setdefault("type", "use_case_repository")
                    uc.setdefault("data_completeness", "high")
                    uc.setdefault("technical_requirements", ["See repository documentation"])
                    uc.setdefault("success_factors", ["Implementation details in repository"])
                    uc.setdefault("challenges", ["Context-specific challenges"])
                    uc.setdefault("ethical_considerations", ["Standard AI ethics apply"])
                    uc.setdefault("recommended_for", ["Organizations with similar context"])
                    
                    validated_cases.append(uc)
            
            return validated_cases
            
        except Exception as e:
            logger.warning(f"Failed to parse use cases from context: {e}")
            return []

    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about stored indexes"""
        try:
            indexes = await self.index_storage.list_indexes()
            index_info = {}
            
            for index_name in indexes:
                metadata = await self.index_storage.get_index_metadata(index_name)
                index_info[index_name] = metadata
            
            return index_info
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {}
    
    async def force_refresh_indexes(self):
        """Force refresh of all indexes (useful for admin operations)"""
        try:
            logger.info("Force refreshing all indexes...")
            await self._refresh_main_index()
            await self._refresh_use_cases_index()
            logger.info("Force refresh completed")
        except Exception as e:
            logger.error(f"Failed to force refresh indexes: {e}")
            raise

# Global RAG service instance
rag_service = RAGService()