from typing import List, Dict, Any, Optional, Tuple
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from models.enums import IndexDomain
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
from enum import Enum

logger = logging.getLogger(__name__)

class RAGService:
    """Enhanced Retrieval-Augmented Generation service with folder-based domain organization"""
    
    def __init__(self):
        self.storage_client = storage.Client()
        
        # Main bucket with folder-based organization
        self.main_bucket = settings.gcp_bucket_name
        self.use_cases_bucket = settings.gcp_use_cases_bucket_name
        
        # Folder paths within main bucket for domain-specific documents
        self.domain_folders = {
            IndexDomain.AI_ETHICS: settings.ai_ethics_folder_path,
            IndexDomain.HUMANITARIAN_CONTEXT: settings.humanitarian_context_folder_path,
            IndexDomain.AI_TECHNICAL: settings.ai_technical_folder_path,
        }
        
        # GCP Index storage
        self.index_storage = GCPIndexStorage()
        
        # Domain-specific indexes
        self.indexes = {
            IndexDomain.AI_ETHICS: None,
            IndexDomain.HUMANITARIAN_CONTEXT: None,
            IndexDomain.AI_TECHNICAL: None,
            IndexDomain.USE_CASES: None,
        }
        
        # Index names for storage
        self.index_names = {
            IndexDomain.AI_ETHICS: "ai_ethics_knowledge_base",
            IndexDomain.HUMANITARIAN_CONTEXT: "humanitarian_context_knowledge_base",
            IndexDomain.AI_TECHNICAL: "ai_technical_knowledge_base",
            IndexDomain.USE_CASES: "use_cases_knowledge_base",
        }
        
        # Chunk limits per domain
        self.chunk_limits = {
            IndexDomain.AI_ETHICS: settings.max_ai_ethics_chunks,
            IndexDomain.HUMANITARIAN_CONTEXT: settings.max_humanitarian_context_chunks,
            IndexDomain.AI_TECHNICAL: settings.max_ai_technical_chunks,
            IndexDomain.USE_CASES: settings.max_use_cases_chunks,
        }
        
        # Similarity thresholds per domain
        self.similarity_thresholds = {
            IndexDomain.AI_ETHICS: settings.ai_ethics_similarity_threshold,
            IndexDomain.HUMANITARIAN_CONTEXT: settings.humanitarian_context_similarity_threshold,
            IndexDomain.AI_TECHNICAL: settings.ai_technical_similarity_threshold,
            IndexDomain.USE_CASES: settings.use_cases_similarity_threshold,
        }
        
        self.last_refresh = {}
        self.document_domains = {}
        self.document_classification_cache = {}
        
    async def initialize_indexes(self):
        """Initialize all enabled indexes on startup"""
        try:
            # Initialize use cases index (separate bucket)
            await self._load_or_create_index(IndexDomain.USE_CASES)
            
            # Initialize domain-specific indexes from folders if enabled
            if settings.enable_ai_ethics_index:
                await self._load_or_create_index(IndexDomain.AI_ETHICS)
                
            if settings.enable_humanitarian_context_index:
                await self._load_or_create_index(IndexDomain.HUMANITARIAN_CONTEXT)
                
            if settings.enable_ai_technical_index:
                await self._load_or_create_index(IndexDomain.AI_TECHNICAL)
                        
            logger.info("RAG indexes and document classification initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG indexes: {e}")
            raise
    
    async def get_context_for_reflection(self, project_description: str) -> str:
        """Get context for reflection phase - combines humanitarian principles with AI ethics"""
        if not settings.rag_enabled:
            return ""
        
        try:
            # Combine humanitarian context and AI ethics for comprehensive reflection
            contexts = await self._multi_domain_search(
                query=f"ethical considerations humanitarian AI projects reflection {project_description}",
                domains=[IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_ETHICS],
                combine_results=True
            )
            
            return " ".join(contexts) if contexts else ""
        except Exception as e:
            logger.warning(f"Failed to get reflection context: {e}")
            return ""
    
    async def get_curated_use_cases_context(self, project_description: str, domain: str) -> List[Dict[str, Any]]:
        """Get use cases from dedicated use cases repository"""
        if not settings.rag_enabled or not self.indexes[IndexDomain.USE_CASES]:
            logger.info("RAG disabled or use cases index not available")
            return []
        
        try:
            query = f"{domain} humanitarian AI use case implementation {project_description}"
            response = await self._query_domain_index(
                IndexDomain.USE_CASES, 
                query, 
                self.chunk_limits[IndexDomain.USE_CASES]
            )
            
            # Extract structured use cases from response
            use_cases = await self._parse_use_cases_from_context(str(response), project_description)
            logger.info(f"Retrieved {len(use_cases)} use cases from curated repository")
            return use_cases
        except Exception as e:
            logger.warning(f"Failed to get curated use cases context: {e}")
            return []
    
    async def get_context_for_feasibility(self, project_description: str, constraints: Dict) -> str:
        """Get context for feasibility assessment - combines technical and humanitarian insights"""
        if not settings.rag_enabled:
            return ""
        
        try:
            constraint_text = " ".join([f"{k}:{v}" for k, v in constraints.items()])
            query = f"feasibility assessment humanitarian AI projects {constraint_text} {project_description}"
            
            # Use both technical and humanitarian context for feasibility
            contexts = await self._multi_domain_search(
                query=query,
                domains=[IndexDomain.AI_TECHNICAL, IndexDomain.HUMANITARIAN_CONTEXT],
                combine_results=True
            )
            
            return " ".join(contexts) if contexts else ""
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
        """Get context for AI solution generation - primarily from technical AI knowledge"""
        if not settings.rag_enabled:
            return ""
        
        try:
            query = f"""
            humanitarian AI {ai_technique} implementation {deployment_strategy} 
            {problem_domain} best practices frameworks guidelines 
            {project_description}
            """
            
            # Prioritize AI technical guidance with humanitarian context
            response = await self._query_domain_index(
                IndexDomain.AI_TECHNICAL, 
                query, 
                self.chunk_limits[IndexDomain.AI_TECHNICAL]
            )
            
            return str(response) if response else ""
        except Exception as e:
            logger.warning(f"Failed to get solution generation context: {e}")
            return ""
    
    async def get_ethical_frameworks_context(
        self, 
        ai_technique: str, 
        project_description: str,
        target_beneficiaries: str
    ) -> str:
        """Get ethical frameworks from AI ethics folder (EU AI Act, NIST, etc.)"""
        if not settings.rag_enabled:
            return ""
        
        try:
            query = f"""
            AI ethics framework guidelines {ai_technique} humanitarian applications
            {target_beneficiaries} responsible AI principles bias fairness
            transparency accountability {project_description}
            EU AI Act NIST UNICEF ethical AI implementation
            """
            
            # Use dedicated AI ethics folder for precise ethical guidance
            response = await self._query_domain_index(
                IndexDomain.AI_ETHICS, 
                query, 
                self.chunk_limits[IndexDomain.AI_ETHICS]
            )
            
            return str(response) if response else ""
        except Exception as e:
            logger.warning(f"Failed to get ethical frameworks context: {e}")
            return ""
    
    async def get_technical_implementation_context(
        self, 
        ai_technique: str, 
        deployment_strategy: str,
        project_description: str
    ) -> str:
        """Get technical implementation guidance from AI technical folder"""
        if not settings.rag_enabled:
            return ""
        
        try:
            query = f"""
            {ai_technique} technical implementation humanitarian {deployment_strategy}
            technical challenges solutions {project_description}
            case studies lessons learned implementation guide
            """
            
            response = await self._query_domain_index(
                IndexDomain.AI_TECHNICAL, 
                query, 
                self.chunk_limits[IndexDomain.AI_TECHNICAL]
            )
            
            return str(response) if response else ""
        except Exception as e:
            logger.warning(f"Failed to get technical implementation context: {e}")
            return ""
    
    async def get_deployment_best_practices_context(
        self, 
        deployment_strategy: str, 
        resource_constraints: Dict[str, Any],
        project_description: str
    ) -> str:
        """Get deployment best practices from technical AI knowledge"""
        if not settings.rag_enabled:
            return ""
        
        try:
            constraint_text = " ".join([f"{k}:{v}" for k, v in resource_constraints.items()])
            query = f"""
            {deployment_strategy} deployment humanitarian AI best practices
            {constraint_text} implementation patterns infrastructure
            resource optimization {project_description}
            deployment challenges solutions scalability
            """
            
            response = await self._query_domain_index(
                IndexDomain.AI_TECHNICAL, 
                query, 
                self.chunk_limits[IndexDomain.AI_TECHNICAL]
            )
            
            return str(response) if response else ""
        except Exception as e:
            logger.warning(f"Failed to get deployment best practices context: {e}")
            return ""
    
    async def get_bias_testing_frameworks_context(
        self, 
        ai_technique: str, 
        target_beneficiaries: str,
        project_description: str
    ) -> str:
        """Get bias testing frameworks from AI ethics folder"""
        if not settings.rag_enabled:
            return ""
        
        try:
            query = f"""
            AI bias testing {ai_technique} fairness evaluation methodologies
            {target_beneficiaries} vulnerable populations bias mitigation
            testing frameworks evaluation metrics {project_description}
            algorithmic fairness humanitarian AI ethics
            """
            
            response = await self._query_domain_index(
                IndexDomain.AI_ETHICS, 
                query, 
                self.chunk_limits[IndexDomain.AI_ETHICS]
            )
            
            return str(response) if response else ""
        except Exception as e:
            logger.warning(f"Failed to get bias testing context: {e}")
            return ""
    
    async def get_monitoring_frameworks_context(
        self, 
        ai_technique: str, 
        deployment_strategy: str,
        project_description: str
    ) -> str:
        """Get AI monitoring frameworks - combines ethics and technical guidance"""
        if not settings.rag_enabled:
            return ""
        
        try:
            query = f"""
            AI monitoring governance frameworks {ai_technique} {deployment_strategy}
            performance monitoring model drift detection humanitarian AI
            ongoing evaluation metrics {project_description}
            responsible AI governance oversight
            """
            
            # Combine AI ethics and technical perspectives on monitoring
            contexts = await self._multi_domain_search(
                query=query,
                domains=[IndexDomain.AI_ETHICS, IndexDomain.AI_TECHNICAL],
                combine_results=True
            )
            
            return " ".join(contexts) if contexts else ""
        except Exception as e:
            logger.warning(f"Failed to get monitoring frameworks context: {e}")
            return ""
    
    async def get_comprehensive_development_context(
        self,
        project_description: str,
        ai_technique: str,
        deployment_strategy: str,
        target_beneficiaries: str,
        resource_constraints: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get comprehensive context using domain-specific indexes"""
        
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
                ai_technique, deployment_strategy, project_description
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
        """Get real-world case studies from use cases and technical folders"""
        if not settings.rag_enabled:
            return ""
        
        try:
            query = f"""
            {ai_technique} humanitarian AI case studies real world implementations
            {problem_domain} success stories lessons learned
            {project_description} similar projects outcomes results
            implementation examples practical applications
            """
            
            # Search both use cases and technical folders for comprehensive examples
            contexts = await self._multi_domain_search(
                query=query,
                domains=[IndexDomain.USE_CASES, IndexDomain.AI_TECHNICAL],
                combine_results=True
            )
            
            return " ".join(contexts) if contexts else ""
        except Exception as e:
            logger.warning(f"Failed to get case studies context: {e}")
            return ""

    async def get_question_specific_guidance_sources(
        self, 
        question_text: str,
        question_area: str,
        project_description: str,
        max_sources: int = 2  # Reduced to only get the most relevant
    ) -> List[Dict[str, Any]]:
        """Get highly relevant guidance sources for a specific question"""
        if not settings.rag_enabled:
            return []
        
        try:
            # Extract key concepts from the question for targeted search
            search_query = self._build_targeted_query(question_text, project_description)
            
            logger.info(f"Searching for guidance with query: {search_query[:100]}...")
            
            # Search across relevant domains
            domains_to_search = self._select_relevant_domains(question_area)
            
            all_sources = []
            for domain in domains_to_search:
                if self.indexes[domain] is not None:
                    sources = await self._query_domain_index_with_sources(domain, search_query)
                    
                    # Add domain context to each source
                    for source in sources[:5]:  # Get more candidates for filtering
                        source['guidance_area'] = question_area
                        source['domain_context'] = domain.value
                    
                    all_sources.extend(sources[:5])
            
            if not all_sources:
                logger.info(f"No sources found for question: {question_text[:50]}...")
                return []
            
            # Evaluate relevance and filter
            relevant_sources = self._evaluate_and_filter_sources(
                sources=all_sources,
                question_text=question_text,
                min_relevance_score=0.7,  # Only high-relevance sources
                max_sources=max_sources
            )
            
            logger.info(f"Filtered to {len(relevant_sources)} highly relevant sources for question")
            return relevant_sources
            
        except Exception as e:
            logger.warning(f"Failed to get question-specific guidance: {e}")
            return []

    def _build_targeted_query(self, question_text: str, project_description: str) -> str:
        """Build a targeted search query from the specific question"""
        
        # Extract key terms from the question (simple but effective)
        question_lower = question_text.lower()
        
        # Remove common question words to focus on content
        stopwords = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'will', 
            'should', 'could', 'would', 'do', 'does', 'is', 'are', 'the', 'a', 'an',
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        # Extract meaningful terms (longer than 3 chars, not stopwords)
        words = question_text.split()
        key_terms = [
            word.strip('.,?!') for word in words 
            if len(word) > 3 and word.lower().strip('.,?!') not in stopwords
        ]
        
        # Take top terms + project context
        main_terms = ' '.join(key_terms[:8])  # Limit to avoid too long queries
        project_context = project_description[:200]  # Brief project context
        
        # Build focused query
        query = f"humanitarian AI {main_terms} {project_context} best practices guidance"
        
        return query

    def _select_relevant_domains(self, question_area: str) -> List[IndexDomain]:
        """Select most relevant domains based on question area"""
        
        # Map question areas to most relevant domains
        domain_mapping = {
            "problem_definition": [IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_ETHICS],
            "target_beneficiaries": [IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_ETHICS],
            "potential_harm": [IndexDomain.AI_ETHICS, IndexDomain.HUMANITARIAN_CONTEXT],
            "data_availability": [IndexDomain.AI_ETHICS, IndexDomain.AI_TECHNICAL],
            "technical_feasibility": [IndexDomain.AI_TECHNICAL, IndexDomain.HUMANITARIAN_CONTEXT],
            "stakeholder_involvement": [IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_ETHICS],
            "cultural_sensitivity": [IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_ETHICS],
            "resource_constraints": [IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_TECHNICAL],
            "success_metrics": [IndexDomain.AI_TECHNICAL, IndexDomain.HUMANITARIAN_CONTEXT],
            "sustainability": [IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_TECHNICAL],
            "privacy_security": [IndexDomain.AI_ETHICS, IndexDomain.AI_TECHNICAL]
        }
        
        return domain_mapping.get(question_area, [IndexDomain.HUMANITARIAN_CONTEXT, IndexDomain.AI_ETHICS])

    def _calculate_relevance_score(
        self,
        question_text: str,
        content: str
    ) -> float:
        """Calculate relevance score based purely on content quality and keyword relevance"""
        
        # Extract key terms from question (remove stopwords)
        stopwords = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'will', 
            'should', 'could', 'would', 'do', 'does', 'is', 'are', 'the', 'a', 'an',
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        question_words = set(
            word.strip('.,?!').lower() 
            for word in question_text.split() 
            if len(word) > 3 and word.lower().strip('.,?!') not in stopwords
        )
        
        content_words = set(content.lower().split())
        
        # 1. Direct keyword relevance (60% weight)
        keyword_overlap = 0
        if question_words:
            keyword_overlap = len(question_words.intersection(content_words)) / len(question_words)
        
        # 2. Contextual relevance (40% weight) - Does this content make sense for the question?
        contextual_score = 0
        
        # Check for humanitarian AI context
        if any(term in content.lower() for term in ['humanitarian', 'aid', 'crisis', 'emergency', 'development']):
            contextual_score += 0.25
        
        if any(term in content.lower() for term in ['artificial intelligence', 'machine learning', 'ai', 'algorithm', 'data']):
            contextual_score += 0.25
        
        # Check for practical guidance indicators (most important)
        if any(term in content.lower() for term in ['best practice', 'guideline', 'recommendation', 'framework', 'approach', 'method']):
            contextual_score += 0.3
        
        # Check for specific implementation content
        if any(term in content.lower() for term in ['implementation', 'deploy', 'collect', 'ensure', 'process', 'manage']):
            contextual_score += 0.2
        
        # Combine scores (keyword relevance is most important)
        final_score = (keyword_overlap * 0.6) + (contextual_score * 0.4)
        
        return min(final_score, 1.0)  # Cap at 1.0

    def _evaluate_and_filter_sources(
        self,
        sources: List[Dict[str, Any]],
        question_text: str,
        min_relevance_score: float = 0.7,
        max_sources: int = 2
    ) -> List[Dict[str, Any]]:
        """Evaluate source relevance based purely on content"""
        
        scored_sources = []
        
        for source in sources:
            content = source.get('content', '')
            
            # Calculate relevance score based only on content
            relevance_score = self._calculate_relevance_score(
                question_text=question_text,
                content=content
            )
            
            # Only include sources above threshold
            if relevance_score >= min_relevance_score:
                source['relevance_score'] = relevance_score
                scored_sources.append(source)
        
        # Sort by relevance score and return top sources
        scored_sources.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_sources[:max_sources]

    async def get_ethical_considerations_for_project(
        self, 
        project_description: str, 
        problem_domain: str,
        target_beneficiaries: str = ""
    ) -> List[Dict[str, Any]]:
        """Get contextually specific ethical considerations with complete metadata"""
        if not settings.rag_enabled:
            logger.warning("RAG disabled for ethical considerations")
            return []
        
        try:
            # Extract comprehensive project information
            from services.project_analysis_service import ProjectAnalysisService
            project_analyzer = ProjectAnalysisService()
            
            project_info = await project_analyzer.extract_project_info(project_description)
            
            # Use extracted information, falling back to provided parameters
            extracted_beneficiaries = project_info.get("target_beneficiaries") or target_beneficiaries or None
            extracted_domain = project_info.get("problem_domain") or problem_domain
            geographic_context = project_info.get("geographic_context") or None
            urgency_level = project_info.get("urgency_level") or None
            ai_approach_hints = project_info.get("ai_approach_hints") or None
            
            logger.info(f"Using extracted context: domain={extracted_domain}, beneficiaries={extracted_beneficiaries}")
            
            # Build balanced queries
            balanced_queries = self._build_balanced_ethics_queries(
                project_description=project_description,
                problem_domain=extracted_domain,
                target_beneficiaries=extracted_beneficiaries,
                geographic_context=geographic_context,
                urgency_level=urgency_level,
                ai_approach_hints=ai_approach_hints
            )
            
            # Retrieve context with balanced limits
            all_contexts = []
            max_contexts_per_query = 8
            max_total_contexts = 30
            
            for query_type, query in balanced_queries.items():
                if query and len(all_contexts) < max_total_contexts:
                    logger.info(f"Retrieving {query_type} context...")
                    
                    try:
                        # Get contexts with timeout protection
                        ethics_task = self._query_domain_index_with_sources(IndexDomain.AI_ETHICS, query)
                        humanitarian_task = self._query_domain_index_with_sources(IndexDomain.HUMANITARIAN_CONTEXT, query)
                        
                        # Timeout protection
                        ethics_contexts, humanitarian_contexts = await asyncio.wait_for(
                            asyncio.gather(ethics_task, humanitarian_task, return_exceptions=True),
                            timeout=20.0
                        )
                        
                        # Add contexts with balanced limits
                        if not isinstance(ethics_contexts, Exception) and ethics_contexts:
                            for context in ethics_contexts[:max_contexts_per_query]:
                                context['query_type'] = query_type
                            all_contexts.extend(ethics_contexts[:max_contexts_per_query])
                            
                        if not isinstance(humanitarian_contexts, Exception) and humanitarian_contexts:
                            for context in humanitarian_contexts[:max_contexts_per_query]:
                                context['query_type'] = query_type
                            all_contexts.extend(humanitarian_contexts[:max_contexts_per_query])
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout retrieving {query_type} context, skipping")
                        continue
                    except Exception as e:
                        logger.warning(f"Error retrieving {query_type} context: {e}")
                        continue
            
            if not all_contexts:
                logger.warning("No contexts retrieved for ethical considerations")
                return []
            
            # Create balanced context
            combined_context = self._create_balanced_context(all_contexts, max_length=6000)
            
            logger.info(f"Using {len(all_contexts)} contexts with {len(combined_context)} characters")
            
            # Parse with complete approach
            ethical_considerations = await self._parse_complete_ethical_considerations(
                combined_context=combined_context,
                project_description=project_description,
                project_info=project_info,
                source_documents=all_contexts[:16]
            )
            
            logger.info(f"Retrieved {len(ethical_considerations)} contextually specific ethical considerations")
            return ethical_considerations
            
        except Exception as e:
            logger.error(f"Failed to get ethical considerations: {e}")
            return []

    def _build_balanced_ethics_queries(
        self,
        project_description: str,
        problem_domain: str,
        target_beneficiaries: Optional[str] = None,
        geographic_context: Optional[str] = None,
        urgency_level: Optional[str] = None,
        ai_approach_hints: Optional[str] = None
    ) -> Dict[str, str]:
        """Build balanced queries - not too many, not too few"""
        
        beneficiaries_text = target_beneficiaries or "vulnerable populations"
        domain_text = problem_domain or "humanitarian"
        
        # Get contextual factors
        vulnerability_factors = self._identify_vulnerabilities_from_available_info(
            target_beneficiaries, problem_domain, geographic_context, urgency_level
        )
        sector_risks = self._identify_domain_specific_risks(problem_domain)
        
        queries = {}
        
        # PRIORITY 1: Vulnerability protection (always include)
        queries["vulnerability_protection"] = f"""
        specific protection requirements safeguards {beneficiaries_text}
        {' '.join(vulnerability_factors[:4])} humanitarian AI ethical standards
        vulnerable population protection {domain_text} context ethical requirements
        {project_description} beneficiary protection ethical guidelines
        """
        
        # PRIORITY 2: Sector-specific ethics (always include)
        queries["sector_specific_ethics"] = f"""
        {domain_text} sector specific ethical guidelines standards requirements
        {sector_risks} humanitarian AI implementation ethical requirements
        {domain_text} professional ethics AI technology deployment standards
        sector-specific ethical considerations {project_description}
        """
        
        # PRIORITY 3: AI technique ethics (include if hints available)
        if ai_approach_hints:
            queries["ai_technique_ethics"] = f"""
            {ai_approach_hints} specific ethical challenges mitigation strategies humanitarian
            {ai_approach_hints} algorithmic accountability transparency requirements
            AI bias fairness {ai_approach_hints} humanitarian applications
            {project_description} {ai_approach_hints} ethical implementation guidelines
            """
        
        # PRIORITY 4: Cultural/contextual (include if geographic context available)
        if geographic_context or (target_beneficiaries and any(term in target_beneficiaries.lower() 
                                 for term in ['refugee', 'displaced', 'multi', 'diverse', 'rural', 'traditional'])):
            cultural_elements = self._extract_cultural_elements(geographic_context, target_beneficiaries)
            queries["cultural_contextual"] = f"""
            {cultural_elements} cultural sensitivity AI deployment humanitarian context
            community engagement consent requirements {beneficiaries_text}
            cultural ethical considerations {domain_text} AI systems implementation
            local context ethical requirements {project_description}
            """
        
        # PRIORITY 5: Emergency ethics (include if high urgency)
        if urgency_level in ['high', 'critical']:
            queries["emergency_ethics"] = f"""
            emergency humanitarian AI deployment {urgency_level} urgency ethical considerations
            crisis response AI ethics standards rapid deployment {domain_text}
            emergency consent procedures {beneficiaries_text} ethical requirements
            rapid implementation ethical safeguards {project_description}
            """
        
        return queries

    def _create_balanced_context(self, all_contexts: List[Dict[str, Any]], max_length: int) -> str:
        """Create context with balanced length limits"""
        combined_parts = []
        current_length = 0
        
        # Prioritize contexts by type
        priority_order = ['vulnerability_protection', 'sector_specific_ethics', 'ai_technique_ethics', 
                         'cultural_contextual', 'emergency_ethics']
        
        # Sort contexts by priority
        sorted_contexts = []
        for priority_type in priority_order:
            for context in all_contexts:
                if context.get('query_type') == priority_type:
                    sorted_contexts.append(context)
        
        # Add any remaining contexts
        for context in all_contexts:
            if context not in sorted_contexts:
                sorted_contexts.append(context)
        
        # Build context with balanced limits
        for i, source in enumerate(sorted_contexts):
            # Less aggressive truncation - keep more context
            content = source['content'][:600]  # 600 chars per context
            
            source_entry = f"[SOURCE_{i}] [{source.get('query_type', 'general')}] {content}"
            
            if current_length + len(source_entry) > max_length:
                break
            
            combined_parts.append(source_entry)
            current_length += len(source_entry) + 2
        
        return "\n\n".join(combined_parts)

    def _extract_cultural_elements(
        self,
        geographic_context: Optional[str] = None,
        target_beneficiaries: Optional[str] = None
    ) -> str:
        """Extract cultural considerations from available context"""
        cultural_factors = []
        
        if geographic_context:
            geo_lower = geographic_context.lower()
            if any(term in geo_lower for term in ['rural', 'traditional', 'indigenous']):
                cultural_factors.append('traditional knowledge systems')
            if any(term in geo_lower for term in ['multi', 'diverse', 'ethnic']):
                cultural_factors.append('multi-cultural considerations')
            if any(term in geo_lower for term in ['conflict', 'war', 'crisis']):
                cultural_factors.append('conflict-sensitive approaches')
        
        if target_beneficiaries:
            ben_lower = target_beneficiaries.lower()
            if any(term in ben_lower for term in ['multi', 'diverse', 'ethnic', 'mixed']):
                cultural_factors.append('ethnic diversity considerations')
            if any(term in ben_lower for term in ['traditional', 'indigenous']):
                cultural_factors.append('indigenous rights protection')
            if any(term in ben_lower for term in ['refugee', 'displaced']):
                cultural_factors.append('displacement-sensitive cultural approaches')
        
        return ' '.join(cultural_factors) if cultural_factors else 'cultural sensitivity'

    async def _parse_complete_ethical_considerations(
        self,
        combined_context: str,
        project_description: str,
        project_info: Dict[str, Any],
        source_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Complete parsing with ALL required fields including missing ones"""
        try:
            # Extract key information
            problem_domain = project_info.get("problem_domain", "humanitarian")
            target_beneficiaries = project_info.get("target_beneficiaries")
            ai_approach_hints = project_info.get("ai_approach_hints")
            urgency_level = project_info.get("urgency_level")
            geographic_context = project_info.get("geographic_context")
            
            # Create comprehensive source reference
            source_refs = "\n".join([
                f"SOURCE_{i}: {doc['filename']} (Context: {doc.get('query_type', 'general')}) (from {doc['source_location']})"
                for i, doc in enumerate(source_documents[:12])
            ])
            
            # Enhanced context summary
            context_details = []
            if target_beneficiaries:
                context_details.append(f"Target Beneficiaries: {target_beneficiaries}")
            if geographic_context:
                context_details.append(f"Geographic Context: {geographic_context}")
            if urgency_level:
                context_details.append(f"Urgency Level: {urgency_level}")
            if ai_approach_hints:
                context_details.append(f"AI Approach: {ai_approach_hints}")
            
            context_summary = "\n".join(context_details) if context_details else "Limited context available"
            
            # Comprehensive prompt with ALL required fields
            prompt = f"""
            You are analyzing ethical requirements for a specific humanitarian AI project.
            
            PROJECT DESCRIPTION: {project_description[:300]}
            
            PROJECT CONTEXT:
            - Domain: {problem_domain}
            {context_summary}
            
            EXPERT CONTEXT FROM HUMANITARIAN AND AI ETHICS DOCUMENTS:
            {combined_context}
            
            AVAILABLE SOURCES:
            {source_refs}
            
            INSTRUCTIONS:
            1. Generate 6-8 SPECIFIC ethical considerations uniquely important for THIS project
            2. Focus on the specific vulnerabilities and risks revealed by the project context
            3. Each consideration should be actionable and contextually relevant
            4. Avoid generic ethical principles unless they have specific relevance to this context
            5. Include specific quotes/excerpts from the sources to support each consideration
            
            EXAMPLES OF CONTEXT-SPECIFIC CONSIDERATIONS:
            - For refugee data: "Identity Data Protection to Prevent Deportation Risk"
            - For health AI: "Medical Confidentiality in Multi-Provider Emergency Settings"  
            - For children: "Age-Appropriate Consent Mechanisms for Minors in Crisis"
            - For rural areas: "Algorithmic Bias from Urban-Trained Models in Rural Contexts"
            
            Return 6-8 considerations in JSON format:
            [
                {{
                    "id": "context_specific_id",
                    "title": "Context-Specific Ethical Consideration Title",
                    "description": "Detailed description specific to this project's context and beneficiaries",
                    "category": "data_protection|bias_fairness|transparency|accountability|privacy|safety|do_no_harm|community_engagement|cultural_sensitivity|emergency_ethics",
                    "priority": "high|medium|low",
                    "source_reference": "SOURCE_X",
                    "source_filename": "Document name from SOURCE_X", 
                    "source_excerpt": "Relevant quote from the source (max 200 chars)",
                    "contextual_factors": ["Specific vulnerability 1", "Specific risk 2"],
                    "actionable_steps": ["Specific action 1", "Specific action 2", "Specific action 3"],
                    "why_important": "Why this is specifically critical for this project context",
                    "beneficiary_impact": "How this specifically affects these particular beneficiaries in this context",
                    "risk_if_ignored": "Specific harm that could occur in this context"
                }}
            ]
            
            CRITICAL: Each consideration must be specific to this project's context, not generic ethical principles.
            IMPORTANT: Include specific source excerpts that support each consideration.
            
            Return ONLY valid JSON array.
            """
            
            response = await llm_service.analyze_text("", prompt)
            
            # Clean and parse JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            considerations = json.loads(cleaned_response.strip())
            
            # Validate and enhance with complete metadata
            validated_considerations = []
            for consideration in considerations:
                if consideration.get("title") and consideration.get("description"):
                    # Map source reference to complete metadata
                    source_ref = consideration.get("source_reference", "")
                    if source_ref.startswith("SOURCE_"):
                        try:
                            source_index = int(source_ref.split("_")[1])
                            if 0 <= source_index < len(source_documents):
                                source_doc = source_documents[source_index]
                                
                                source_url = self._generate_public_source_url(
                                    filename=source_doc["filename"],
                                    source_location=source_doc["source_location"]
                                )
                                
                                # Complete metadata update including ALL fields
                                consideration.update({
                                    "source_filename": source_doc["filename"],
                                    "source_bucket": source_doc["bucket"],
                                    "source_folder": source_doc.get("folder", ""),
                                    "source_location": source_doc["source_location"],
                                    "source_domain": source_doc["domain"],
                                    "query_type": source_doc.get("query_type", "general"),
                                    "source_page": source_doc.get("page", ""),
                                    "source_updated": source_doc.get("updated"),  # No default - can be None
                                    "source_size": source_doc.get("size"),        # No default - can be None
                                    "source_url": source_url,
                                    "extracted_context": {
                                        "problem_domain": problem_domain,
                                        "target_beneficiaries": target_beneficiaries,
                                        "geographic_context": geographic_context,
                                        "urgency_level": urgency_level,
                                        "ai_approach_hints": ai_approach_hints
                                    }
                                })
                        except (ValueError, IndexError):
                            pass
                    
                    # Set minimal defaults - avoid misleading values
                    consideration.setdefault("category", None)
                    consideration.setdefault("priority", None)
                    consideration.setdefault("contextual_factors", [])
                    consideration.setdefault("actionable_steps", [])
                    consideration.setdefault("source_excerpt", None)        # Can be None if not extracted
                    consideration.setdefault("beneficiary_impact", None)    # Can be None if not extracted
                    consideration.setdefault("risk_if_ignored", None)       # Can be None if not extracted
                    
                    validated_considerations.append(consideration)
            
            return validated_considerations
            
        except Exception as e:
            logger.warning(f"Failed to parse complete ethical considerations: {e}")
            return []

    def _identify_vulnerabilities_from_available_info(
        self,
        target_beneficiaries: Optional[str] = None,
        problem_domain: Optional[str] = None,
        geographic_context: Optional[str] = None,
        urgency_level: Optional[str] = None
    ) -> List[str]:
        """Identify vulnerability factors from any available information"""
        vulnerabilities = []
        
        if target_beneficiaries:
            beneficiaries_lower = target_beneficiaries.lower()
            
            if any(term in beneficiaries_lower for term in ['refugee', 'displaced', 'asylum']):
                vulnerabilities.extend(['displacement protection', 'identity security'])
            if any(term in beneficiaries_lower for term in ['children', 'minors', 'child']):
                vulnerabilities.extend(['child protection', 'guardian consent'])
            if any(term in beneficiaries_lower for term in ['women', 'girls']):
                vulnerabilities.extend(['gender protection'])
            if any(term in beneficiaries_lower for term in ['elderly', 'disabled']):
                vulnerabilities.extend(['accessibility requirements'])
            if any(term in beneficiaries_lower for term in ['rural', 'remote']):
                vulnerabilities.extend(['digital access barriers'])
        
        if problem_domain:
            domain_lower = problem_domain.lower()
            if 'health' in domain_lower:
                vulnerabilities.extend(['health data privacy'])
            elif any(term in domain_lower for term in ['shelter', 'housing']):
                vulnerabilities.extend(['housing rights'])
            elif 'protection' in domain_lower:
                vulnerabilities.extend(['safety risks'])
        
        if urgency_level in ['high', 'critical']:
            vulnerabilities.extend(['emergency procedures'])
        
        return vulnerabilities

    def _identify_domain_specific_risks(self, problem_domain: str) -> str:
        """Identify sector-specific ethical risks and requirements"""
        if not problem_domain:
            return "humanitarian principles"
            
        domain_lower = problem_domain.lower()
        
        if any(term in domain_lower for term in ['health', 'medical']):
            return 'medical ethics patient privacy'
        elif any(term in domain_lower for term in ['shelter', 'housing', 'camp']):
            return 'housing rights community consent'
        elif any(term in domain_lower for term in ['education', 'learning']):
            return 'educational access child protection'
        elif any(term in domain_lower for term in ['food', 'nutrition']):
            return 'food rights equitable distribution'
        elif any(term in domain_lower for term in ['water', 'sanitation']):
            return 'water rights environmental health'
        elif any(term in domain_lower for term in ['protection', 'security']):
            return 'protection standards confidentiality'
        else:
            return 'humanitarian standards'

    async def _multi_domain_search(
        self, 
        query: str, 
        domains: List[IndexDomain], 
        combine_results: bool = True
    ) -> List[str]:
        """Search across multiple domain indexes and optionally combine results"""
        try:
            # Search all specified domains concurrently
            search_tasks = [
                self._query_domain_index(domain, query, self.chunk_limits[domain])
                for domain in domains
                if self.indexes[domain] is not None
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Filter out exceptions and empty results
            valid_results = [
                str(result) for result in results 
                if not isinstance(result, Exception) and result
            ]
            
            return valid_results
        except Exception as e:
            logger.warning(f"Failed multi-domain search: {e}")
            return []
    
    async def _query_domain_index(
        self, 
        domain: IndexDomain, 
        query: str, 
        similarity_top_k: int
    ) -> Optional[str]:
        """Query a specific domain index"""
        try:
            index = self.indexes[domain]
            if index is None:
                logger.warning(f"Index for domain {domain.value} not available")
                return None
            
            query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.warning(f"Failed to query {domain.value} index: {e}")
            return None
    
    async def _query_domain_index_with_sources(
        self, 
        domain: IndexDomain, 
        query: str
    ) -> List[Dict[str, Any]]:
        """Query domain index and return results with source metadata"""
        try:
            index = self.indexes[domain]
            if index is None:
                return []
            
            retriever = index.as_retriever(similarity_top_k=self.chunk_limits[domain])
            source_nodes = retriever.retrieve(query)
            
            # Compile context with source tracking
            context_with_sources = []
            for i, node in enumerate(source_nodes):
                # Determine the source location (folder or bucket)
                bucket = node.metadata.get("bucket", "")
                filename = node.metadata.get("filename", "Unknown Document")
                
                # For domain folders, include folder info
                if domain in self.domain_folders:
                    folder_path = self.domain_folders[domain]
                    source_location = f"{bucket}/{folder_path}"
                else:
                    source_location = bucket
                
                source_info = {
                    "content": node.text,
                    "source_id": f"source_{i}",
                    "filename": filename,
                    "bucket": bucket,
                    "folder": self.domain_folders.get(domain, ""),
                    "domain": domain.value,
                    "source_location": source_location,
                    "page": node.metadata.get("page"),
                    "updated": node.metadata.get("updated"),  # Can be None
                    "size": node.metadata.get("size")         # Can be None
                }
                context_with_sources.append(source_info)
            
            return context_with_sources
        except Exception as e:
            logger.warning(f"Failed to query {domain.value} index with sources: {e}")
            return []
    
    async def refresh_indexes_if_needed(self):
        """Check if indexes need refresh and update if necessary"""
        return  # Manual refresh only for now
    
    async def _load_or_create_index(self, domain: IndexDomain):
        """Load existing index from GCP or create new one for specified domain"""
        try:
            index_name = self.index_names[domain]
            
            # Try to load existing index from GCP
            storage_context = await self.index_storage.load_storage_context(index_name)
            
            if storage_context:
                self.indexes[domain] = load_index_from_storage(storage_context)
                logger.info(f"Loaded existing {domain.value} index from GCP")
            else:
                # Create new index
                await self._refresh_domain_index(domain)
        except Exception as e:
            logger.warning(f"Failed to load {domain.value} index from GCP, creating new: {e}")
            await self._refresh_domain_index(domain)
    
    async def _refresh_domain_index(self, domain: IndexDomain):
        """Refresh domain-specific index and save to GCP"""
        try:
            documents = await self._load_documents_for_domain(domain)
            if documents:
                self.indexes[domain] = await self._create_and_save_index(
                    documents, 
                    self.index_names[domain]
                )
                logger.info(f"Refreshed {domain.value} index with {len(documents)} documents")
            else:
                logger.warning(f"No documents found for domain {domain.value}")
        except Exception as e:
            logger.error(f"Failed to refresh {domain.value} index: {e}")
    
    async def _load_documents_for_domain(self, domain: IndexDomain) -> List[Document]:
        """Load documents for a specific domain (either from folder or separate bucket)"""
        try:
            if domain == IndexDomain.USE_CASES:
                # Load from separate use cases bucket
                return await self._load_documents_from_bucket(self.use_cases_bucket)
            else:
                # Load from folder within main bucket
                folder_path = self.domain_folders.get(domain)
                if not folder_path:
                    logger.warning(f"No folder path configured for domain {domain.value}")
                    return []
                
                return await self._load_documents_from_folder(self.main_bucket, folder_path)
        except Exception as e:
            logger.error(f"Failed to load documents for domain {domain.value}: {e}")
            return []
    
    async def _load_documents_from_folder(self, bucket_name: str, folder_path: str) -> List[Document]:
        """Load documents from a specific folder within a bucket"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            
            # List blobs with the folder prefix
            blobs = list(bucket.list_blobs(prefix=folder_path))
            
            documents = []
            for blob in blobs:
                # Skip folder "directories" (they end with /)
                if blob.name.endswith('/'):
                    continue
                    
                if blob.name.endswith(('.pdf', '.txt', '.docx', '.md')):
                    try:
                        content = await self._extract_text_from_blob(blob)
                        if content.strip():
                            doc = Document(
                                text=content,
                                metadata={
                                    "filename": blob.name.replace(folder_path, ""),  # Remove folder prefix
                                    "full_path": blob.name,
                                    "bucket": bucket_name,
                                    "folder": folder_path,
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
            logger.error(f"Failed to load documents from {bucket_name}/{folder_path}: {e}")
            return []
    
    async def _load_documents_from_bucket(self, bucket_name: str) -> List[Document]:
        """Load all documents from GCP bucket (for use cases)"""
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
                import fitz
                content = blob.download_as_bytes()
                pdf = fitz.open(stream=content, filetype="pdf")
                text = ""
                for page in pdf:
                    text += page.get_text()
                pdf.close()
                return text
            else:
                return blob.download_as_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to extract text from {blob.name}: {e}")
            return ""
    
    async def _create_and_save_index(self, documents: List[Document], index_name: str) -> VectorStoreIndex:
        """Create vector index from documents and save to GCP"""
        try:
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                vector_store=SimpleVectorStore(),
                index_store=SimpleIndexStore()
            )
            
            parser = SentenceSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            nodes = parser.get_nodes_from_documents(documents)
            
            index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context,
                show_progress=True
            )
            
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
            
            # Validate and enhance use cases with minimal defaults
            validated_cases = []
            for uc in use_cases:
                if uc.get("title") and uc.get("description"):
                    # Add missing fields with minimal defaults
                    uc.setdefault("source", "Use Cases Repository")
                    uc.setdefault("source_url", "")
                    uc.setdefault("type", "use_case_repository")
                    uc.setdefault("data_completeness", None)  # Can be None
                    uc.setdefault("technical_requirements", [])
                    uc.setdefault("success_factors", [])
                    uc.setdefault("challenges", [])
                    uc.setdefault("ethical_considerations", [])
                    uc.setdefault("recommended_for", [])
                    
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
            
            # Add domain-specific information
            domain_info = {}
            for domain, index in self.indexes.items():
                if domain == IndexDomain.USE_CASES:
                    source_location = self.use_cases_bucket
                else:
                    folder_path = self.domain_folders.get(domain, "")
                    source_location = f"{self.main_bucket}/{folder_path}"
                
                domain_info[domain.value] = {
                    "loaded": index is not None,
                    "source_location": source_location,
                    "chunk_limit": self.chunk_limits[domain],
                    "similarity_threshold": self.similarity_thresholds[domain]
                }
            
            return {
                "stored_indexes": index_info,
                "domain_indexes": domain_info,
                "folder_organization": {
                    "main_bucket": self.main_bucket,
                    "use_cases_bucket": self.use_cases_bucket,
                    "domain_folders": {domain.value: path for domain, path in self.domain_folders.items()}
                }
            }
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {}
    
    async def force_refresh_indexes(self):
        """Force refresh of all indexes"""
        try:
            logger.info("Force refreshing all indexes...")
            
            # Refresh all enabled indexes
            refresh_tasks = []
            
            for domain in self.indexes.keys():
                refresh_tasks.append(self._refresh_domain_index(domain))
            
            await asyncio.gather(*refresh_tasks, return_exceptions=True)
            
            logger.info("Force refresh completed")
        except Exception as e:
            logger.error(f"Failed to force refresh indexes: {e}")
            raise

    def _generate_public_source_url(self, filename: str, source_location: str) -> str:
        """Generate direct URL to public GCP Storage document"""
        if not filename or not source_location:
            return ""
        
        return f"https://storage.googleapis.com/{source_location}{filename}"