import asyncio
import re
from typing import List, Dict, Any, Optional
from google.cloud import storage
from utils.humanitarian_sources import HumanitarianDataSources
from core.llm_service import llm_service
from config.settings import settings
import logging
import json

logger = logging.getLogger(__name__)

class UseCaseSearchService:
    def __init__(self):
        self.data_sources = HumanitarianDataSources()
        self.storage_client = storage.Client()
        
    async def search_ai_use_cases(
        self, 
        project_description: str, 
        problem_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Search for AI use cases with optimized search strategies
        Returns raw use cases WITHOUT enrichment (enrichment happens in ScopingService)
        """
        logger.info(f"Searching AI use cases for domain: {problem_domain}")
        
        try:
            all_use_cases = []
            
            # Priority 1: Use cases bucket (most relevant)
            if settings.enable_use_cases_bucket:
                try:
                    bucket_cases = await self._search_use_cases_bucket(
                        project_description, problem_domain
                    )
                    all_use_cases.extend(bucket_cases)
                    logger.info(f"Found {len(bucket_cases)} use cases from bucket")
                except Exception as e:
                    logger.warning(f"Use cases bucket search failed: {e}")
            
            # Priority 2: External sources with optimized search
            try:
                external_cases = await self._search_external_sources_optimized(
                    project_description, problem_domain
                )
                all_use_cases.extend(external_cases)
                logger.info(f"Found {len(external_cases)} use cases from external sources")
            except Exception as e:
                logger.warning(f"External sources search failed: {e}")
            
            # Remove duplicates
            unique_cases = self._remove_duplicates(all_use_cases)
            logger.info(f"After deduplication: {len(unique_cases)} unique cases")
            
            # Apply basic quality filter
            quality_filtered = self._apply_basic_quality_filter(unique_cases)
            logger.info(f"After quality filtering: {len(quality_filtered)} cases")
            
            return quality_filtered
            
        except Exception as e:
            logger.error(f"Critical error in AI use case search: {e}")
            return []
    
    async def _search_external_sources_optimized(
        self, 
        project_description: str, 
        problem_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Optimized search across external sources with strategic queries
        Makes ONE call per source with the best query strategy
        """
        all_use_cases = []
        
        # Create optimized search queries for the domain
        search_queries = self._create_optimized_search_queries(problem_domain)
        logger.info(f"Using {len(search_queries)} optimized search queries")
        
        # Search each enabled source with the best query for that source
        search_tasks = []
        
        if settings.enable_arxiv:
            # Use AI-focused query for arXiv
            best_query = search_queries["ai_focused"]
            search_tasks.append(
                self._search_single_source_with_retry("arxiv", best_query, problem_domain)
            )
        
        if settings.enable_semantic_scholar:
            # Use research query for Semantic Scholar
            best_query = search_queries["research_focused"]
            search_tasks.append(
                self._search_single_source_with_retry("semantic_scholar", best_query, problem_domain)
            )
        
        if settings.enable_reliefweb:
            # Use humanitarian query for ReliefWeb
            best_query = search_queries["humanitarian_focused"]
            search_tasks.append(
                self._search_single_source_with_retry("reliefweb", best_query, problem_domain)
            )
        
        if settings.enable_hdx:
            # Use data-focused query for HDX
            best_query = search_queries["data_focused"]
            search_tasks.append(
                self._search_single_source_with_retry("hdx", best_query, problem_domain)
            )
        
        try:
            # Run searches with controlled concurrency
            semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await task
            
            # Execute searches with timeout
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[run_with_semaphore(task) for task in search_tasks],
                    return_exceptions=True
                ),
                timeout=60  # Reduced timeout since we're making fewer calls
            )
            
            # Collect results
            for i, result in enumerate(results):
                source_names = ["arXiv", "Semantic Scholar", "ReliefWeb", "HDX"]
                source_name = source_names[i] if i < len(source_names) else f"Source {i}"
                
                if isinstance(result, Exception):
                    logger.warning(f"{source_name} search failed: {result}")
                elif isinstance(result, list):
                    # Convert results to use cases
                    for item in result:
                        if self._is_result_ai_relevant(item):
                            use_case = self._convert_result_to_use_case(item)
                            if use_case:
                                all_use_cases.append(use_case)
                    logger.info(f"{source_name}: {len(result)} results, {len([r for r in result if self._is_result_ai_relevant(r)])} AI-relevant")
            
            logger.info(f"Total external use cases found: {len(all_use_cases)}")
            return all_use_cases
            
        except asyncio.TimeoutError:
            logger.error("Search timeout - some sources may not have responded")
            return all_use_cases
        except Exception as e:
            logger.error(f"Error in optimized external search: {e}")
            return all_use_cases

    def _create_optimized_search_queries(self, problem_domain: str) -> Dict[str, str]:
        """Create optimized search queries for different source types"""
        domain_terms = self._get_domain_search_terms(problem_domain)
        primary_terms = " OR ".join(domain_terms[:3])
        
        return {
            "ai_focused": f"({primary_terms}) AND (artificial intelligence OR machine learning OR deep learning)",
            "research_focused": f"({primary_terms}) AND (AI OR algorithm OR prediction OR classification)",
            "humanitarian_focused": f"{problem_domain.replace('_', ' ')} AND (technology OR digital OR innovation)",
            "data_focused": f"{problem_domain.replace('_', ' ')} AND (data analysis OR analytics OR monitoring)"
        }

    async def _search_single_source_with_retry(
        self, 
        source_type: str, 
        query: str, 
        domain: str
    ) -> List[Dict[str, Any]]:
        """Search a single source with retry logic"""
        
        for attempt in range(settings.request_retry_count):
            try:
                if source_type == "arxiv":
                    return await self._search_arxiv_optimized(query, domain)
                elif source_type == "semantic_scholar":
                    return await self._search_semantic_scholar_optimized(query, domain)
                elif source_type == "reliefweb":
                    return await self._search_reliefweb_optimized(query, domain)
                elif source_type == "hdx":
                    return await self._search_hdx_optimized(query, domain)
                else:
                    return []
                    
            except Exception as e:
                if attempt < settings.request_retry_count - 1:
                    wait_time = settings.request_retry_delay * (attempt + 1)
                    logger.warning(f"{source_type} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{source_type} failed after {attempt + 1} attempts: {e}")
                    return []
        
        return []

    async def _search_arxiv_optimized(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Optimized arXiv search with single query"""
        results = []
        
        # Use session manager context
        async with self.data_sources.session_manager.get_session() as session:
            try:
                params = {
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": settings.max_results_per_source,
                    "sortBy": "relevance"
                }
                
                async with session.get(
                    "http://export.arxiv.org/api/query", 
                    params=params
                ) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        parsed_papers = self.data_sources._parse_arxiv_xml(xml_content)
                        
                        for paper in parsed_papers:
                            if self.data_sources._is_potentially_relevant(paper, domain):
                                results.append({
                                    "title": paper.get("title", "").strip(),
                                    "description": paper.get("summary", "")[:500] + "..." if len(paper.get("summary", "")) > 500 else paper.get("summary", ""),
                                    "source": "arXiv",
                                    "source_url": paper.get("link", ""),
                                    "type": "academic_paper",
                                    "authors": paper.get("authors", []),
                                    "published_date": paper.get("published", ""),
                                    "categories": paper.get("categories", [])
                                })
                    else:
                        logger.warning(f"arXiv returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"arXiv search failed: {e}")
        
        return results

    async def _search_semantic_scholar_optimized(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Optimized Semantic Scholar search"""
        results = []
        
        headers = {}
        if settings.semantic_scholar_api_key:
            headers['x-api-key'] = settings.semantic_scholar_api_key
        
        async with self.data_sources.session_manager.get_session() as session:
            try:
                params = {
                    "query": query,
                    "limit": settings.max_results_per_source,
                    "fields": "title,abstract,url,year,authors,venue,citationCount,isOpenAccess"
                }
                
                async with session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = data.get("data", [])
                        
                        for paper in papers:
                            title = paper.get("title", "").strip()
                            abstract = paper.get("abstract", "")
                            
                            if title and abstract and len(abstract) > 30:
                                results.append({
                                    "title": title,
                                    "description": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                                    "source": "Semantic Scholar",
                                    "source_url": paper.get("url", ""),
                                    "type": "academic_paper",
                                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                                    "year": paper.get("year"),
                                    "venue": paper.get("venue"),
                                    "citation_count": paper.get("citationCount", 0),
                                    "open_access": paper.get("isOpenAccess", False)
                                })
                    elif response.status == 429:
                        logger.warning("Semantic Scholar rate limit exceeded")
                        await asyncio.sleep(1)
                    else:
                        logger.warning(f"Semantic Scholar returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed: {e}")
        
        return results

    async def _search_reliefweb_optimized(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Optimized ReliefWeb search"""
        results = []
        
        async with self.data_sources.session_manager.get_session() as session:
            try:
                params = {
                    "appname": settings.reliefweb_app_name or "humanitarian-ai-toolkit",
                    "query[value]": query,
                    "query[fields][]": "title",
                    "fields[include][]": "title",
                    "limit": settings.max_results_per_source
                }
                
                async with session.get(
                    f"{settings.reliefweb_api_url}/reports", 
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        reports = data.get("data", [])
                        
                        for report in reports:
                            fields = report.get("fields", {})
                            title = fields.get("title", "").strip()
                            
                            if title:
                                results.append({
                                    "title": title,
                                    "description": f"Humanitarian report from ReliefWeb covering {query} topics. Access full report for detailed information.",
                                    "source": "ReliefWeb",
                                    "source_url": f"https://reliefweb.int{fields.get('url_alias', '')}",
                                    "type": "humanitarian_report",
                                    "date": "",
                                    "organization": "ReliefWeb Partners"
                                })
                    else:
                        logger.warning(f"ReliefWeb returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"ReliefWeb search failed: {e}")
        
        return results

    async def _search_hdx_optimized(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Optimized HDX search"""
        results = []
        
        async with self.data_sources.session_manager.get_session() as session:
            try:
                params = {
                    "q": query,
                    "rows": settings.max_results_per_source,
                    "sort": "score desc",
                    "fq": "capacity:public"
                }
                
                async with session.get(
                    "https://data.humdata.org/api/3/action/package_search", 
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("success"):
                            packages = data.get("result", {}).get("results", [])
                            
                            for package in packages:
                                title = package.get("title", "").strip()
                                notes = package.get("notes", "").strip()
                                
                                if title and len(notes) > 20:
                                    results.append({
                                        "title": title,
                                        "description": notes[:500] + "..." if len(notes) > 500 else notes,
                                        "source": "Humanitarian Data Exchange",
                                        "source_url": f"https://data.humdata.org/dataset/{package.get('name', '')}",
                                        "type": "dataset",
                                        "organization": package.get("organization", {}).get("title", ""),
                                        "tags": [tag.get("name", "") for tag in package.get("tags", [])],
                                        "last_modified": package.get("metadata_modified", ""),
                                        "num_resources": package.get("num_resources", 0)
                                    })
                        else:
                            logger.warning("HDX API returned success=false")
                    else:
                        logger.warning(f"HDX returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"HDX search failed: {e}")
        
        return results

    def _convert_result_to_use_case(self, result: Dict) -> Optional[Dict[str, Any]]:
        """Convert search result to basic use case structure"""
        try:
            title = result.get("title", "").strip()
            description = result.get("description", "").strip()
            
            if not title or len(description) < 50:
                return None
            
            # Determine use case type based on source
            if result.get("type") == "academic_paper":
                return self._convert_academic_to_basic_use_case(result)
            elif result.get("type") in ["humanitarian_report", "dataset"]:
                return self._convert_humanitarian_to_basic_use_case(result)
            else:
                # Generic conversion
                return {
                    "id": f"external_{hash(title) % 10000}",
                    "title": title[:150] + "..." if len(title) > 150 else title,
                    "description": description[:400] + "..." if len(description) > 400 else description,
                    "source": result.get("source", "External Source"),
                    "source_url": result.get("source_url", ""),
                    "type": result.get("type", "external"),
                    "category": "General",
                    "technical_requirements": [],
                    "challenges": []
                }
            
        except Exception as e:
            logger.warning(f"Failed to convert result to use case: {e}")
            return None

    async def _search_use_cases_bucket(
        self, 
        project_description: str, 
        problem_domain: str
    ) -> List[Dict[str, Any]]:
        """Search the dedicated use cases bucket with improved filtering"""
        use_cases = []
        
        try:
            bucket = self.storage_client.bucket(settings.gcp_use_cases_bucket_name)
            blobs = list(bucket.list_blobs())
            
            domain_terms = self._get_domain_search_terms(problem_domain)
            
            for blob in blobs:
                if blob.name.endswith(('.pdf', '.txt', '.md', '.json')):
                    try:
                        if blob.name.endswith('.txt') or blob.name.endswith('.md'):
                            content = blob.download_as_text(encoding='utf-8')
                        else:
                            continue
                        
                        # Enhanced relevance check
                        if self._is_content_ai_relevant(content, domain_terms):
                            use_case = await self._extract_basic_use_case_from_document(
                                content, blob.name, "Use Cases Repository"
                            )
                            if use_case:
                                use_cases.append(use_case)
                                
                    except Exception as e:
                        logger.warning(f"Failed to process {blob.name}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Failed to search use cases bucket: {e}")
        
        return use_cases

    def _is_content_ai_relevant(self, content: str, domain_terms: List[str]) -> bool:
        """Enhanced AI relevance check for content"""
        content_lower = content.lower()
        
        # Check for AI/Technology terms (required)
        ai_terms = [
            "artificial intelligence", "machine learning", "AI", "ML", "deep learning",
            "neural network", "algorithm", "prediction", "classification", "automation",
            "large language model", "LLM", "transformer", "GPT", "BERT",
            "natural language processing", "NLP", "computer vision", "generative AI",
            "data analysis", "predictive", "optimization", "intelligent", "smart"
        ]
        has_ai = any(term in content_lower for term in ai_terms)
        
        # Check for domain relevance
        has_domain = any(domain_term.lower() in content_lower for domain_term in domain_terms[:8])
        
        # Check for humanitarian context
        humanitarian_terms = [
            "humanitarian", "development", "aid", "relief", "crisis", "emergency",
            "social", "community", "welfare", "assistance", "support", "vulnerable",
            "beneficiaries", "displaced", "refugees"
        ]
        has_humanitarian = any(term in content_lower for term in humanitarian_terms)
        
        # Must have AI relevance AND (domain relevance OR humanitarian context)
        return has_ai and (has_domain or has_humanitarian)

    async def _extract_basic_use_case_from_document(
        self, 
        content: str, 
        filename: str, 
        source: str
    ) -> Optional[Dict[str, Any]]:
        """Extract basic use case information without full enrichment"""
        try:
            prompt = f"""
            Extract a structured AI use case from this document content:
            
            Filename: {filename}
            Content: {content[:2000]}...
            
            Extract and return JSON with these BASIC fields only:
            {{
                "title": "Clear, descriptive title of the AI use case",
                "description": "2-3 sentence description of what the AI system does",
                "category": "Prediction|Classification|Optimization|Monitoring|NLP|Computer Vision|Other",
                "type": "ai_application",
                "basic_technical_requirements": ["requirement1", "requirement2"],
                "basic_challenges": ["challenge1", "challenge2"]
            }}
            
            Focus on extracting factual information only. Do NOT add educational content.
            Return ONLY valid JSON. If the content doesn't describe a clear AI use case, return {{"title": null}}.
            """
            
            response = await llm_service.analyze_text("", prompt)
            cleaned_response = self._clean_json_response(response)
            extracted_data = json.loads(cleaned_response)
            
            if not extracted_data.get("title"):
                return None
            
            # Build basic use case structure
            use_case = {
                "id": f"bucket_{hash(filename) % 10000}",
                "title": extracted_data["title"],
                "description": extracted_data["description"],
                "source": source,
                "source_url": f"gs://{settings.gcp_use_cases_bucket_name}/{filename}",
                "type": "use_case_repository",
                "category": extracted_data.get("category", "General"),
                "technical_requirements": extracted_data.get("basic_technical_requirements", []),
                "challenges": extracted_data.get("basic_challenges", [])
            }
            
            return use_case
            
        except Exception as e:
            logger.warning(f"Failed to extract use case from {filename}: {e}")
            return None
    
    def _is_result_ai_relevant(self, result: Dict) -> bool:
        """Enhanced AI relevance check for search results"""
        title = result.get('title', '').lower()
        description = result.get('description', '').lower()
        text_content = f"{title} {description}"
        
        # Enhanced AI keywords including modern terms
        ai_keywords = [
            # Core AI/ML
            "artificial intelligence", "machine learning", "AI", "ML", "deep learning",
            "neural network", "algorithm", "prediction", "classification", "automation",
            
            # Modern AI
            "large language model", "LLM", "transformer", "GPT", "BERT",
            "natural language processing", "NLP", "computer vision", "generative AI",
            
            # Applied AI
            "predictive analytics", "data mining", "pattern recognition", "intelligent system",
            "smart system", "automated analysis", "optimization algorithm", "recommendation system"
        ]
        
        # Count AI term matches
        ai_matches = sum(1 for keyword in ai_keywords if keyword in text_content)
        
        # Technology indicators
        tech_indicators = [
            "technology", "digital", "system", "platform", "tool", "software", 
            "application", "solution", "innovation", "analytics", "data"
        ]
        tech_matches = sum(1 for indicator in tech_indicators if indicator in text_content)
        
        # Must have explicit AI terms OR strong tech indicators
        return ai_matches > 0 or tech_matches >= 3

    def _convert_academic_to_basic_use_case(self, paper: Dict) -> Optional[Dict[str, Any]]:
        """Convert academic paper to basic use case structure"""
        try:
            title = paper.get("title", "").strip()
            description = paper.get("description", "").strip()
            
            if not title or len(description) < 50:
                return None
            
            use_case = {
                "id": f"academic_{hash(title) % 10000}",
                "title": title[:150] + "..." if len(title) > 150 else title,
                "description": description[:400] + "..." if len(description) > 400 else description,
                "source": paper.get("source", "Academic Research"),
                "source_url": paper.get("source_url", ""),
                "type": "academic_paper",
                "category": "Research Application",
                "technical_requirements": [
                    "Access to research publications and methodologies",
                    "Data for validation and testing",
                    "Research collaboration or academic partnership"
                ],
                "challenges": [
                    "Adapting research methods to practical implementation",
                    "Bridging the gap between research and field application",
                    "Securing funding for development phase"
                ],
                "authors": paper.get("authors", []),
                "published_date": paper.get("published_date", ""),
                "citation_count": paper.get("citation_count", 0),
                "venue": paper.get("venue", "")
            }
            
            return use_case
            
        except Exception as e:
            logger.warning(f"Failed to convert academic paper: {e}")
            return None

    def _convert_humanitarian_to_basic_use_case(self, report: Dict) -> Optional[Dict[str, Any]]:
        """Convert humanitarian report to basic use case structure"""
        try:
            title = report.get("title", "").strip()
            description = report.get("description", "").strip()
            
            if not title or len(description) < 50:
                return None
            
            use_case = {
                "id": f"humanitarian_{hash(title) % 10000}",
                "title": title[:150] + "..." if len(title) > 150 else title,
                "description": description[:400] + "..." if len(description) > 400 else description,
                "source": report.get("source", "Humanitarian Organization"),
                "source_url": report.get("source_url", ""),
                "type": "humanitarian_report",
                "category": "Field Implementation",
                "technical_requirements": [
                    "Field deployment infrastructure",
                    "Local data collection capabilities",
                    "Staff training and capacity building"
                ],
                "challenges": [
                    "Operating in challenging field conditions",
                    "Limited technical infrastructure",
                    "Adapting to local contexts and needs"
                ],
                "organization": report.get("organization", ""),
                "date": report.get("date", ""),
                "tags": report.get("tags", [])
            }
            
            return use_case
            
        except Exception as e:
            logger.warning(f"Failed to convert humanitarian report: {e}")
            return None
    
    def _get_domain_search_terms(self, domain: str) -> List[str]:
        """Get comprehensive search terms for a domain"""
        domain_mapping = {
            "health": [
                "health", "medical", "healthcare", "disease", "epidemic", "pandemic", 
                "hospital", "clinic", "patient", "diagnosis", "treatment", "medicine",
                "public health", "epidemiology", "health monitoring", "telemedicine",
                "digital health", "health analytics", "medical AI", "health prediction"
            ],
            "education": [
                "education", "learning", "school", "teaching", "training", "student", 
                "literacy", "curriculum", "educational technology", "e-learning", 
                "adaptive learning", "educational AI", "personalized learning",
                "tutoring system", "learning analytics", "educational assessment"
            ],
            "food_security": [
                "food security", "agriculture", "farming", "nutrition", "hunger", 
                "malnutrition", "crop", "harvest", "agricultural AI", "precision agriculture",
                "food systems", "supply chain", "food distribution", "yield prediction",
                "smart farming", "crop monitoring", "agricultural automation"
            ],
            "water_sanitation": [
                "water", "sanitation", "WASH", "clean water", "hygiene", "water quality",
                "water management", "smart water", "water monitoring", "water AI",
                "water systems", "water scarcity", "water treatment", "water analytics"
            ],
            "disaster_response": [
                "disaster", "emergency", "crisis", "disaster response", "emergency management",
                "early warning", "disaster prediction", "crisis AI", "emergency AI",
                "disaster management", "humanitarian response", "relief operations",
                "emergency coordination", "disaster analytics", "crisis monitoring"
            ],
            "migration_displacement": [
                "migration", "refugee", "displacement", "asylum", "migration AI",
                "population movement", "migration prediction", "refugee management",
                "displacement tracking", "migration patterns", "refugee analytics"
            ],
            "shelter_housing": [
                "shelter", "housing", "accommodation", "settlement", "camp management",
                "infrastructure", "construction", "urban planning", "housing AI",
                "smart shelters", "settlement planning", "housing analytics"
            ],
            "protection": [
                "protection", "human rights", "safety", "security", "violence prevention",
                "child protection", "gender-based violence", "legal aid", "protection AI",
                "risk assessment", "safety monitoring", "protection analytics"
            ]
        }
        
        return domain_mapping.get(domain, [domain.replace('_', ' ')])
    
    def _apply_basic_quality_filter(self, use_cases: List[Dict]) -> List[Dict]:
        """Apply basic quality filter to remove very low quality cases"""
        filtered_cases = []
        
        for case in use_cases:
            # Basic quality criteria
            title = case.get('title', '')
            description = case.get('description', '')
            
            # Skip if title or description is too short or generic
            if (len(title) < 10 or 
                len(description) < 30 or
                title.lower() in ['untitled', 'unknown', 'use case'] or
                description.lower() in ['no description', 'description not available']):
                continue
            
            # Skip if title is mostly numbers or special characters
            if len(re.sub(r'[^a-zA-Z\s]', '', title)) < len(title) * 0.6:
                continue
            
            filtered_cases.append(case)
        
        return filtered_cases
    
    def _remove_duplicates(self, use_cases: List[Dict]) -> List[Dict]:
        """Remove duplicate use cases based on title similarity"""
        seen_titles = set()
        unique_cases = []
        
        for case in use_cases:
            title = case.get('title', '').lower().strip()
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', title)
            normalized_title = ' '.join(normalized_title.split())
            
            if normalized_title and normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_cases.append(case)
        
        return unique_cases

    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON"""
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        start_idx = response.find('{')
        if start_idx != -1:
            response = response[start_idx:]
        
        end_idx = response.rfind('}')
        if end_idx != -1:
            response = response[:end_idx + 1]
        
        return response.strip()