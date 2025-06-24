import asyncio
import re
from typing import List, Dict, Any, Optional
from models.project import Dataset
from config.settings import settings
from utils.session_manager import session_manager
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class HumanitarianDataSources:
    def __init__(self):
        self.max_results_per_source = settings.max_results_per_source
        self.session_manager = session_manager  # Add reference for use in search service
    
    async def _search_with_retry(
        self, 
        search_func, 
        query: str, 
        domain: str, 
        limit: int, 
        source_name: str
    ) -> List[Dict[str, Any]]:
        """Execute search with retry logic"""
        for attempt in range(settings.request_retry_count + 1):
            try:
                result = await search_func(query, domain, limit)
                return result
            except Exception as e:
                if attempt < settings.request_retry_count:
                    wait_time = settings.request_retry_delay * (attempt + 1)
                    logger.warning(f"{source_name} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{source_name} failed after {attempt + 1} attempts: {e}")
                    return []
        return []
    
    async def _search_arxiv_safe(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search arXiv with optimized single query strategy"""
        results = []
        
        # Use session manager context
        async with session_manager.get_session() as session:
            try:
                params = {
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance"
                }
                
                async with session.get(
                    "http://export.arxiv.org/api/query", 
                    params=params
                ) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        parsed_papers = self._parse_arxiv_xml(xml_content)
                        
                        for paper in parsed_papers:
                            # More lenient relevance check
                            if self._is_potentially_relevant(paper, domain):
                                result = {
                                    "title": paper.get("title", "").strip(),
                                    "description": paper.get("summary", "")[:500] + "..." if len(paper.get("summary", "")) > 500 else paper.get("summary", ""),
                                    "source": "arXiv",
                                    "source_url": paper.get("link", ""),
                                    "type": "academic_paper",
                                    "authors": paper.get("authors", []),
                                    "published_date": paper.get("published", ""),
                                    "categories": paper.get("categories", [])
                                }
                                
                                # Avoid duplicates by checking titles
                                if not any(existing.get("title") == result["title"] for existing in results):
                                    results.append(result)
                                        
                    else:
                        logger.warning(f"arXiv returned status {response.status} for query: {query}")
                            
            except Exception as e:
                logger.warning(f"arXiv search failed: {query} - {e}")
        
        return results[:limit]

    async def _search_reliefweb_safe(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search ReliefWeb with optimized single query"""
        results = []
        
        base_url = "https://api.reliefweb.int/v1"
        
        # Use session manager context
        async with session_manager.get_session() as session:
            try:
                # Use the working parameter format with increased limits
                params = {
                    "appname": settings.reliefweb_app_name or "humanitarian-ai-toolkit",
                    "query[value]": query,
                    "query[fields][]": "title",
                    "fields[include][]": "title",
                    "limit": limit
                }
                
                async with session.get(f"{base_url}/reports", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        reports = data.get("data", [])
                        
                        for report in reports:
                            fields = report.get("fields", {})
                            title = fields.get("title", "").strip()
                            
                            if title:
                                result = {
                                    "title": title,
                                    "description": f"Humanitarian report from ReliefWeb covering {query} topics. Access full report for detailed information.",
                                    "source": "ReliefWeb",
                                    "source_url": f"https://reliefweb.int{fields.get('url_alias', '')}",
                                    "type": "humanitarian_report",
                                    "date": "",
                                    "organization": "ReliefWeb Partners"
                                }
                                
                                if not any(existing.get("title") == result["title"] for existing in results):
                                    results.append(result)
                                    
                    else:
                        logger.warning(f"ReliefWeb returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"ReliefWeb search failed for '{query}': {e}")
        
        return results[:limit]

    async def _search_hdx_safe(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search HDX with optimized single query"""
        results = []
        
        # Use session manager context
        async with session_manager.get_session() as session:
            try:
                params = {
                    "q": query,
                    "rows": limit,
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
                                    result = {
                                        "title": title,
                                        "description": notes[:500] + "..." if len(notes) > 500 else notes,
                                        "source": "Humanitarian Data Exchange",
                                        "source_url": f"https://data.humdata.org/dataset/{package.get('name', '')}",
                                        "type": "dataset",
                                        "organization": package.get("organization", {}).get("title", ""),
                                        "tags": [tag.get("name", "") for tag in package.get("tags", [])],
                                        "last_modified": package.get("metadata_modified", ""),
                                        "num_resources": package.get("num_resources", 0)
                                    }
                                    
                                    if not any(existing.get("title") == result["title"] for existing in results):
                                        results.append(result)
                        else:
                            logger.warning("HDX API returned success=false")
                    else:
                        logger.warning(f"HDX returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"HDX search failed for '{query}': {e}")
        
        return results[:limit]

    async def _search_semantic_scholar_safe(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search Semantic Scholar with optimized single query"""
        results = []
        
        headers = {}
        if settings.semantic_scholar_api_key:
            headers['x-api-key'] = settings.semantic_scholar_api_key
        
        # Use session manager context
        async with session_manager.get_session() as session:
            try:
                params = {
                    "query": query,
                    "limit": limit,
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
                                result = {
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
                                }
                                
                                if not any(existing.get("title") == result["title"] for existing in results):
                                    results.append(result)
                                    
                    elif response.status == 429:
                        logger.warning("Semantic Scholar rate limit exceeded")
                        await asyncio.sleep(1)
                    else:
                        logger.warning(f"Semantic Scholar returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
        
        return results[:limit]

    # This method is now deprecated - individual sources should be called directly
    async def search_all_sources(
        self, 
        query: str, 
        domain: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        DEPRECATED: This method causes multiple redundant calls
        Use individual source methods directly instead
        """
        logger.warning("search_all_sources is deprecated - use individual source methods instead")
        return []
    
    def _is_potentially_relevant(self, paper: Dict, domain: str) -> bool:
        """More lenient relevance check"""
        title = paper.get('title', '').lower()
        summary = paper.get('summary', '').lower()
        text_content = f"{title} {summary}"
        
        # Check for domain terms (more lenient)
        domain_terms = self._get_domain_terms(domain)
        has_domain = any(term.lower() in text_content for term in domain_terms)
        
        # Or check for general humanitarian terms
        humanitarian_terms = ['humanitarian', 'social', 'development', 'crisis', 'aid', 'welfare', 'community', 'relief']
        has_humanitarian = any(term in text_content for term in humanitarian_terms)
        
        # Or check for AI/tech terms
        ai_terms = ['artificial intelligence', 'machine learning', 'algorithm', 'prediction', 'classification', 'data', 'technology', 'digital']
        has_ai = any(term in text_content for term in ai_terms)
        
        return (has_domain or has_humanitarian) and has_ai
    
    def _get_domain_terms(self, domain: str) -> List[str]:
        """Get domain-specific search terms"""
        domain_mapping = {
            "food_security": ["food", "nutrition", "agriculture", "farming", "crop", "harvest", "hunger", "malnutrition"],
            "health": ["health", "medical", "disease", "healthcare", "clinic", "hospital", "treatment", "epidemic"],
            "education": ["education", "learning", "school", "teaching", "literacy", "student", "training"],
            "water_sanitation": ["water", "sanitation", "hygiene", "wash", "clean water"],
            "disaster_response": ["disaster", "emergency", "crisis", "response", "relief", "warning"],
            "migration_displacement": ["migration", "refugee", "displacement", "asylum"],
            "protection": ["protection", "safety", "security", "violence", "rights"],
            "livelihoods": ["livelihood", "income", "employment", "economic", "poverty"]
        }
        
        return domain_mapping.get(domain, [domain.replace('_', ' ')])
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response safely - IMPROVED"""
        papers = []
        
        try:
            # Clean XML content first
            xml_content = xml_content.replace('xmlns="http://www.w3.org/2005/Atom"', '')
            root = ET.fromstring(xml_content)
            
            # Find entries without namespace
            entries = root.findall('.//entry')
            
            for entry in entries:
                paper = {}
                
                # Extract title
                title_elem = entry.find('title')
                if title_elem is not None:
                    paper['title'] = re.sub(r'\s+', ' ', title_elem.text).strip()
                
                # Extract summary
                summary_elem = entry.find('summary')
                if summary_elem is not None:
                    paper['summary'] = re.sub(r'\s+', ' ', summary_elem.text).strip()
                
                # Extract link
                link_elem = entry.find('link[@type="text/html"]')
                if link_elem is not None:
                    paper['link'] = link_elem.get('href')
                else:
                    # Fallback to id link
                    id_elem = entry.find('id')
                    if id_elem is not None:
                        paper['link'] = id_elem.text.strip()
                
                # Extract authors
                authors = []
                for author in entry.findall('author'):
                    name_elem = author.find('name')
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                paper['authors'] = authors
                
                # Extract published date
                published_elem = entry.find('published')
                if published_elem is not None:
                    paper['published'] = published_elem.text.strip()
                
                # Extract categories
                categories = []
                for category in entry.findall('category'):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                paper['categories'] = categories
                
                # Only include papers with title and summary
                if paper.get('title') and paper.get('summary'):
                    papers.append(paper)
                    
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {e}")
            # Try alternative parsing approach
            try:
                # Use namespace-aware parsing
                root = ET.fromstring(xml_content)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                
                for entry in root.findall('atom:entry', namespace):
                    paper = {}
                    
                    # Extract title
                    title_elem = entry.find('atom:title', namespace)
                    if title_elem is not None:
                        paper['title'] = re.sub(r'\s+', ' ', title_elem.text).strip()
                    
                    # Extract summary
                    summary_elem = entry.find('atom:summary', namespace)
                    if summary_elem is not None:
                        paper['summary'] = re.sub(r'\s+', ' ', summary_elem.text).strip()
                    
                    # Extract link
                    link_elem = entry.find('atom:link[@type="text/html"]', namespace)
                    if link_elem is not None:
                        paper['link'] = link_elem.get('href')
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', namespace):
                        name_elem = author.find('atom:name', namespace)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    paper['authors'] = authors
                    
                    # Extract published date
                    published_elem = entry.find('atom:published', namespace)
                    if published_elem is not None:
                        paper['published'] = published_elem.text.strip()
                    
                    # Extract categories
                    categories = []
                    for category in entry.findall('atom:category', namespace):
                        term = category.get('term')
                        if term:
                            categories.append(term)
                    paper['categories'] = categories
                    
                    # Only include papers with title and summary
                    if paper.get('title') and paper.get('summary'):
                        papers.append(paper)
                        
            except Exception as e2:
                logger.error(f"Failed alternative ArXiv XML parsing: {e2}")
        except Exception as e:
            logger.error(f"Unexpected error parsing ArXiv XML: {e}")
        
        logger.info(f"Parsed {len(papers)} papers from ArXiv XML")
        return papers
    
    # Humanitarian dataset methods - UPDATED to use search query
    async def get_reliefweb_datasets(self, search_query: str) -> List[Dataset]:
        """Get datasets from ReliefWeb search results based on search query"""
        logger.info(f"Searching ReliefWeb for datasets with query: {search_query}")
        
        # Extract key terms from search query for better matching
        key_terms = [term.strip() for term in search_query.split() if len(term.strip()) > 3][:5]
        search_term = " ".join(key_terms)
        
        results = await self._search_reliefweb_safe(search_term, "general", 10)
        
        datasets = []
        for result in results:
            dataset = Dataset(
                name=result.get("title", "Unknown"),
                source="ReliefWeb",
                url=result.get("source_url", ""),
                description=result.get("description", ""),
                data_types=["humanitarian_reports", "text"],
                ethical_concerns=["privacy", "consent"],
                suitability_score=0.7
            )
            datasets.append(dataset)
        
        logger.info(f"Found {len(datasets)} datasets from ReliefWeb")
        return datasets
    
    async def get_hdx_datasets(self, search_query: str) -> List[Dataset]:
        """Get datasets from HDX (Humanitarian Data Exchange) based on search query"""
        logger.info(f"Searching HDX for datasets with query: {search_query}")
        
        # Extract key terms from search query for better matching
        key_terms = [term.strip() for term in search_query.split() if len(term.strip()) > 3][:5]
        search_term = " ".join(key_terms)
        
        results = await self._search_hdx_safe(search_term, "general", 15)
        
        datasets = []
        for result in results:
            # Only process actual datasets from HDX
            if result.get("type") == "dataset":
                dataset = Dataset(
                    name=result.get("title", "Unknown"),
                    source="Humanitarian Data Exchange (HDX)",
                    url=result.get("source_url", ""),
                    description=result.get("description", ""),
                    data_types=self._extract_data_types_from_hdx(result),
                    ethical_concerns=["data_privacy", "beneficiary_consent", "responsible_data_use"],
                    suitability_score=0.8  # HDX datasets tend to be high quality
                )
                
                # Add additional HDX-specific metadata
                if result.get("organization"):
                    dataset.description += f" | Organization: {result['organization']}"
                if result.get("tags"):
                    dataset.description += f" | Tags: {', '.join(result['tags'][:3])}"
                if result.get("num_resources"):
                    dataset.description += f" | Resources: {result['num_resources']} files"
                
                datasets.append(dataset)
        
        logger.info(f"Found {len(datasets)} datasets from HDX")
        return datasets

    def _extract_data_types_from_hdx(self, hdx_result: Dict) -> List[str]:
        """Extract data types from HDX dataset metadata"""
        data_types = ["humanitarian_data"]
        
        # Analyze tags to determine data types
        tags = hdx_result.get("tags", [])
        tag_text = " ".join(tags).lower()
        
        if any(term in tag_text for term in ["population", "demographic", "census"]):
            data_types.append("demographic")
        if any(term in tag_text for term in ["health", "medical", "disease"]):
            data_types.append("health")
        if any(term in tag_text for term in ["education", "school", "literacy"]):
            data_types.append("education")
        if any(term in tag_text for term in ["food", "nutrition", "agriculture"]):
            data_types.append("food_security")
        if any(term in tag_text for term in ["water", "sanitation", "wash"]):
            data_types.append("water_sanitation")
        if any(term in tag_text for term in ["disaster", "emergency", "crisis"]):
            data_types.append("disaster_response")
        if any(term in tag_text for term in ["refugee", "migration", "displacement"]):
            data_types.append("migration")
        if any(term in tag_text for term in ["protection", "violence", "safety"]):
            data_types.append("protection")
        if any(term in tag_text for term in ["economic", "livelihood", "employment"]):
            data_types.append("economic")
        if any(term in tag_text for term in ["geographic", "geospatial", "location"]):
            data_types.append("geospatial")
        
        # Analyze title and description for additional context
        title_desc = f"{hdx_result.get('title', '')} {hdx_result.get('description', '')}".lower()
        
        if "survey" in title_desc:
            data_types.append("survey")
        if "assessment" in title_desc:
            data_types.append("assessment")
        if "monitoring" in title_desc:
            data_types.append("monitoring")
        if any(term in title_desc for term in ["admin", "boundary", "administrative"]):
            data_types.append("administrative")
        
        return list(set(data_types))  # Remove duplicates
    
    async def get_un_datasets(self, search_query: str) -> List[Dataset]:
        """Get standard UN datasets filtered by search query relevance"""
        logger.info(f"Filtering UN datasets for query: {search_query}")
        
        all_datasets = [
            Dataset(
                name="UNHCR Global Trends",
                source="UN High Commissioner for Refugees",
                url="https://www.unhcr.org/global-trends-report",
                description="Annual statistics on refugees, asylum-seekers, internally displaced and stateless people",
                data_types=["statistical", "demographic", "temporal"],
                ethical_concerns=["anonymization", "sensitive_populations"],
                suitability_score=0.8
            ),
            Dataset(
                name="WHO Health Emergency Data",
                source="World Health Organization",
                url="https://www.who.int/emergencies/surveillance",
                description="Global health emergency surveillance and outbreak data",
                data_types=["health", "epidemiological", "geospatial"],
                ethical_concerns=["health_privacy", "community_consent"],
                suitability_score=0.75
            ),
            Dataset(
                name="WFP Food Security Monitoring",
                source="World Food Programme",
                url="https://hungermap.wfp.org/",
                description="Real-time food security monitoring and hunger data",
                data_types=["food_security", "economic", "satellite"],
                ethical_concerns=["community_privacy", "economic_sensitivity"],
                suitability_score=0.85
            ),
            Dataset(
                name="OCHA Financial Tracking Service",
                source="UN Office for the Coordination of Humanitarian Affairs",
                url="https://fts.unocha.org/",
                description="Comprehensive database of humanitarian funding flows and requirements",
                data_types=["financial", "funding", "geographic"],
                ethical_concerns=["financial_transparency", "donor_privacy"],
                suitability_score=0.7
            ),
            Dataset(
                name="ACAPS Crisis Analysis",
                source="Assessment Capacities Project",
                url="https://www.acaps.org/",
                description="Crisis analysis and humanitarian needs assessment data",
                data_types=["crisis_analysis", "needs_assessment", "severity"],
                ethical_concerns=["crisis_sensitivity", "population_privacy"],
                suitability_score=0.8
            ),
            Dataset(
                name="IOM Displacement Tracking Matrix",
                source="International Organization for Migration",
                url="https://dtm.iom.int/",
                description="Real-time population displacement and mobility tracking data",
                data_types=["displacement", "mobility", "demographic"],
                ethical_concerns=["displacement_sensitivity", "location_privacy"],
                suitability_score=0.75
            ),
            Dataset(
                name="UNESCO Education Statistics",
                source="United Nations Educational, Scientific and Cultural Organization",
                url="http://data.uis.unesco.org/",
                description="Global education statistics including literacy rates, school enrollment, and educational outcomes",
                data_types=["education", "statistical", "demographic"],
                ethical_concerns=["student_privacy", "institutional_consent"],
                suitability_score=0.8
            ),
            Dataset(
                name="UNICEF Multiple Indicator Cluster Surveys (MICS)",
                source="United Nations Children's Fund",
                url="https://mics.unicef.org/",
                description="Household surveys providing data on children and women's health, education, and protection",
                data_types=["health", "education", "protection", "household_survey"],
                ethical_concerns=["child_protection", "household_privacy", "sensitive_data"],
                suitability_score=0.85
            )
        ]
        
        # Filter datasets based on search query relevance
        query_lower = search_query.lower()
        relevant_datasets = []
        
        for dataset in all_datasets:
            # Check if any key terms from the query match the dataset
            dataset_text = f"{dataset.name} {dataset.description}".lower()
            
            # Simple relevance scoring
            relevance = 0
            for word in query_lower.split():
                if len(word) > 3 and word in dataset_text:
                    relevance += 1
            
            # Include if there's some relevance or if it's a general query
            if relevance > 0 or len(query_lower.split()) < 3:
                relevant_datasets.append(dataset)
        
        # If no specific matches, return a subset of most generally useful datasets
        if not relevant_datasets:
            relevant_datasets = all_datasets[:4]
        
        logger.info(f"Found {len(relevant_datasets)} relevant UN datasets")
        return relevant_datasets