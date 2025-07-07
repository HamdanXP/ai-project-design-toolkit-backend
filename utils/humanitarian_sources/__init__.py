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
        self.session_manager = session_manager
    
    async def _search_with_retry(
        self, 
        search_func, 
        query: str, 
        domain: str, 
        limit: int, 
        source_name: str
    ) -> List[Dict[str, Any]]:
        """Execute search with retry logic and deterministic ordering"""
        for attempt in range(settings.request_retry_count + 1):
            try:
                result = await search_func(query, domain, limit)
                # Sort results deterministically for consistency
                if result:
                    result.sort(key=lambda x: (x.get("title", ""), x.get("source", "")))
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
        """Search arXiv with deterministic results"""
        results = []
        
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
                                
                                # Avoid duplicates by checking normalized titles
                                normalized_title = self._normalize_title(result["title"])
                                if not any(self._normalize_title(existing.get("title", "")) == normalized_title for existing in results):
                                    results.append(result)
                                        
                    else:
                        logger.warning(f"arXiv returned status {response.status} for query: {query}")
                            
            except Exception as e:
                logger.warning(f"arXiv search failed: {query} - {e}")
        
        # Sort deterministically before limiting
        results.sort(key=lambda x: (x.get("title", ""), x.get("published_date", "")))
        return results[:limit]

    async def _search_reliefweb_safe(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search ReliefWeb with deterministic results"""
        results = []
        
        base_url = "https://api.reliefweb.int/v1"
        
        async with session_manager.get_session() as session:
            try:
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
                                
                                # Avoid duplicates using normalized titles
                                normalized_title = self._normalize_title(result["title"])
                                if not any(self._normalize_title(existing.get("title", "")) == normalized_title for existing in results):
                                    results.append(result)
                                    
                    else:
                        logger.warning(f"ReliefWeb returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"ReliefWeb search failed for '{query}': {e}")
        
        # Sort deterministically before limiting
        results.sort(key=lambda x: (x.get("title", ""), x.get("organization", "")))
        return results[:limit]

    async def _search_hdx_safe(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search HDX with deterministic results"""
        results = []
        
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
                                    
                                    # Avoid duplicates using normalized titles
                                    normalized_title = self._normalize_title(result["title"])
                                    if not any(self._normalize_title(existing.get("title", "")) == normalized_title for existing in results):
                                        results.append(result)
                        else:
                            logger.warning("HDX API returned success=false")
                    else:
                        logger.warning(f"HDX returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"HDX search failed for '{query}': {e}")
        
        # Sort deterministically before limiting
        results.sort(key=lambda x: (x.get("title", ""), x.get("organization", "")))
        return results[:limit]

    async def _search_semantic_scholar_safe(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search Semantic Scholar with deterministic results"""
        results = []
        
        headers = {}
        if settings.semantic_scholar_api_key:
            headers['x-api-key'] = settings.semantic_scholar_api_key
        
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
                                
                                # Avoid duplicates using normalized titles
                                normalized_title = self._normalize_title(result["title"])
                                if not any(self._normalize_title(existing.get("title", "")) == normalized_title for existing in results):
                                    results.append(result)
                                    
                    elif response.status == 429:
                        logger.warning("Semantic Scholar rate limit exceeded")
                        await asyncio.sleep(1)
                    else:
                        logger.warning(f"Semantic Scholar returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
        
        # Sort deterministically before limiting
        results.sort(key=lambda x: (x.get("title", ""), x.get("year", 0) or 0))
        return results[:limit]

    def _normalize_title(self, title: str) -> str:
        """Normalize title for consistent deduplication"""
        if not title:
            return ""
        # Remove punctuation, extra spaces, convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _is_potentially_relevant(self, paper: Dict, domain: str) -> bool:
        """More lenient relevance check with consistent evaluation"""
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
        """Get domain-specific search terms with consistent ordering"""
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
        
        terms = domain_mapping.get(domain, [domain.replace('_', ' ')])
        return sorted(terms)  # Sort for consistency
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response with consistent ordering"""
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
                
                # Extract authors (sorted for consistency)
                authors = []
                for author in entry.findall('author'):
                    name_elem = author.find('name')
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                paper['authors'] = sorted(authors)  # Sort for consistency
                
                # Extract published date
                published_elem = entry.find('published')
                if published_elem is not None:
                    paper['published'] = published_elem.text.strip()
                
                # Extract categories (sorted for consistency)
                categories = []
                for category in entry.findall('category'):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                paper['categories'] = sorted(categories)  # Sort for consistency
                
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
                    
                    # Extract authors (sorted for consistency)
                    authors = []
                    for author in entry.findall('atom:author', namespace):
                        name_elem = author.find('atom:name', namespace)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    paper['authors'] = sorted(authors)
                    
                    # Extract published date
                    published_elem = entry.find('atom:published', namespace)
                    if published_elem is not None:
                        paper['published'] = published_elem.text.strip()
                    
                    # Extract categories (sorted for consistency)
                    categories = []
                    for category in entry.findall('atom:category', namespace):
                        term = category.get('term')
                        if term:
                            categories.append(term)
                    paper['categories'] = sorted(categories)
                    
                    # Only include papers with title and summary
                    if paper.get('title') and paper.get('summary'):
                        papers.append(paper)
                        
            except Exception as e2:
                logger.error(f"Failed alternative ArXiv XML parsing: {e2}")
        except Exception as e:
            logger.error(f"Unexpected error parsing ArXiv XML: {e}")
        
        # Sort papers for consistent output
        papers.sort(key=lambda x: (x.get('title', ''), x.get('published', '')))
        
        logger.info(f"Parsed {len(papers)} papers from ArXiv XML")
        return papers
    
    async def get_reliefweb_datasets(self, search_query: str) -> List[Dataset]:
        """Get datasets from ReliefWeb with consistent ordering"""
        logger.info(f"Searching ReliefWeb for datasets with query: {search_query}")
        
        # Extract key terms from search query for better matching
        key_terms = [term.strip() for term in search_query.split() if len(term.strip()) > 3][:5]
        search_term = " ".join(sorted(key_terms))  # Sort for consistency
        
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
        
        # Sort datasets for consistent output
        datasets.sort(key=lambda x: (x.name, x.source))
        
        logger.info(f"Found {len(datasets)} datasets from ReliefWeb")
        return datasets
    
    async def get_hdx_datasets(self, search_query: str) -> List[Dataset]:
        """Get datasets from HDX with consistent ordering"""
        logger.info(f"Searching HDX for datasets with query: {search_query}")
        
        # Extract key terms from search query for better matching
        key_terms = [term.strip() for term in search_query.split() if len(term.strip()) > 3][:5]
        search_term = " ".join(sorted(key_terms))  # Sort for consistency
        
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
                
                # Add additional HDX-specific metadata consistently
                metadata_parts = []
                if result.get("organization"):
                    metadata_parts.append(f"Organization: {result['organization']}")
                if result.get("tags"):
                    sorted_tags = sorted(result['tags'][:3])  # Sort for consistency
                    metadata_parts.append(f"Tags: {', '.join(sorted_tags)}")
                if result.get("num_resources"):
                    metadata_parts.append(f"Resources: {result['num_resources']} files")
                
                if metadata_parts:
                    dataset.description += " | " + " | ".join(metadata_parts)
                
                datasets.append(dataset)
        
        # Sort datasets for consistent output
        datasets.sort(key=lambda x: (-(x.suitability_score or 0), x.name))
        
        logger.info(f"Found {len(datasets)} datasets from HDX")
        return datasets

    def _extract_data_types_from_hdx(self, hdx_result: Dict) -> List[str]:
        """Extract data types from HDX dataset metadata with consistent ordering"""
        data_types = ["humanitarian_data"]
        
        # Analyze tags to determine data types
        tags = hdx_result.get("tags", [])
        tag_text = " ".join(sorted(tags)).lower()  # Sort for consistency
        
        type_mapping = {
            "demographic": ["population", "demographic", "census"],
            "health": ["health", "medical", "disease"],
            "education": ["education", "school", "literacy"],
            "food_security": ["food", "nutrition", "agriculture"],
            "water_sanitation": ["water", "sanitation", "wash"],
            "disaster_response": ["disaster", "emergency", "crisis"],
            "migration": ["refugee", "migration", "displacement"],
            "protection": ["protection", "violence", "safety"],
            "economic": ["economic", "livelihood", "employment"],
            "geospatial": ["geographic", "geospatial", "location"]
        }
        
        for data_type, keywords in type_mapping.items():
            if any(keyword in tag_text for keyword in keywords):
                data_types.append(data_type)
        
        # Analyze title and description for additional context
        title_desc = f"{hdx_result.get('title', '')} {hdx_result.get('description', '')}".lower()
        
        additional_types = {
            "survey": ["survey"],
            "assessment": ["assessment"],
            "monitoring": ["monitoring"],
            "administrative": ["admin", "boundary", "administrative"]
        }
        
        for data_type, keywords in additional_types.items():
            if any(keyword in title_desc for keyword in keywords):
                data_types.append(data_type)
        
        return sorted(list(set(data_types)))  # Remove duplicates and sort
    
    async def get_un_datasets(self, search_query: str) -> List[Dataset]:
        """Get standard UN datasets with consistent filtering and ordering"""
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
        
        # Filter datasets based on search query relevance with consistent scoring
        query_lower = search_query.lower()
        query_words = sorted(query_lower.split())  # Sort for consistency
        relevant_datasets = []
        
        for dataset in all_datasets:
            # Check if any key terms from the query match the dataset
            dataset_text = f"{dataset.name} {dataset.description}".lower()
            
            # Consistent relevance scoring
            relevance = 0
            for word in query_words:
                if len(word) > 3 and word in dataset_text:
                    relevance += 1
            
            # Add relevance score for consistent ordering
            dataset.relevance_score = relevance
            
            # Include if there's some relevance or if it's a general query
            if relevance > 0 or len(query_words) < 3:
                relevant_datasets.append(dataset)
        
        # Sort by relevance score, then suitability score, then name for consistency
        relevant_datasets.sort(key=lambda x: (
            -(getattr(x, 'relevance_score', 0)),
            -(x.suitability_score or 0),
            x.name
        ))
        
        # If no specific matches, return a subset of most generally useful datasets
        if not relevant_datasets:
            relevant_datasets = sorted(all_datasets, key=lambda x: (-(x.suitability_score or 0), x.name))[:4]
        
        logger.info(f"Found {len(relevant_datasets)} relevant UN datasets")
        return relevant_datasets