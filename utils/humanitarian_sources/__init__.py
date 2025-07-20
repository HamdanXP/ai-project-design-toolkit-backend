import asyncio
import re
from typing import List, Dict, Any, Optional
from models.project import Dataset
from config.settings import settings
from utils.session_manager import session_manager
import logging

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
                                name = package.get("name", "")
                                
                                if title and len(notes) > 20 and name:
                                    dataset_url = f"https://data.humdata.org/dataset/{name}"
                                    
                                    result = {
                                        "title": title,
                                        "description": notes[:500] + "..." if len(notes) > 500 else notes,
                                        "source": "Humanitarian Data Exchange (HDX)",
                                        "source_url": dataset_url,
                                        "type": "dataset",
                                        "organization": package.get("organization", {}).get("title", ""),
                                        "tags": [tag.get("name", "") for tag in package.get("tags", [])],
                                        "last_modified": package.get("metadata_modified", ""),
                                        "num_resources": package.get("num_resources", 0)
                                    }
                                    
                                    normalized_title = self._normalize_title(result["title"])
                                    if not any(self._normalize_title(existing.get("title", "")) == normalized_title for existing in results):
                                        results.append(result)
                        else:
                            logger.warning("HDX API returned success=false")
                    else:
                        logger.warning(f"HDX returned status {response.status}")
                        
            except Exception as e:
                logger.warning(f"HDX search failed for '{query}': {e}")
        
        results.sort(key=lambda x: (x.get("title", ""), x.get("organization", "")))
        return results[:limit]

    async def _search_unhcr_api(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search UNHCR using their official Refugee Statistics API"""
        results = []
        
        async with session_manager.get_session() as session:
            try:
                params = {
                    "limit": min(limit, 100),
                    "page": 1,
                    "yearFrom": 2020,
                    "yearTo": 2024,
                    "sort": "metadata_modified desc"
                }
                
                async with session.get(
                    "https://api.unhcr.org/population/v1/asylum-applications/", 
                    params=params,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # UNHCR API returns data directly without "success" wrapper
                        if data and "items" in data:
                            applications = data.get("items", [])
                            
                            for app in applications[:limit]:
                                coo_name = app.get("coo_name", "")
                                coa_name = app.get("coa_name", "")
                                year = app.get("year", "")
                                applied = app.get("applied", 0)
                                
                                # Skip entries with missing country names (indicated by "-")
                                if coo_name and coa_name and coo_name != "-" and coa_name != "-" and applied > 0:
                                    result = {
                                        "title": f"UNHCR Asylum Applications - {coo_name} to {coa_name} ({year})",
                                        "description": f"Asylum applications from {coo_name} to {coa_name} in {year}. Applications: {applied:,}",
                                        "source": "UNHCR Refugee Statistics API",
                                        "source_url": f"https://www.unhcr.org/refugee-statistics/download/?query={coo_name.replace(' ', '%20')}",
                                        "type": "asylum_applications",
                                        "country_origin": coo_name,
                                        "country_asylum": coa_name,
                                        "year": year,
                                        "applications": applied
                                    }
                                    
                                    normalized_title = self._normalize_title(result["title"])
                                    if not any(self._normalize_title(existing.get("title", "")) == normalized_title for existing in results):
                                        results.append(result)
                        else:
                            logger.warning(f"UNHCR API returned unexpected response structure: {data}")
                            
                    else:
                        logger.warning(f"UNHCR API returned status {response.status}")
                        response_text = await response.text()
                        logger.debug(f"UNHCR response: {response_text[:200]}")
                        
            except asyncio.TimeoutError:
                logger.warning("UNHCR API request timed out")
            except Exception as e:
                logger.warning(f"UNHCR API search failed for '{query}': {e}")
        
        results.sort(key=lambda x: (x.get("year", ""), x.get("country_origin", "")))
        return results[:limit]

    async def _search_who_api(self, query: str, domain: str, limit: int) -> List[Dict[str, Any]]:
        """Search WHO Global Health Observatory using OData API with query-based filtering"""
        results = []
        
        async with session_manager.get_session() as session:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
                
                # Extract key terms from search query for WHO filtering
                query_words = [word.lower() for word in query.split() if len(word) > 3]
                
                # Create WHO-relevant filter based on actual search query
                who_health_terms = []
                humanitarian_health_mapping = {
                    'migration': ['migration', 'displacement'],
                    'refugee': ['refugee', 'displaced'],
                    'displacement': ['displacement', 'migration'],
                    'emergency': ['emergency', 'crisis'],
                    'humanitarian': ['emergency', 'crisis'],
                    'health': ['health', 'mortality'],
                    'nutrition': ['nutrition', 'malnutrition'],
                    'mortality': ['mortality', 'death'],
                    'disease': ['disease', 'epidemic']
                }
                
                # Find WHO health terms that relate to the search query
                for query_word in query_words:
                    if query_word in humanitarian_health_mapping:
                        who_health_terms.extend(humanitarian_health_mapping[query_word])
                
                # If no health-related terms found, use generic humanitarian health terms
                if not who_health_terms:
                    who_health_terms = ['emergency', 'crisis', 'humanitarian']
                
                # Remove duplicates and create filter
                who_health_terms = list(set(who_health_terms))
                filter_conditions = [f"contains(IndicatorName,'{term}')" for term in who_health_terms[:3]]  # Limit to 3 terms
                filter_query = " or ".join(filter_conditions)
                
                async with session.get(
                    f"https://ghoapi.azureedge.net/api/Indicator?$filter={filter_query}&$top={limit}", 
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            
                            indicators = data.get("value", []) if isinstance(data, dict) else data
                            
                            count = 0
                            for indicator in indicators:
                                if count >= limit:
                                    break
                                    
                                indicator_name = indicator.get("IndicatorName", "").strip()
                                indicator_code = indicator.get("IndicatorCode", "")
                                
                                if indicator_name and indicator_code:
                                    result = {
                                        "title": f"WHO Health Indicator - {indicator_name}",
                                        "description": f"Global health statistics from WHO Global Health Observatory: {indicator_name}. Indicator Code: {indicator_code}",
                                        "source": "WHO Global Health Observatory API",
                                        "source_url": f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{indicator_code}",
                                        "type": "health_statistics",
                                        "indicator_code": indicator_code,
                                        "data_type": "global_health"
                                    }
                                    
                                    normalized_title = self._normalize_title(result["title"])
                                    if not any(self._normalize_title(existing.get("title", "")) == normalized_title for existing in results):
                                        results.append(result)
                                        count += 1
                                        
                        except ValueError as e:
                            logger.warning(f"Failed to parse WHO GHO JSON response: {e}")
                            response_text = await response.text()
                            logger.debug(f"WHO response preview: {response_text[:500]}")
                            
                    elif response.status == 503:
                        logger.warning("WHO GHO API temporarily unavailable (503)")
                    elif response.status == 404:
                        logger.warning("WHO GHO API endpoint not found (404)")
                    else:
                        logger.warning(f"WHO GHO API returned status {response.status}")
                        response_text = await response.text()
                        logger.debug(f"WHO response content: {response_text[:200]}")
                        
            except asyncio.TimeoutError:
                logger.warning("WHO GHO API request timed out")
            except Exception as e:
                logger.warning(f"WHO GHO API search failed for '{query}': {e}")
        
        results.sort(key=lambda x: (x.get("title", ""), x.get("indicator_code", "")))
        return results[:limit]

    def get_verified_humanitarian_datasets(self, search_query: str, problem_domain: str = "general_humanitarian") -> List[Dataset]:
        """Get curated humanitarian datasets filtered by domain and search relevance"""
        logger.info(f"Getting verified humanitarian datasets for domain: {problem_domain}, query: {search_query}")
        
        domain_datasets = {
            "food_security": [
                Dataset(
                    name="WFP Food Price Database",
                    source="World Food Programme", 
                    url="https://data.humdata.org/dataset/wfp-food-prices",
                    description="Structured food price time series covering 76 countries and 1,500 markets with historical data suitable for price prediction and market analysis models",
                    data_types=["food_security", "economic", "market_data", "time_series"],
                    ethical_concerns=["market_sensitivity", "economic_data"],
                    suitability_score=0.85
                )
            ],
            "migration_displacement": [
                Dataset(
                    name="IOM Displacement Tracking Matrix",
                    source="International Organization for Migration",
                    url="https://dtm.iom.int/datasets",
                    description="Comprehensive displacement tracking data with demographic breakdowns, baseline assessments, and mobility flow analysis suitable for displacement prediction and migration modeling",
                    data_types=["displacement", "mobility", "demographic", "geospatial"],
                    ethical_concerns=["displacement_sensitivity", "location_privacy"],
                    suitability_score=0.9
                )
            ],
            "education": [
                Dataset(
                    name="UNICEF Multiple Indicator Cluster Surveys (MICS)",
                    source="United Nations Children's Fund",
                    url="https://mics.unicef.org/surveys",
                    description="Structured household survey data on children and women's health, education, protection with standardized indicators suitable for predictive modeling on child welfare",
                    data_types=["health", "education", "protection", "household_survey", "demographic"],
                    ethical_concerns=["child_protection", "household_privacy", "sensitive_data"],
                    suitability_score=0.9
                )
            ],
            "health": [
                Dataset(
                    name="UNICEF Multiple Indicator Cluster Surveys (MICS)",
                    source="United Nations Children's Fund",
                    url="https://mics.unicef.org/surveys",
                    description="Structured household survey data on children and women's health, education, protection with standardized indicators suitable for predictive modeling on child welfare",
                    data_types=["health", "education", "protection", "household_survey", "demographic"],
                    ethical_concerns=["child_protection", "household_privacy", "sensitive_data"],
                    suitability_score=0.9
                )
            ],
            "protection": [
                Dataset(
                    name="UNICEF Multiple Indicator Cluster Surveys (MICS)",
                    source="United Nations Children's Fund",
                    url="https://mics.unicef.org/surveys",
                    description="Structured household survey data on children and women's health, education, protection with standardized indicators suitable for predictive modeling on child welfare",
                    data_types=["health", "education", "protection", "household_survey", "demographic"],
                    ethical_concerns=["child_protection", "household_privacy", "sensitive_data"],
                    suitability_score=0.9
                )
            ]
        }
        
        # Get datasets for specific domain, return empty list if domain not supported by verified datasets
        if problem_domain in domain_datasets:
            relevant_datasets = domain_datasets[problem_domain].copy()
        else:
            # Return empty list for unsupported domains - they should get data from APIs instead
            logger.info(f"Domain '{problem_domain}' not supported by verified datasets, relying on API sources")
            relevant_datasets = []
        
        # Apply search query relevance filtering
        query_lower = search_query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 3]
        filtered_datasets = []
        
        for dataset in relevant_datasets:
            dataset_text = f"{dataset.name} {dataset.description}".lower()
            
            relevance_matches = 0
            for word in query_words:
                if word in dataset_text:
                    relevance_matches += 1
            
            base_suitability = dataset.suitability_score or 0.5
            if query_words:
                relevance_boost = (relevance_matches / len(query_words)) * 0.2
                adjusted_suitability = min(base_suitability + relevance_boost, 1.0)
            else:
                adjusted_suitability = base_suitability
            
            adjusted_dataset = Dataset(
                name=dataset.name,
                source=dataset.source,
                url=dataset.url,
                description=dataset.description,
                size_estimate=dataset.size_estimate,
                data_types=dataset.data_types,
                ethical_concerns=dataset.ethical_concerns,
                suitability_score=adjusted_suitability
            )
            
            if relevance_matches > 0 or len(query_words) < 2:
                filtered_datasets.append(adjusted_dataset)
        
        filtered_datasets.sort(key=lambda x: (
            -(x.suitability_score or 0),
            x.name
        ))
        
        logger.info(f"Found {len(filtered_datasets)} relevant verified datasets for domain {problem_domain}")
        return filtered_datasets

    def _normalize_title(self, title: str) -> str:
        """Normalize title for consistent deduplication"""
        if not title:
            return ""
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    async def get_hdx_datasets(self, search_query: str) -> List[Dataset]:
        """Get datasets from HDX"""
        logger.info(f"Searching HDX for datasets with query: {search_query}")
        
        results = await self._search_hdx_safe(search_query, "general", 15)
        
        datasets = []
        for result in results:
            if result.get("type") == "dataset":
                dataset = Dataset(
                    name=result.get("title", "Unknown"),
                    source="Humanitarian Data Exchange (HDX)",
                    url=result.get("source_url", ""),
                    description=result.get("description", ""),
                    data_types=self._extract_data_types_from_hdx(result),
                    ethical_concerns=["data_privacy", "beneficiary_consent", "responsible_data_use"],
                    suitability_score=0.8
                )
                datasets.append(dataset)
        
        logger.info(f"Found {len(datasets)} datasets from HDX")
        return datasets

    async def get_unhcr_datasets(self, search_query: str) -> List[Dataset]:
        """Get datasets from UNHCR API"""
        logger.info(f"Searching UNHCR API for datasets with query: {search_query}")
        
        results = await self._search_unhcr_api(search_query, "general", 10)
        
        datasets = []
        for result in results:
            dataset = Dataset(
                name=result.get("title", "Unknown"),
                source="UNHCR Refugee Statistics API",
                url=result.get("source_url", ""),
                description=result.get("description", ""),
                data_types=["refugee_statistics", "displacement", "demographic"],
                ethical_concerns=["displacement_sensitivity", "anonymization"],
                suitability_score=0.85
            )
            datasets.append(dataset)
        
        logger.info(f"Found {len(datasets)} datasets from UNHCR API")
        return datasets

    async def get_who_datasets(self, search_query: str) -> List[Dataset]:
        """Get datasets from WHO GHO API"""
        logger.info(f"Searching WHO GHO API for datasets with query: {search_query}")
        
        results = await self._search_who_api(search_query, "general", 8)
        
        datasets = []
        for result in results:
            dataset = Dataset(
                name=result.get("title", "Unknown"),
                source="WHO Global Health Observatory API",
                url=result.get("source_url", ""),
                description=result.get("description", ""),
                data_types=["health", "epidemiological", "statistical"],
                ethical_concerns=["health_privacy", "community_consent"],
                suitability_score=0.8
            )
            datasets.append(dataset)
        
        logger.info(f"Found {len(datasets)} datasets from WHO API")
        return datasets

    def _extract_data_types_from_hdx(self, hdx_result: Dict) -> List[str]:
        """Extract data types from HDX dataset metadata with consistent ordering"""
        data_types = ["humanitarian_data"]
        
        tags = hdx_result.get("tags", [])
        tag_text = " ".join(sorted(tags)).lower()
        
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
        
        return sorted(list(set(data_types)))