import asyncio
import re
from typing import List, Dict, Any, Optional
from models.project import Dataset
from config.settings import settings
from utils.session_manager import session_manager
import logging

logger = logging.getLogger(__name__)

class HumanitarianDataSources:
    """Humanitarian data source connector optimized for practical dataset discovery from HDX and verified humanitarian repositories"""
    
    def __init__(self):
        self.max_results_per_source = settings.max_results_per_source
        self.session_manager = session_manager
    
    async def _search_with_retry(
        self, 
        search_func, 
        query: str, 
        limit: int, 
        source_name: str
    ) -> List[Dict[str, Any]]:
        """Execute search with retry logic for reliable data source access"""
        for attempt in range(settings.request_retry_count + 1):
            try:
                result = await search_func(query, limit)
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

    async def _search_hdx_optimized(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Optimized HDX search combining basic and format-specific strategies"""
        results = []
        
        basic_results = await self._search_hdx_basic(query, limit)
        results.extend(basic_results)
        
        if len(basic_results) < limit // 2:
            format_results = await self._search_hdx_with_formats(query, limit // 2)
            results.extend(format_results)
        
        unique_results = self._deduplicate_hdx_results(results)
        suitable_results = [r for r in unique_results if self._is_practically_suitable(r)]
        
        return suitable_results[:limit]

    async def _search_hdx_basic(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Execute basic HDX API search with humanitarian focus"""
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
                            results = self._process_hdx_packages(packages, "basic_search")
                        
            except Exception as e:
                logger.warning(f"HDX basic search failed for '{query}': {e}")
        
        return results

    async def _search_hdx_with_formats(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """HDX search with structured data format filtering"""
        results = []
        
        async with session_manager.get_session() as session:
            try:
                format_query = f"{query} (format:CSV OR format:JSON OR format:XLSX OR microdata OR survey OR indicators)"
                
                params = {
                    "q": format_query,
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
                            results = self._process_hdx_packages(packages, "format_search")
                        
            except Exception as e:
                logger.warning(f"HDX format search failed for '{query}': {e}")
        
        return results

    def _process_hdx_packages(self, packages: List[Dict], search_type: str) -> List[Dict[str, Any]]:
        """Process HDX package data into standardized dataset format"""
        results = []
        
        for package in packages:
            title = package.get("title", "").strip()
            notes = package.get("notes", "").strip()
            name = package.get("name", "")
            resources = package.get("resources", [])
            
            if not (title and len(notes) > 15 and name):
                continue
            
            dataset_url = f"https://data.humdata.org/dataset/{name}"
            
            result = {
                "title": title,
                "description": notes[:500] + "..." if len(notes) > 500 else notes,
                "source": "Humanitarian Data Exchange (HDX)",
                "source_url": dataset_url,
                "type": "hdx_dataset",
                "search_type": search_type,
                "organization": package.get("organization", {}).get("title", ""),
                "tags": [tag.get("name", "") for tag in package.get("tags", [])],
                "last_modified": package.get("metadata_modified", ""),
                "num_resources": package.get("num_resources", 0),
                "formats": [r.get("format", "") for r in resources],
                "resources": resources
            }
            
            results.append(result)
        
        return results

    def _is_practically_suitable(self, result: Dict[str, Any]) -> bool:
        """Assess practical suitability for humanitarian AI/ML projects"""
        
        title = result.get("title", "").lower()
        description = result.get("description", "").lower()
        formats = [fmt.lower() for fmt in result.get("formats", [])]
        tags = [tag.lower() for tag in result.get("tags", [])]
        
        structured_formats = ['csv', 'json', 'xlsx', 'xml', 'geojson']
        has_structured_format = any(fmt in structured_formats for fmt in formats)
        
        data_content_indicators = [
            'data', 'statistics', 'indicators', 'survey', 'assessment', 'monitoring',
            'records', 'measurements', 'observations', 'census', 'register'
        ]
        
        combined_content = f"{title} {description} {' '.join(tags)}"
        has_data_content = any(indicator in combined_content for indicator in data_content_indicators)
        
        exclude_types = [
            'report only', 'methodology document', 'guidelines document', 
            'training materials', 'policy brief', 'infographic', 'presentation'
        ]
        
        is_excluded = any(exclude_type in description for exclude_type in exclude_types)
        
        humanitarian_context = [
            'humanitarian', 'crisis', 'emergency', 'development', 'aid',
            'refugee', 'displacement', 'health', 'education', 'food security',
            'protection', 'shelter', 'wash', 'nutrition', 'livelihood'
        ]
        
        has_humanitarian_context = any(context in combined_content for context in humanitarian_context)
        
        suitability_score = sum([
            has_structured_format or has_data_content,
            not is_excluded,
            has_humanitarian_context
        ])
        
        return suitability_score >= 2

    def _deduplicate_hdx_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results from HDX searches"""
        seen_titles = set()
        seen_urls = set()
        unique_results = []
        
        for result in results:
            normalized_title = self._normalize_title(result.get("title", ""))
            url = result.get("source_url", "")
            
            is_duplicate = (normalized_title in seen_titles) or (url and url in seen_urls)
            
            if not is_duplicate:
                seen_titles.add(normalized_title)
                if url:
                    seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results

    def get_verified_humanitarian_datasets(self, search_query: str, problem_domain: str = "") -> List[Dataset]:
        """Retrieve curated verified humanitarian datasets with relevance filtering"""
        logger.info(f"Accessing verified datasets - domain: {problem_domain}, query: {search_query}")
        
        domain_datasets = {
            "food_security": [
                Dataset(
                    name="WFP Global Food Prices Database",
                    source="World Food Programme", 
                    url="https://data.humdata.org/dataset/global-wfp-food-prices",
                    description="Comprehensive food price time series data covering 98 countries and 3,000 markets with structured CSV format suitable for price prediction, market analysis, and food security modeling",
                    data_types=["food_security", "economic", "market_data", "time_series"],
                    ethical_concerns=["market_sensitivity", "economic_data"],
                    suitability_score=0.9
                ),
                Dataset(
                    name="FAO Food Security Indicators",
                    source="Food and Agriculture Organization",
                    url="https://data.humdata.org/dataset/fao-food-security-indicators",
                    description="Country-level food security indicators including undernourishment prevalence, food production indices, and dietary diversity metrics in structured format for food security analysis",
                    data_types=["food_security", "statistical", "indicators", "demographic"],
                    ethical_concerns=["food_access_privacy", "nutritional_data"],
                    suitability_score=0.87
                )
            ],
            "migration_displacement": [
                Dataset(
                    name="IOM Displacement Tracking Matrix Global Data",
                    source="International Organization for Migration",
                    url="https://data.humdata.org/dataset/global-iom-dtm-from-api",
                    description="Structured displacement tracking data with demographic breakdowns, administrative level aggregations, and temporal patterns suitable for displacement prediction and migration flow modeling",
                    data_types=["displacement", "mobility", "demographic", "geospatial"],
                    ethical_concerns=["displacement_sensitivity", "location_privacy"],
                    suitability_score=0.85
                ),
                Dataset(
                    name="UNHCR Refugee Population Statistics",
                    source="UN High Commissioner for Refugees",
                    url="https://data.humdata.org/dataset/unhcr-refugee-statistics",
                    description="Global refugee population statistics with origin and asylum country breakdowns, temporal trends, and demographic characteristics suitable for refugee flow analysis",
                    data_types=["migration", "refugee", "demographic", "time_series"],
                    ethical_concerns=["refugee_privacy", "country_sensitivity"],
                    suitability_score=0.89
                ),
                Dataset(
                    name="Meta International Migration Flows",
                    source="Meta Data for Good",
                    url="https://dataforgood.facebook.com/dfg/international-migration-flows",
                    description="Monthly migration flows covering country-to-country migration from 2019-2022, available for over 180 countries with structured data suitable for migration pattern analysis and prediction modeling",
                    data_types=["migration", "mobility", "international", "time_series"],
                    ethical_concerns=["migration_privacy", "data_aggregation", "cross_border_sensitivity"],
                    suitability_score=0.88
                )
            ],
            "education": [
                Dataset(
                    name="UNICEF Multiple Indicator Cluster Surveys (MICS)",
                    source="United Nations Children's Fund",
                    url="https://mics.unicef.org/surveys",
                    description="Standardized household survey microdata on children and women's education, health, and protection with structured indicators and cross-country comparability suitable for predictive modeling",
                    data_types=["education", "health", "protection", "household_survey", "demographic"],
                    ethical_concerns=["child_protection", "household_privacy", "sensitive_data"],
                    suitability_score=0.9
                ),
                Dataset(
                    name="UNESCO Institute for Statistics Education Data",
                    source="UNESCO Institute for Statistics",
                    url="https://data.humdata.org/dataset/unesco-education-statistics",
                    description="Global education statistics including enrollment rates, literacy levels, completion rates, and education expenditure with country-level disaggregation suitable for education outcome modeling",
                    data_types=["education", "statistical", "indicators", "demographic"],
                    ethical_concerns=["education_privacy", "institutional_data"],
                    suitability_score=0.86
                )
            ],
            "health": [
                Dataset(
                    name="WHO Global Health Observatory Data",
                    source="World Health Organization",
                    url="https://data.humdata.org/dataset/who-global-health-observatory",
                    description="Comprehensive health indicators including disease prevalence, mortality rates, health service coverage, and risk factors with country-level data suitable for health outcome prediction",
                    data_types=["health", "epidemiological", "statistical", "indicators"],
                    ethical_concerns=["health_privacy", "disease_surveillance"],
                    suitability_score=0.92
                ),
                Dataset(
                    name="UNICEF MICS Health Module",
                    source="United Nations Children's Fund",
                    url="https://mics.unicef.org/surveys",
                    description="Comprehensive health indicators from household surveys including child mortality, maternal health, immunization, and nutrition data with standardized formats for health outcome prediction",
                    data_types=["health", "demographic", "household_survey", "statistical"],
                    ethical_concerns=["health_privacy", "child_protection", "sensitive_data"],
                    suitability_score=0.9
                ),
                Dataset(
                    name="Meta High Resolution Population Density Maps",
                    source="Meta Data for Good",
                    url="https://dataforgood.facebook.com/dfg/tools/high-resolution-population-density-maps",
                    description="High-resolution population density maps built using satellite imagery and census data, publicly available for 160+ countries, suitable for health access modeling and epidemiological analysis",
                    data_types=["population", "demographic", "geospatial", "satellite_imagery"],
                    ethical_concerns=["location_privacy", "demographic_sensitivity"],
                    suitability_score=0.85
                )
            ],
            "protection": [
                Dataset(
                    name="UNICEF MICS Child Protection Indicators",
                    source="United Nations Children's Fund",
                    url="https://mics.unicef.org/surveys",
                    description="Structured data on child protection, violence indicators, birth registration, and child discipline from household surveys suitable for protection risk modeling and analysis",
                    data_types=["protection", "child_welfare", "household_survey", "demographic"],
                    ethical_concerns=["child_protection", "violence_data", "sensitive_data"],
                    suitability_score=0.85
                ),
                Dataset(
                    name="ACLED Conflict and Protest Data",
                    source="Armed Conflict Location & Event Data Project",
                    url="https://data.humdata.org/dataset/acled-conflict-data",
                    description="Real-time conflict and protest event data with geographic coordinates, event types, and fatality information suitable for conflict prediction and security analysis",
                    data_types=["conflict", "security", "geospatial", "time_series"],
                    ethical_concerns=["conflict_sensitivity", "security_data", "location_risk"],
                    suitability_score=0.83
                )
            ],
            "disaster_response": [
                Dataset(
                    name="EM-DAT International Disaster Database",
                    source="Centre for Research on the Epidemiology of Disasters",
                    url="https://data.humdata.org/dataset/emdat-disaster-data",
                    description="Global disaster database with event details, impact metrics, and economic losses suitable for disaster risk modeling and impact prediction",
                    data_types=["disaster", "economic", "time_series", "statistical"],
                    ethical_concerns=["disaster_sensitivity", "economic_impact"],
                    suitability_score=0.88
                ),
                Dataset(
                    name="Meta Population During Crisis",
                    source="Meta Data for Good",
                    url="https://dataforgood.facebook.com/dfg/tools/facebook-population-maps",
                    description="Population change data during crisis events comparing pre-crisis baselines to crisis periods, suitable for disaster impact assessment and response planning models",
                    data_types=["population", "crisis_response", "demographic", "time_series"],
                    ethical_concerns=["crisis_sensitivity", "location_privacy", "emergency_data"],
                    suitability_score=0.83
                )
            ],
            "livelihoods": [
                Dataset(
                    name="World Bank Poverty and Inequality Data",
                    source="World Bank",
                    url="https://data.humdata.org/dataset/world-bank-poverty-data",
                    description="Global poverty indicators including poverty rates, inequality measures, and economic mobility data with country and regional disaggregation for economic modeling",
                    data_types=["economic", "poverty", "statistical", "indicators"],
                    ethical_concerns=["economic_privacy", "poverty_sensitivity"],
                    suitability_score=0.89
                ),
                Dataset(
                    name="Meta Relative Wealth Index",
                    source="Meta Data for Good",
                    url="https://dataforgood.facebook.com/dfg/tools/relative-wealth-index",
                    description="Relative standard of living predictions using privacy-protecting connectivity data, satellite imagery, and other novel data sources, suitable for poverty mapping and economic modeling",
                    data_types=["economic", "wealth", "demographic", "satellite_imagery"],
                    ethical_concerns=["economic_privacy", "wealth_sensitivity", "algorithmic_bias"],
                    suitability_score=0.86
                )
            ],
            "water_sanitation": [
                Dataset(
                    name="WHO/UNICEF Joint Monitoring Programme",
                    source="World Health Organization / UNICEF",
                    url="https://data.humdata.org/dataset/who-unicef-wash-data",
                    description="Global monitoring data on water, sanitation, and hygiene access with household and institutional coverage indicators suitable for WASH access modeling",
                    data_types=["water_sanitation", "health", "household_survey", "indicators"],
                    ethical_concerns=["health_privacy", "household_access"],
                    suitability_score=0.87
                )
            ],
            "logistics_supply": [
                Dataset(
                    name="WFP Logistics Operational Data",
                    source="World Food Programme",
                    url="https://data.humdata.org/dataset/wfp-logistics-data",
                    description="Logistics performance data including transportation costs, delivery times, and supply chain efficiency metrics suitable for supply chain optimization modeling",
                    data_types=["logistics", "transportation", "operational", "time_series"],
                    ethical_concerns=["operational_security", "supply_chain_sensitivity"],
                    suitability_score=0.82
                )
            ],
            "shelter_housing": [
                Dataset(
                    name="Global Urban Areas Dataset",
                    source="Meta Data for Good",
                    url="https://dataforgood.facebook.com/dfg/tools/globalurbanareas",
                    description="High resolution dataset of 37,000 polygonal urban areas around the globe providing detailed coverage for urban planning and housing analysis suitable for settlement modeling",
                    data_types=["urban_planning", "geospatial", "settlement", "infrastructure"],
                    ethical_concerns=["location_privacy", "urban_planning_sensitivity"],
                    suitability_score=0.77
                )
            ]
        }
        
        if problem_domain and problem_domain in domain_datasets:
            relevant_datasets = domain_datasets[problem_domain].copy()
        elif problem_domain:
            similar_domains = self._get_similar_domains(problem_domain)
            relevant_datasets = []
            for similar_domain in similar_domains:
                if similar_domain in domain_datasets:
                    relevant_datasets.extend(domain_datasets[similar_domain][:2])
        else:
            relevant_datasets = []
            priority_domains = ["health", "education", "protection", "food_security"]
            for domain in priority_domains:
                if domain in domain_datasets:
                    relevant_datasets.extend(domain_datasets[domain][:1])
        
        filtered_datasets = self._filter_by_practical_relevance(relevant_datasets, search_query, problem_domain)
        
        filtered_datasets.sort(key=lambda x: (
            -(x.suitability_score or 0),
            x.source,
            x.name
        ))
        
        logger.info(f"Retrieved {len(filtered_datasets)} relevant verified datasets for {problem_domain}")
        return filtered_datasets

    def _get_similar_domains(self, domain: str) -> List[str]:
        """Map domains to similar humanitarian sectors for broader dataset discovery"""
        domain_similarity = {
            "emergency_response": ["disaster_response", "health", "logistics_supply"],
            "community_development": ["livelihoods", "education", "health", "water_sanitation"],
            "refugee_support": ["migration_displacement", "protection", "shelter_housing"],
            "child_welfare": ["protection", "education", "health"],
            "women_empowerment": ["protection", "health", "livelihoods", "education"],
            "urban_development": ["shelter_housing", "water_sanitation", "livelihoods"],
            "rural_development": ["food_security", "water_sanitation", "livelihoods"]
        }
        
        return domain_similarity.get(domain, ["health", "education", "protection"])

    def _filter_by_practical_relevance(self, datasets: List[Dataset], search_query: str, problem_domain: str) -> List[Dataset]:
        """Filter datasets based on practical relevance to search context"""
        if not datasets:
            return []
        
        query_terms = search_query.lower().split() if search_query else []
        domain_terms = problem_domain.lower().split("_") if problem_domain else []
        all_terms = query_terms + domain_terms
        
        relevant_datasets = []
        
        for dataset in datasets:
            dataset_content = f"{dataset.name} {dataset.description}".lower()
            
            relevance_score = 0
            for term in all_terms:
                if len(term) > 2 and term in dataset_content:
                    relevance_score += 1
            
            if relevance_score > 0 or len(all_terms) == 0:
                if hasattr(dataset, 'suitability_score') and dataset.suitability_score:
                    adjusted_score = min(dataset.suitability_score + (relevance_score * 0.05), 1.0)
                    dataset.suitability_score = adjusted_score
                relevant_datasets.append(dataset)
        
        return relevant_datasets

    def _normalize_title(self, title: str) -> str:
        """Normalize dataset titles for consistent comparison"""
        if not title:
            return ""
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        return re.sub(r'\s+', ' ', normalized).strip()

    async def get_hdx_datasets(self, search_query: str) -> List[Dataset]:
        """Retrieve datasets from HDX using optimized search strategy"""
        logger.info(f"Executing HDX search with query: {search_query}")
        
        if not search_query.strip():
            logger.warning("Empty search query provided")
            return []
        
        results = await self._search_hdx_optimized(search_query, 15)
        
        datasets = []
        for result in results:
            if result.get("type") == "hdx_dataset":
                base_score = 0.75
                
                if result.get("last_modified"):
                    try:
                        from datetime import datetime
                        last_modified = datetime.fromisoformat(result["last_modified"].replace('Z', '+00:00'))
                        days_old = (datetime.now().replace(tzinfo=last_modified.tzinfo) - last_modified).days
                        if days_old < 365:
                            base_score += 0.1
                    except:
                        pass
                
                num_resources = result.get("num_resources", 0)
                if num_resources > 2:
                    base_score += 0.05
                
                dataset = Dataset(
                    name=result.get("title", "Unknown Dataset"),
                    source="Humanitarian Data Exchange (HDX)",
                    url=result.get("source_url", ""),
                    description=result.get("description", ""),
                    data_types=self._extract_practical_data_types(result),
                    ethical_concerns=["data_privacy", "beneficiary_consent", "responsible_data_use"],
                    suitability_score=min(base_score, 1.0)
                )
                datasets.append(dataset)
        
        logger.info(f"Retrieved {len(datasets)} practical datasets from HDX")
        return datasets

    def _extract_practical_data_types(self, hdx_result: Dict) -> List[str]:
        """Extract practical data types from HDX metadata for humanitarian context"""
        data_types = ["humanitarian_data"]
        
        tags = hdx_result.get("tags", [])
        description = hdx_result.get("description", "").lower()
        title = hdx_result.get("title", "").lower()
        organization = hdx_result.get("organization", "").lower()
        
        combined_text = f"{' '.join(tags)} {description} {title} {organization}".lower()
        
        practical_type_mapping = {
            "demographic": ["population", "demographic", "census", "household"],
            "health": ["health", "medical", "disease", "mortality", "nutrition"],
            "education": ["education", "school", "literacy", "learning"],
            "food_security": ["food", "nutrition", "agriculture", "crop", "hunger"],
            "water_sanitation": ["water", "sanitation", "wash", "hygiene"],
            "disaster_response": ["disaster", "emergency", "crisis", "hazard"],
            "migration": ["refugee", "migration", "displacement", "asylum"],
            "protection": ["protection", "violence", "safety", "security"],
            "economic": ["economic", "livelihood", "employment", "poverty"],
            "geospatial": ["geographic", "geospatial", "coordinates", "mapping"],
            "time_series": ["time series", "temporal", "trend", "annual"],
            "survey": ["survey", "questionnaire", "assessment", "monitoring"]
        }
        
        for data_type, keywords in practical_type_mapping.items():
            if any(keyword in combined_text for keyword in keywords):
                data_types.append(data_type)
        
        formats = hdx_result.get("formats", [])
        if any(fmt.lower() in ['csv', 'json', 'xlsx'] for fmt in formats):
            data_types.append("structured")
        
        return sorted(list(set(data_types)))