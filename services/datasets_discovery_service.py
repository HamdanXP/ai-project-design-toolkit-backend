from typing import List, Dict, Any
import logging
import json
import re
from config import settings
from models.project import Dataset
from utils.humanitarian_sources import HumanitarianDataSources
from core.llm_service import llm_service

logger = logging.getLogger(__name__)

class DatasetDiscoveryService:
    """LLM-driven service for discovering humanitarian datasets through systematic analysis and targeted search query generation optimized for HDX API and humanitarian data repositories"""
   
    def __init__(self):
        self.data_sources = HumanitarianDataSources()
   
    async def recommend_datasets(
        self,
        project_description: str,
        use_case_title: str = "",
        use_case_description: str = "",
        problem_domain: str = ""
    ) -> List[Dataset]:
        """Recommend datasets through systematic project analysis and targeted humanitarian data source queries"""
        
        try:
            logger.info(f"Analyzing project for dataset discovery - domain: {problem_domain}, use case: {use_case_title}")
            
            data_requirements = await self._analyze_project_data_requirements(
                project_description, use_case_title, use_case_description, problem_domain
            )
            
            if not data_requirements:
                logger.warning("Failed to analyze project data requirements")
                return []
            
            search_queries = await self._generate_targeted_humanitarian_queries(data_requirements)
            
            if not search_queries:
                logger.warning("Failed to generate targeted queries")
                return []
            
            logger.info(f"Generated {len(search_queries)} systematic search strategies")
            
            all_datasets = []
            
            try:
                verified_datasets = self.data_sources.get_verified_humanitarian_datasets(
                    search_queries[0], problem_domain
                )
                all_datasets.extend(verified_datasets)
                logger.info(f"Retrieved {len(verified_datasets)} verified datasets")
            except Exception as e:
                logger.warning(f"Failed to fetch verified datasets: {e}")
            
            if settings.enable_hdx:
                try:
                    for query in search_queries[:3]:
                        hdx_datasets = await self.data_sources.get_hdx_datasets(query)
                        all_datasets.extend(hdx_datasets)
                    logger.info(f"Completed HDX queries with {len(search_queries[:3])} targeted searches")
                except Exception as e:
                    logger.warning(f"Failed to fetch HDX datasets: {e}")
            
            if not all_datasets:
                logger.info("No datasets found from humanitarian sources")
                return []
           
            unique_datasets = self._remove_duplicates(all_datasets)
            logger.info(f"After deduplication: {len(unique_datasets)} unique datasets")
           
            if unique_datasets:
                relevant_datasets = await self._assess_dataset_alignment(
                    unique_datasets, data_requirements
                )
                
                relevant_datasets.sort(key=lambda x: x.suitability_score or 0, reverse=True)
                top_datasets = relevant_datasets[:settings.max_datasets_returned]
                logger.info(f"Returning {len(top_datasets)} aligned datasets")
                return top_datasets
            
            return []

        except Exception as e:
            logger.error(f"Failed dataset discovery process: {e}")
            return []

    async def _analyze_project_data_requirements(
        self, 
        project_description: str, 
        use_case_title: str, 
        use_case_description: str,
        problem_domain: str
    ) -> Dict[str, Any]:
        """Systematically analyze project to extract specific data requirements for humanitarian contexts"""
        
        project_context = f"Project: {project_description}"
        if use_case_title:
            project_context += f"\nUse Case: {use_case_title}"
        if use_case_description:
            project_context += f"\nDetails: {use_case_description}"
        if problem_domain:
            project_context += f"\nDomain: {problem_domain}"
        
        prompt = f"""
        Analyze this humanitarian AI project to extract specific data requirements:
        
        {project_context}
        
        Systematically identify:
        
        1. PRIMARY DATA TYPES needed (what kind of information/measurements):
           - Demographics, health indicators, geographic data, time series, etc.
        
        2. DATA COLLECTION CONTEXT (how/where data was gathered):
           - Surveys, administrative records, satellite imagery, mobile data, etc.
        
        3. HUMANITARIAN SECTORS involved:
           - Health, education, food security, protection, shelter, WASH, etc.
        
        4. GEOGRAPHIC SCOPE requirements:
           - Country-level, subnational, community-level, global, etc.
        
        5. TEMPORAL ASPECTS needed:
           - Real-time, historical trends, periodic updates, one-time assessment, etc.
        
        6. POPULATION FOCUS:
           - General population, displaced persons, children, women, elderly, etc.
        
        Return structured analysis as JSON:
        {{
            "primary_data_types": ["type1", "type2", "type3"],
            "collection_methods": ["method1", "method2"],
            "humanitarian_sectors": ["sector1", "sector2"],
            "geographic_scope": "scope_description",
            "temporal_needs": "temporal_description",
            "target_populations": ["population1", "population2"],
            "key_indicators": ["indicator1", "indicator2", "indicator3"]
        }}
        
        Focus on identifying data characteristics that would help locate relevant humanitarian datasets.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            cleaned = self._clean_json_response(response)
            requirements = json.loads(cleaned)
            
            logger.info(f"Extracted data requirements: {len(requirements.get('primary_data_types', []))} data types, {len(requirements.get('humanitarian_sectors', []))} sectors")
            return requirements
                
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            return {}

    async def _generate_targeted_humanitarian_queries(self, data_requirements: Dict[str, Any]) -> List[str]:
        """Generate HDX-optimized search queries based on systematic data requirements analysis"""
        
        if not data_requirements:
            return []
        
        requirements_summary = ""
        for key, value in data_requirements.items():
            if isinstance(value, list):
                requirements_summary += f"{key}: {', '.join(value[:3])}\n"
            else:
                requirements_summary += f"{key}: {value}\n"
        
        prompt = f"""
        Generate 4 targeted search queries for humanitarian datasets based on this systematic analysis:
        
        DATA REQUIREMENTS ANALYSIS:
        {requirements_summary}
        
        Create HDX-optimized search queries that:
        
        1. TARGET HUMANITARIAN DATA REPOSITORIES:
           - Use terminology common in humanitarian dataset titles and descriptions
           - Focus on data content rather than analysis methods
           - Include sector-specific humanitarian terms
        
        2. OPTIMIZE FOR HDX API SEARCH:
           - Use 2-5 words per query for best search performance
           - Combine data type + sector + context for specificity
           - Avoid overly technical or academic language
        
        3. ENSURE SYSTEMATIC COVERAGE:
           - Query 1: Primary data type + main humanitarian sector
           - Query 2: Collection method + geographic/population scope
           - Query 3: Key indicators + temporal aspect
           - Query 4: Alternative data perspective or secondary sector
        
        4. HUMANITARIAN DATASET FOCUS:
           - Target operational humanitarian data sources
           - Include terms used by UN agencies, NGOs, governments
           - Focus on datasets suitable for AI/ML analysis
        
        Examples of effective humanitarian queries:
        - "health facility data"
        - "displacement tracking survey"
        - "food security indicators"
        - "education enrollment statistics"
        
        Return exactly 4 queries as JSON array:
        ["targeted_query_1", "targeted_query_2", "targeted_query_3", "targeted_query_4"]
        
        Each query should be optimized for finding structured humanitarian datasets.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            cleaned = self._clean_json_response(response)
            queries = json.loads(cleaned)
            
            if isinstance(queries, list) and len(queries) >= 4:
                valid_queries = [q.strip() for q in queries[:4] if q and q.strip()]
                if len(valid_queries) >= 3:
                    logger.info(f"Generated systematic queries: {valid_queries}")
                    return valid_queries
            
            logger.warning("Invalid query generation result")
            return []
                
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return []

    async def _assess_dataset_alignment(
        self,
        datasets: List[Dataset],
        data_requirements: Dict[str, Any]
    ) -> List[Dataset]:
        """Assess how well datasets align with the systematic data requirements analysis"""
        
        if not datasets or not data_requirements:
            return datasets
        
        datasets_info = ""
        for i, dataset in enumerate(datasets[:15]):
            datasets_info += f"""
Dataset {i+1}:
Name: {dataset.name}
Source: {dataset.source}
Description: {dataset.description[:400]}
Data Types: {', '.join(dataset.data_types) if dataset.data_types else 'Not specified'}
---
"""
        
        requirements_text = ""
        for key, value in data_requirements.items():
            if isinstance(value, list) and value:
                requirements_text += f"{key}: {', '.join(value)}\n"
            elif value:
                requirements_text += f"{key}: {value}\n"
        
        prompt = f"""
        Assess dataset alignment with systematic data requirements:
        
        PROJECT DATA REQUIREMENTS:
        {requirements_text}
        
        AVAILABLE DATASETS:
        {datasets_info}
        
        For each dataset, evaluate alignment on scale 0-100 based on:
        
        1. DATA TYPE MATCH (40 points):
           - How well dataset content matches required data types
           - Relevance of variables and indicators
        
        2. HUMANITARIAN SECTOR ALIGNMENT (30 points):
           - Coverage of required humanitarian sectors
           - Operational relevance to project domain
        
        3. SCOPE AND SCALE APPROPRIATENESS (20 points):
           - Geographic scope alignment
           - Population coverage match
           - Temporal coverage suitability
        
        4. AI/ML SUITABILITY (10 points):
           - Data structure and format
           - Quality indicators from description
        
        Return assessment as JSON array:
        [
            {{"dataset_number": 1, "alignment_score": 85, "primary_strengths": "Strong demographic data coverage for target population"}},
            {{"dataset_number": 2, "alignment_score": 45, "primary_strengths": "Related sector but limited geographic scope"}},
            ...
        ]
        
        Focus on practical applicability for the identified data requirements.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            cleaned = self._clean_json_response(response)
            assessments = json.loads(cleaned)
            
            aligned_datasets = []
            for assessment in assessments:
                dataset_idx = assessment.get("dataset_number", 0) - 1
                alignment_score = assessment.get("alignment_score", 0)
                strengths = assessment.get("primary_strengths", "")
                
                if 0 <= dataset_idx < len(datasets):
                    dataset = datasets[dataset_idx]
                    dataset.suitability_score = alignment_score / 100.0
                    
                    if alignment_score >= 30:
                        aligned_datasets.append(dataset)
                        logger.debug(f"Dataset '{dataset.name}' alignment: {alignment_score}% - {strengths}")
            
            logger.info(f"Dataset alignment assessment: {len(aligned_datasets)} aligned from {len(datasets)} total")
            return aligned_datasets
            
        except Exception as e:
            logger.error(f"Dataset alignment assessment failed: {e}")
            for dataset in datasets:
                if not hasattr(dataset, 'suitability_score') or dataset.suitability_score is None:
                    dataset.suitability_score = 0.5
            return datasets

    def _remove_duplicates(self, datasets: List[Dataset]) -> List[Dataset]:
        """Remove duplicate datasets based on title and URL similarity"""
        seen_titles = set()
        seen_urls = set()
        unique_datasets = []
        
        for dataset in datasets:
            normalized_title = self._normalize_text(dataset.name)
            normalized_url = dataset.url.lower().strip() if dataset.url else ""
            
            is_similar = any(
                self._are_titles_similar(normalized_title, seen_title) 
                for seen_title in seen_titles
            )
            
            if is_similar or (normalized_url and normalized_url in seen_urls):
                continue
            
            seen_titles.add(normalized_title)
            if normalized_url:
                seen_urls.add(normalized_url)
            unique_datasets.append(dataset)
        
        return unique_datasets

    def _are_titles_similar(self, title1: str, title2: str) -> bool:
        """Determine if two dataset titles represent the same or very similar datasets"""
        if not title1 or not title2:
            return False
        
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return (intersection / union) > 0.8 if union > 0 else False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison and deduplication"""
        if not text:
            return ""
        import re
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _clean_json_response(self, response: str) -> str:
        """Extract and clean JSON content from LLM responses"""
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        start_chars = ['{', '[']
        end_chars = ['}', ']']
        
        start_idx = -1
        start_char = None
        for char in start_chars:
            idx = response.find(char)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx
                start_char = char
        
        if start_idx == -1:
            raise ValueError("No JSON found")
        
        end_char = '}' if start_char == '{' else ']'
        end_idx = response.rfind(end_char)
        
        if end_idx == -1:
            raise ValueError(f"No closing {end_char} found")
        
        return response[start_idx:end_idx + 1].strip()