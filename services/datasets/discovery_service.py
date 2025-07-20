from typing import List, Dict, Any
import logging
from config import settings
from models.project import Dataset
from utils.humanitarian_sources import HumanitarianDataSources
from core.llm_service import llm_service

logger = logging.getLogger(__name__)

class DatasetDiscoveryService:
    """Enhanced service for discovering humanitarian datasets with verified sources"""
   
    def __init__(self):
        self.data_sources = HumanitarianDataSources()
   
    async def recommend_datasets(
        self,
        project_description: str,
        use_case_title: str = "",
        use_case_description: str = "",
        problem_domain: str = "general_humanitarian"
    ) -> List[Dataset]:
        """Recommend relevant datasets from verified humanitarian sources suitable for AI training"""
        
        try:
            logger.info(f"Getting datasets for AI training - domain: {problem_domain}, use case: {use_case_title}")
            
            search_query = self._generate_dataset_search_query(
                project_description, use_case_title, use_case_description, problem_domain
            )
            
            logger.info(f"Generated search query: {search_query}")
            
            all_datasets = []
            
            try:
                verified_datasets = self.data_sources.get_verified_humanitarian_datasets(search_query, problem_domain)
                all_datasets.extend(verified_datasets)
                logger.info(f"Found {len(verified_datasets)} verified structured datasets")
            except Exception as e:
                logger.warning(f"Failed to fetch verified datasets: {e}")
            
            if settings.enable_hdx:
                try:
                    hdx_datasets = await self.data_sources.get_hdx_datasets(search_query)
                    all_datasets.extend(hdx_datasets)
                    logger.info(f"Found {len(hdx_datasets)} datasets from HDX")
                except Exception as e:
                    logger.warning(f"Failed to fetch HDX datasets: {e}")
            
            if settings.enable_unhcr:
                try:
                    unhcr_datasets = await self.data_sources.get_unhcr_datasets(search_query)
                    all_datasets.extend(unhcr_datasets)
                    logger.info(f"Found {len(unhcr_datasets)} datasets from UNHCR")
                except Exception as e:
                    logger.warning(f"Failed to fetch UNHCR datasets: {e}")
            
            if settings.enable_who_gho:
                try:
                    who_datasets = await self.data_sources.get_who_datasets(search_query)
                    all_datasets.extend(who_datasets)
                    logger.info(f"Found {len(who_datasets)} datasets from WHO")
                except Exception as e:
                    logger.warning(f"Failed to fetch WHO datasets: {e}")
           
            if not all_datasets:
                logger.info("No structured datasets found from humanitarian sources")
                return []
           
            unique_datasets = self._remove_duplicates(all_datasets)
            logger.info(f"After deduplication: {len(unique_datasets)} unique datasets")
           
            for dataset in unique_datasets:
                dataset.suitability_score = await self._assess_dataset_suitability(
                    dataset, project_description, use_case_title, use_case_description
                )
           
            unique_datasets.sort(key=lambda x: x.suitability_score or 0, reverse=True)
            
            top_datasets = unique_datasets[:settings.max_datasets_returned]
            logger.info(f"Returning {len(top_datasets)} top-ranked structured datasets")
            return top_datasets

        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            return []

    def _remove_duplicates(self, datasets: List[Dataset]) -> List[Dataset]:
        """Remove duplicate datasets based on normalized titles and URLs"""
        seen_titles = set()
        seen_urls = set()
        unique_datasets = []
        
        for dataset in datasets:
            normalized_title = self._normalize_text(dataset.name)
            normalized_url = dataset.url.lower().strip() if dataset.url else ""
            
            if normalized_title in seen_titles or (normalized_url and normalized_url in seen_urls):
                continue
            
            seen_titles.add(normalized_title)
            if normalized_url:
                seen_urls.add(normalized_url)
            unique_datasets.append(dataset)
        
        return unique_datasets
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for deduplication"""
        if not text:
            return ""
        import re
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _generate_dataset_search_query(
        self, 
        project_description: str, 
        use_case_title: str, 
        use_case_description: str,
        problem_domain: str
    ) -> str:
        """Generate a focused search query for datasets using domain and key terms"""
        
        domain_terms = self._get_domain_dataset_terms(problem_domain)
        
        key_terms = []
        
        project_keywords = self._extract_key_terms(project_description)
        key_terms.extend(project_keywords[:3])
        
        if use_case_title:
            use_case_keywords = self._extract_key_terms(use_case_title)
            key_terms.extend(use_case_keywords[:2])
        
        if use_case_description:
            description_keywords = self._extract_key_terms(use_case_description)
            key_terms.extend(description_keywords[:2])
        
        search_terms = domain_terms[:3]
        search_terms.extend([term for term in key_terms if term not in search_terms][:3])
        
        search_query = " ".join(search_terms)
        
        logger.info(f"Generated focused search query for {problem_domain}: {search_query}")
        return search_query

    def _get_domain_dataset_terms(self, domain: str) -> List[str]:
        """Get dataset-specific search terms for each humanitarian domain"""
        domain_dataset_terms = {
            "health": [
                "health data", "medical records", "epidemiological", "disease surveillance", 
                "health indicators", "patient data", "health outcomes", "clinical data", "emergency health"
            ],
            "education": [
                "education data", "student performance", "learning outcomes", "school enrollment", 
                "literacy rates", "educational assessment", "academic achievement", "education emergency"
            ],
            "food_security": [
                "food security data", "nutrition survey", "crop yield", "food consumption", 
                "agricultural data", "harvest data", "food access", "malnutrition", "hunger data"
            ],
            "water_sanitation": [
                "water quality data", "sanitation coverage", "water access", "WASH indicators", 
                "water infrastructure", "hygiene practices", "water supply"
            ],
            "disaster_response": [
                "disaster data", "emergency response", "crisis indicators", "damage assessment", 
                "evacuation data", "relief distribution", "emergency needs", "disaster preparedness"
            ],
            "migration_displacement": [
                "migration data", "refugee statistics", "displacement tracking", "population movement", 
                "asylum data", "border crossings", "settlement data", "forced displacement"
            ],
            "shelter_housing": [
                "housing data", "shelter conditions", "settlement mapping", "infrastructure data", 
                "accommodation records", "construction data", "displacement camps"
            ],
            "protection": [
                "protection data", "violence indicators", "safety metrics", "legal aid records", 
                "human rights data", "security incidents", "child protection"
            ],
            "livelihoods": [
                "economic data", "employment statistics", "income data", "livelihood indicators", 
                "market data", "poverty data", "economic opportunities"
            ],
            "logistics_supply": [
                "supply chain data", "distribution records", "logistics data", "transportation data", 
                "inventory data", "delivery tracking", "humanitarian logistics"
            ]
        }
        
        return domain_dataset_terms.get(domain, [domain.replace('_', ' '), "humanitarian data"])

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for search queries"""
        if not text:
            return []
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'be', 'have', 'do', 'say', 'get', 'make',
            'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use',
            'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'this',
            'that', 'these', 'those', 'will', 'would', 'could', 'should', 'can', 'may',
            'might', 'must', 'shall', 'is', 'are', 'was', 'were', 'been', 'being'
        }
        
        words = text.lower().split()
        
        key_terms = []
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if (len(clean_word) > 3 and 
                clean_word not in stop_words and 
                not clean_word.isdigit()):
                key_terms.append(clean_word)
        
        unique_terms = []
        seen = set()
        for term in key_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        return unique_terms[:10]

    async def _assess_dataset_suitability(
        self,
        dataset: Dataset,
        project_description: str,
        use_case_title: str = "",
        use_case_description: str = ""
    ) -> float:
        """Assess how suitable a dataset is for the AI project"""
       
        use_case_context = f"{use_case_title} - {use_case_description}".strip(" -")
        if not use_case_context:
            use_case_context = "General AI application for humanitarian purposes"
        
        prompt = f"""
        Rate the suitability of this humanitarian dataset for an AI project (0-1 score):
       
        Project: {project_description}
        Use Case: {use_case_context}
       
        Dataset: {dataset.name}
        Source: {dataset.source}
        Description: {dataset.description}
        Data Types: {dataset.data_types}
       
        Consider for AI suitability:
        - Relevance to the AI project goals and use case
        - Data quality and completeness for machine learning
        - Ethical considerations for AI development
        - Accessibility and licensing for AI applications
        - Volume and structure suitable for training AI models
        - Alignment with humanitarian objectives
        - Reliability of the data source
       
        Return only a number between 0 and 1 (e.g., 0.7).
        """
       
        try:
            response = await llm_service.analyze_text("", prompt)
            score_str = response.strip()
            import re
            number_match = re.search(r'(\d*\.?\d+)', score_str)
            if number_match:
                score = float(number_match.group(1))
            else:
                score = 0.5
                
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Failed to assess dataset suitability: {e}")
            return 0.5