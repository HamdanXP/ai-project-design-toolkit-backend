from typing import List, Dict, Any
import logging
from models.project import Dataset
from utils.humanitarian_sources import HumanitarianDataSources
from core.llm_service import llm_service

logger = logging.getLogger(__name__)

class DatasetDiscoveryService:
    """Enhanced service for discovering datasets with improved query generation"""
   
    def __init__(self):
        self.data_sources = HumanitarianDataSources()
   
    async def recommend_datasets(
        self,
        project_description: str,
        use_case_title: str = "",
        use_case_description: str = "",
        problem_domain: str = "general_humanitarian"
    ) -> List[Dataset]:
        """
        Recommend relevant datasets using improved search query generation
        """
        
        try:
            logger.info(f"Getting datasets for domain: {problem_domain}, use case: {use_case_title}")
            
            # Generate focused search query instead of using full descriptions
            search_query = self._generate_dataset_search_query(
                project_description, use_case_title, use_case_description, problem_domain
            )
            
            logger.info(f"Generated search query: {search_query}")
            
            datasets = []
            
            # Search humanitarian data sources with focused query
            try:
                reliefweb_datasets = await self.data_sources.get_reliefweb_datasets(search_query)
                datasets.extend(reliefweb_datasets)
                logger.info(f"Found {len(reliefweb_datasets)} datasets from ReliefWeb")
            except Exception as e:
                logger.warning(f"Failed to fetch ReliefWeb datasets: {e}")
           
            try:
                un_datasets = await self.data_sources.get_un_datasets(search_query)
                datasets.extend(un_datasets)
                logger.info(f"Found {len(un_datasets)} datasets from UN sources")
            except Exception as e:
                logger.warning(f"Failed to fetch UN datasets: {e}")
                
            try:
                hdx_datasets = await self.data_sources.get_hdx_datasets(search_query)
                datasets.extend(hdx_datasets)
                logger.info(f"Found {len(hdx_datasets)} datasets from HDX")
            except Exception as e:
                logger.warning(f"Failed to fetch HDX datasets: {e}")
           
            # If no datasets found, return empty list - NO FALLBACKS
            if not datasets:
                logger.info("No datasets found from humanitarian sources")
                return []
           
            # Analyze dataset suitability for AI use
            for dataset in datasets:
                dataset.suitability_score = await self._assess_dataset_suitability(
                    dataset, project_description, use_case_title, use_case_description
                )
           
            # Sort by suitability score and return top datasets
            datasets.sort(key=lambda x: x.suitability_score or 0, reverse=True)
            
            top_datasets = datasets[:10]
            logger.info(f"Returning {len(top_datasets)} top-ranked datasets")
            return top_datasets

        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            return []

    def _generate_dataset_search_query(
        self, 
        project_description: str, 
        use_case_title: str, 
        use_case_description: str,
        problem_domain: str
    ) -> str:
        """
        Generate a focused search query for datasets using domain and key terms
        """
        
        # Start with domain-specific terms
        domain_terms = self._get_domain_dataset_terms(problem_domain)
        
        # Extract key terms from project and use case
        key_terms = []
        
        # Extract from project description (limit to key concepts)
        project_keywords = self._extract_key_terms(project_description)
        key_terms.extend(project_keywords[:3])  # Top 3 most relevant
        
        # Extract from use case if available
        if use_case_title:
            use_case_keywords = self._extract_key_terms(use_case_title)
            key_terms.extend(use_case_keywords[:2])  # Top 2 from title
        
        if use_case_description:
            description_keywords = self._extract_key_terms(use_case_description)
            key_terms.extend(description_keywords[:2])  # Top 2 from description
        
        # Combine domain terms with extracted keywords
        search_terms = domain_terms[:3]  # Top 3 domain terms
        search_terms.extend([term for term in key_terms if term not in search_terms][:3])
        
        # Create focused search query
        search_query = " ".join(search_terms)
        
        logger.info(f"Generated focused search query for {problem_domain}: {search_query}")
        return search_query

    def _get_domain_dataset_terms(self, domain: str) -> List[str]:
        """Get dataset-specific search terms for each domain"""
        domain_dataset_terms = {
            "health": [
                "health data", "medical records", "epidemiological", "disease surveillance", 
                "health indicators", "patient data", "health outcomes", "clinical data"
            ],
            "education": [
                "education data", "student performance", "learning outcomes", "school enrollment", 
                "literacy rates", "educational assessment", "academic achievement"
            ],
            "food_security": [
                "food security data", "nutrition survey", "crop yield", "food consumption", 
                "agricultural data", "harvest data", "food access", "malnutrition"
            ],
            "water_sanitation": [
                "water quality data", "sanitation coverage", "water access", "WASH indicators", 
                "water infrastructure", "hygiene practices", "water supply"
            ],
            "disaster_response": [
                "disaster data", "emergency response", "crisis indicators", "damage assessment", 
                "evacuation data", "relief distribution", "emergency needs"
            ],
            "migration_displacement": [
                "migration data", "refugee statistics", "displacement tracking", "population movement", 
                "asylum data", "border crossings", "settlement data"
            ],
            "shelter_housing": [
                "housing data", "shelter conditions", "settlement mapping", "infrastructure data", 
                "accommodation records", "construction data"
            ],
            "protection": [
                "protection data", "violence indicators", "safety metrics", "legal aid records", 
                "human rights data", "security incidents"
            ],
            "livelihoods": [
                "economic data", "employment statistics", "income data", "livelihood indicators", 
                "market data", "poverty data", "economic opportunities"
            ],
            "logistics_supply": [
                "supply chain data", "distribution records", "logistics data", "transportation data", 
                "inventory data", "delivery tracking"
            ]
        }
        
        return domain_dataset_terms.get(domain, [domain.replace('_', ' '), "humanitarian data"])

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for search queries"""
        if not text:
            return []
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'be', 'have', 'do', 'say', 'get', 'make',
            'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use',
            'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'this',
            'that', 'these', 'those', 'will', 'would', 'could', 'should', 'can', 'may',
            'might', 'must', 'shall', 'is', 'are', 'was', 'were', 'been', 'being'
        }
        
        # Split and clean text
        words = text.lower().split()
        
        # Filter meaningful terms
        key_terms = []
        for word in words:
            # Remove punctuation and filter
            clean_word = ''.join(c for c in word if c.isalnum())
            if (len(clean_word) > 3 and 
                clean_word not in stop_words and 
                not clean_word.isdigit()):
                key_terms.append(clean_word)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in key_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        return unique_terms[:10]  # Return top 10 terms

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