from typing import List, Dict, Any, Optional
from config import settings
from services.use_case_service import UseCaseService
from services.datasets.discovery_service import DatasetDiscoveryService
from models.project import Dataset
import logging

logger = logging.getLogger(__name__)

class ScopingService:
    """Enhanced scoping service with infrastructure-aware use case search"""
    
    def __init__(self):
        self.use_case_service = UseCaseService()
        self.dataset_discovery = DatasetDiscoveryService()
    
    async def get_educational_use_cases(
        self, 
        project_description: str, 
        problem_domain: str,
        technical_infrastructure: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Get AI use cases with educational content, filtered by infrastructure compatibility"""
        
        logger.info(f"Getting infrastructure-aware AI use cases for domain: {problem_domain}")
        if technical_infrastructure:
            logger.info(f"Infrastructure context: {technical_infrastructure}")
        
        try:
            use_cases = await self.use_case_service.search_ai_use_cases(
                project_description, problem_domain, technical_infrastructure
            )
            
            logger.info(f"Found {len(use_cases)} infrastructure-compatible AI use cases")
            return use_cases
            
        except Exception as e:
            logger.error(f"Failed to get educational use cases: {e}")
            return []
    
    async def recommend_datasets(
        self, 
        project_description: str, 
        use_case_title: str = "", 
        use_case_description: str = "", 
        problem_domain: str = "general_humanitarian"
    ) -> List[Dataset]:
        """
        Recommend datasets from verified humanitarian sources
        """
        
        try:
            logger.info(f"Getting datasets for domain: {problem_domain}")
            
            datasets = await self.dataset_discovery.recommend_datasets(
                project_description, 
                use_case_title, 
                use_case_description, 
                problem_domain
            )
            
            datasets.sort(key=lambda x: (
                -(x.suitability_score or 0.0),
                x.source,
                x.name
            ))
            
            source_counts = {}
            for dataset in datasets:
                source = dataset.source
                source_counts[source] = source_counts.get(source, 0) + 1
            
            logger.info(f"Found {len(datasets)} datasets from sources: {source_counts}")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            return []