from typing import List, Dict, Any, Optional
from config import settings
from services.use_case_service import UseCaseService
from services.datasets_discovery_service import DatasetDiscoveryService
from models.project import Dataset
import logging

logger = logging.getLogger(__name__)

class ScopingService:
    """Humanitarian AI project scoping service providing infrastructure-aware use case discovery and practical dataset recommendations"""
    
    def __init__(self):
        self.use_case_service = UseCaseService()
        self.dataset_discovery = DatasetDiscoveryService()
    
    async def get_educational_use_cases(
        self, 
        project_description: str, 
        problem_domain: str,
        technical_infrastructure: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Discover relevant AI use cases with educational content, filtered by infrastructure compatibility"""
        
        logger.info(f"Discovering use cases for domain: {problem_domain}")
        if technical_infrastructure:
            logger.info(f"Infrastructure constraints: {technical_infrastructure}")
        
        try:
            use_cases = await self.use_case_service.search_ai_use_cases(
                project_description, problem_domain, technical_infrastructure
            )
            
            logger.info(f"Discovered {len(use_cases)} infrastructure-compatible AI use cases")
            return use_cases
            
        except Exception as e:
            logger.error(f"Use case discovery failed: {e}")
            return []
    
    async def recommend_datasets(
        self, 
        project_description: str, 
        use_case_title: str = "", 
        use_case_description: str = "", 
        problem_domain: str = ""
    ) -> List[Dataset]:
        """Recommend practical humanitarian datasets using enhanced discovery with terminology matching"""
        
        try:
            logger.info(f"Starting dataset recommendation for domain: {problem_domain}")
            
            datasets = await self.dataset_discovery.recommend_datasets(
                project_description, 
                use_case_title, 
                use_case_description, 
                problem_domain
            )
            
            if datasets:
                datasets.sort(key=lambda x: (
                    -(x.suitability_score or 0.0),
                    x.source,
                    x.name
                ))
                
                source_distribution = {}
                for dataset in datasets:
                    source = dataset.source
                    source_distribution[source] = source_distribution.get(source, 0) + 1
                
                logger.info(f"Recommended {len(datasets)} datasets from sources: {source_distribution}")
            else:
                logger.info("No datasets found matching project requirements")
            
            return datasets
            
        except Exception as e:
            logger.error(f"Dataset recommendation failed: {e}")
            return []