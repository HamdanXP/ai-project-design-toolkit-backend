from typing import List, Dict, Any, Optional
from models.project import Dataset
from utils.humanitarian_sources import HumanitarianDataSources
from services.llm_analyzer import LLMAnalyzer
from core.exceptions import DataProcessingError
import aiohttp
import logging

logger = logging.getLogger(__name__)

class DatasetService:
    """Service for dataset discovery, analysis, and management"""
    
    def __init__(self):
        self.humanitarian_sources = HumanitarianDataSources()
        self.llm_analyzer = LLMAnalyzer()
    
    async def discover_datasets(
        self,
        project_description: str,
        domain: str,
        target_population: str
    ) -> List[Dataset]:
        """Discover relevant datasets from multiple sources"""
        try:
            datasets = []
            
            # Get datasets from humanitarian sources
            humanitarian_datasets = await self._get_humanitarian_datasets(
                project_description, domain
            )
            datasets.extend(humanitarian_datasets)
            
            # Get open data sources
            open_datasets = await self._get_open_datasets(
                project_description, domain
            )
            datasets.extend(open_datasets)
            
            # Analyze and score datasets
            for dataset in datasets:
                dataset.suitability_score = await self._analyze_dataset_suitability(
                    dataset, project_description, target_population
                )
            
            # Sort by suitability score
            datasets.sort(key=lambda x: x.suitability_score or 0, reverse=True)
            
            return datasets[:10]  # Return top 10 most suitable
            
        except Exception as e:
            logger.error(f"Failed to discover datasets: {e}")
            raise DataProcessingError(f"Dataset discovery failed: {str(e)}")
    
    async def analyze_dataset_quality(
        self,
        dataset: Dataset,
        sample_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze dataset quality and characteristics"""
        try:
            quality_analysis = {
                "completeness_score": 0.8,  # Would analyze actual data
                "consistency_score": 0.7,
                "accuracy_indicators": [],
                "bias_indicators": [],
                "coverage_assessment": {},
                "recommendations": []
            }
            
            # If sample data provided, analyze it
            if sample_data:
                quality_analysis.update(
                    await self._analyze_sample_data(sample_data, dataset)
                )
            
            # Generate recommendations based on analysis
            quality_analysis["recommendations"] = await self._generate_quality_recommendations(
                dataset, quality_analysis
            )
            
            return quality_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze dataset quality: {e}")
            return {
                "completeness_score": 0.5,
                "consistency_score": 0.5,
                "accuracy_indicators": ["Quality analysis unavailable"],
                "bias_indicators": ["Bias assessment needed"],
                "coverage_assessment": {"status": "Requires manual review"},
                "recommendations": ["Conduct thorough data quality assessment"]
            }
    
    async def validate_dataset_ethics(
        self,
        dataset: Dataset,
        project_context: str
    ) -> Dict[str, Any]:
        """Validate ethical considerations for dataset use"""
        try:
            # Check for common ethical concerns
            ethical_assessment = {
                "privacy_risk": "medium",
                "consent_status": "unknown",
                "bias_risk": "medium", 
                "representation_issues": [],
                "compliance_notes": [],
                "approval_needed": True
            }
            
            # Analyze dataset description for ethical flags
            if dataset.description:
                ethical_concerns = await self.llm_analyzer.generate_ethical_concerns(
                    project_context, dataset.description
                )
                ethical_assessment["identified_concerns"] = ethical_concerns
            
            # Check against humanitarian data standards
            humanitarian_compliance = await self._check_humanitarian_compliance(dataset)
            ethical_assessment.update(humanitarian_compliance)
            
            return ethical_assessment
            
        except Exception as e:
            logger.error(f"Failed to validate dataset ethics: {e}")
            return {
                "privacy_risk": "high",
                "consent_status": "unknown",
                "bias_risk": "high",
                "representation_issues": ["Assessment required"],
                "compliance_notes": ["Manual review needed"],
                "approval_needed": True,
                "identified_concerns": ["Ethical review required"]
            }
    
    async def prepare_dataset_integration(
        self,
        datasets: List[Dataset],
        project_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare integration plan for selected datasets"""
        try:
            integration_plan = {
                "preprocessing_steps": [],
                "harmonization_needs": [],
                "quality_checks": [],
                "ethical_safeguards": [],
                "technical_requirements": [],
                "timeline_estimate": "4-6 weeks"
            }
            
            for dataset in datasets:
                # Analyze preprocessing needs
                preprocessing = await self._analyze_preprocessing_needs(
                    dataset, project_requirements
                )
                integration_plan["preprocessing_steps"].extend(preprocessing)
                
                # Check for harmonization requirements
                if len(datasets) > 1:
                    harmonization = await self._analyze_harmonization_needs(
                        dataset, datasets
                    )
                    integration_plan["harmonization_needs"].extend(harmonization)
                
                # Add ethical safeguards
                ethical_safeguards = await self._get_ethical_safeguards(dataset)
                integration_plan["ethical_safeguards"].extend(ethical_safeguards)
            
            # Remove duplicates and sort by priority
            integration_plan = self._deduplicate_integration_plan(integration_plan)
            
            return integration_plan
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset integration: {e}")
            return {
                "preprocessing_steps": ["Standard data cleaning required"],
                "harmonization_needs": ["Schema alignment needed"],
                "quality_checks": ["Validate data integrity"],
                "ethical_safeguards": ["Implement privacy protections"],
                "technical_requirements": ["Data processing infrastructure"],
                "timeline_estimate": "6-8 weeks"
            }
    
    async def _get_humanitarian_datasets(
        self, 
        project_description: str, 
        domain: str
    ) -> List[Dataset]:
        """Get datasets from humanitarian sources"""
        datasets = []
        
        try:
            # ReliefWeb datasets
            reliefweb_datasets = await self.humanitarian_sources.get_reliefweb_datasets(
                project_description
            )
            datasets.extend(reliefweb_datasets)
            
            # UN datasets
            un_datasets = await self.humanitarian_sources.get_un_datasets(
                project_description
            )
            datasets.extend(un_datasets)
            
        except Exception as e:
            logger.warning(f"Failed to get humanitarian datasets: {e}")
        
        return datasets
    
    async def _get_open_datasets(
        self,
        project_description: str,
        domain: str
    ) -> List[Dataset]:
        """Get datasets from open data sources"""
        datasets = []
        
        # Add common open datasets relevant to humanitarian work
        open_datasets = [
            Dataset(
                name="World Bank Open Data",
                source="World Bank",
                url="https://data.worldbank.org/",
                description="Development indicators and socio-economic data",
                data_types=["economic", "social", "demographic"],
                ethical_concerns=["data_quality", "representation"]
            ),
            Dataset(
                name="OpenStreetMap Humanitarian Data",
                source="Humanitarian OpenStreetMap Team",
                url="https://www.hotosm.org/",
                description="Geospatial data for humanitarian response",
                data_types=["geospatial", "infrastructure", "population"],
                ethical_concerns=["location_privacy", "data_accuracy"]
            ),
            Dataset(
                name="Global Health Observatory",
                source="World Health Organization",
                url="https://www.who.int/data/gho",
                description="Global health statistics and indicators",
                data_types=["health", "epidemiological", "demographic"],
                ethical_concerns=["health_privacy", "data_sensitivity"]
            )
        ]
        
        # Filter based on domain relevance
        domain_keywords = {
            "health": ["health", "medical", "epidemic"],
            "education": ["education", "learning", "school"],
            "disaster": ["disaster", "emergency", "crisis"],
            "food": ["food", "nutrition", "agriculture"]
        }
        
        relevant_keywords = domain_keywords.get(domain.lower(), [])
        
        for dataset in open_datasets:
            if any(keyword in dataset.description.lower() for keyword in relevant_keywords):
                datasets.append(dataset)
        
        return datasets
    
    async def _analyze_dataset_suitability(
        self,
        dataset: Dataset,
        project_description: str,
        target_population: str
    ) -> float:
        """Analyze how suitable a dataset is for the project"""
        try:
            # Use LLM analyzer for detailed assessment
            assessment = await self.llm_analyzer.assess_dataset_relevance(
                dataset.description,
                project_description,
                target_population
            )
            
            return assessment.get("relevance_score", 0.5)
            
        except Exception as e:
            logger.warning(f"Failed to analyze dataset suitability: {e}")
            return 0.5
    
    async def _analyze_sample_data(
        self,
        sample_data: str,
        dataset: Dataset
    ) -> Dict[str, Any]:
        """Analyze sample data for quality indicators"""
        # Simplified analysis - would use actual data analysis in production
        return {
            "sample_size": len(sample_data.split('\n')),
            "missing_values_detected": "null" in sample_data.lower() or "" in sample_data,
            "data_format": "structured" if "," in sample_data else "unstructured",
            "potential_issues": []
        }
    
    async def _generate_quality_recommendations(
        self,
        dataset: Dataset,
        quality_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for dataset quality improvement"""
        recommendations = []
        
        if quality_analysis.get("completeness_score", 0) < 0.8:
            recommendations.append("Address missing data through imputation or collection")
        
        if quality_analysis.get("consistency_score", 0) < 0.7:
            recommendations.append("Implement data validation and standardization")
        
        if "bias" in dataset.ethical_concerns:
            recommendations.append("Conduct bias assessment and mitigation")
        
        recommendations.append("Establish data governance and quality monitoring")
        
        return recommendations
    
    async def _check_humanitarian_compliance(
        self,
        dataset: Dataset
    ) -> Dict[str, Any]:
        """Check compliance with humanitarian data standards"""
        compliance = {
            "data_responsibility_standard": "unknown",
            "humanitarian_data_exchange": "not_verified",
            "protection_sensitive": True if "protection" in dataset.description.lower() else False,
            "gdpr_compliance": "requires_assessment"
        }
        
        # Check if dataset is from known humanitarian sources
        humanitarian_sources = ["reliefweb", "unhcr", "who", "unicef", "wfp"]
        if any(source in dataset.source.lower() for source in humanitarian_sources):
            compliance["data_responsibility_standard"] = "likely_compliant"
            compliance["humanitarian_data_exchange"] = "verified_source"
        
        return compliance
    
    async def _analyze_preprocessing_needs(
        self,
        dataset: Dataset,
        project_requirements: Dict[str, Any]
    ) -> List[str]:
        """Analyze preprocessing needs for a dataset"""
        preprocessing_steps = []
        
        # Basic preprocessing steps
        preprocessing_steps.extend([
            f"Clean and validate {dataset.name} data",
            f"Handle missing values in {dataset.name}",
            f"Normalize data formats for {dataset.name}"
        ])
        
        # Add dataset-specific steps based on data types
        if "text" in dataset.data_types:
            preprocessing_steps.append(f"Text preprocessing for {dataset.name}")
        
        if "geospatial" in dataset.data_types:
            preprocessing_steps.append(f"Geospatial data processing for {dataset.name}")
        
        if "time_series" in dataset.data_types:
            preprocessing_steps.append(f"Time series alignment for {dataset.name}")
        
        return preprocessing_steps
    
    async def _analyze_harmonization_needs(
        self,
        dataset: Dataset,
        all_datasets: List[Dataset]
    ) -> List[str]:
        """Analyze needs for harmonizing multiple datasets"""
        harmonization_steps = []
        
        # Check for schema alignment needs
        harmonization_steps.append(
            f"Align {dataset.name} schema with other datasets"
        )
        
        # Check for temporal alignment
        harmonization_steps.append(
            f"Synchronize temporal dimensions for {dataset.name}"
        )
        
        # Check for geographical alignment
        if "geospatial" in dataset.data_types:
            harmonization_steps.append(
                f"Align geographical boundaries for {dataset.name}"
            )
        
        return harmonization_steps
    
    async def _get_ethical_safeguards(self, dataset: Dataset) -> List[str]:
        """Get ethical safeguards for dataset use"""
        safeguards = []
        
        # Standard safeguards
        safeguards.extend([
            f"Implement privacy protection for {dataset.name}",
            f"Establish data access controls for {dataset.name}",
            f"Document data lineage for {dataset.name}"
        ])
        
        # Dataset-specific safeguards
        if "sensitive" in dataset.ethical_concerns:
            safeguards.append(f"Enhanced protection for sensitive data in {dataset.name}")
        
        if "privacy" in dataset.ethical_concerns:
            safeguards.append(f"Anonymization protocols for {dataset.name}")
        
        return safeguards
    
    def _deduplicate_integration_plan(
        self, 
        integration_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Remove duplicates from integration plan"""
        for key in integration_plan:
            if isinstance(integration_plan[key], list):
                integration_plan[key] = list(set(integration_plan[key]))
        
        return integration_plan
