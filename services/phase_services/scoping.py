from typing import List, Dict, Any, Optional
from config import settings
from core.llm_service import llm_service
from services.use_case_search_service import UseCaseSearchService
from services.datasets.discovery_service import DatasetDiscoveryService
from models.project import UseCase, Dataset, DeploymentEnvironment
from models.phase import ScopingRequest, ScopingResponse
import logging
import json

logger = logging.getLogger(__name__)

class ScopingService:
    """
    Simplified ScopingService focused on humanitarian use case discovery and basic feasibility
    """
    
    def __init__(self):
        self.use_case_search = UseCaseSearchService()
        self.dataset_discovery = DatasetDiscoveryService()

    async def get_educational_use_cases(
        self, 
        project_description: str, 
        problem_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Get AI use cases with improved filtering and basic educational enrichment
        """
        logger.info(f"Getting educational AI use cases for domain: {problem_domain}")
        
        try:
            # Step 1: Get raw use cases from search service
            raw_use_cases = await self.use_case_search.search_ai_use_cases(
                project_description, problem_domain
            )
            
            if not raw_use_cases:
                logger.info("No raw use cases found")
                return []
            
            logger.info(f"Found {len(raw_use_cases)} raw use cases")
            
            # Step 2: Apply enhanced relevance filtering BEFORE enrichment
            filtered_use_cases = self._apply_enhanced_filtering(
                raw_use_cases, problem_domain, project_description
            )
            
            if not filtered_use_cases:
                logger.info("No use cases passed relevance filtering")
                return []
            
            logger.info(f"After filtering: {len(filtered_use_cases)} relevant use cases")
            
            # Step 3: Score and rank the filtered use cases
            scored_use_cases = self._score_and_rank_use_cases(
                filtered_use_cases, project_description, problem_domain
            )
            
            # Step 4: Take only the top N cases for enrichment
            top_use_cases = scored_use_cases[:settings.max_use_cases_returned]
            logger.info(f"Selected top {len(top_use_cases)} use cases for enrichment")
            
            # Step 5: Basic enrichment (optional, simplified)
            enriched_use_cases = await self._add_basic_educational_content(
                top_use_cases, project_description, problem_domain
            )
            
            logger.info(f"Successfully processed {len(enriched_use_cases)} use cases")
            return enriched_use_cases
            
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
        Recommend relevant datasets from humanitarian sources
        """
        
        try:
            logger.info(f"Getting datasets for project with use case: {use_case_title}")
            
            datasets = await self.dataset_discovery.recommend_datasets(
                project_description, use_case_title, use_case_description, problem_domain
            )
            
            logger.info(f"Dataset discovery completed: {len(datasets)} datasets found")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            return []

    def _apply_enhanced_filtering(
        self, 
        use_cases: List[Dict], 
        domain: str, 
        project_description: str
    ) -> List[Dict]:
        """
        Enhanced filtering with multiple criteria for better relevance
        """
        filtered_cases = []
        
        for case in use_cases:
            relevance_score = self._calculate_comprehensive_relevance(
                case, domain, project_description
            )
            
            # Only include cases that meet minimum relevance threshold
            if relevance_score >= settings.minimum_relevance_score:
                case['_relevance_score'] = relevance_score
                filtered_cases.append(case)
        
        logger.info(f"Filtered {len(use_cases)} -> {len(filtered_cases)} use cases")
        return filtered_cases

    def _calculate_comprehensive_relevance(
        self, 
        use_case: Dict, 
        domain: str, 
        project_description: str
    ) -> float:
        """
        Calculate comprehensive relevance score for filtering
        """
        score = 0.0
        text_content = f"{use_case.get('title', '')} {use_case.get('description', '')}".lower()
        
        # 1. Domain relevance (40% weight)
        domain_score = self._calculate_domain_relevance(use_case, domain)
        score += domain_score * 0.4
        
        # 2. AI/Technology relevance (30% weight)
        ai_score = self._calculate_ai_relevance(use_case)
        score += ai_score * 0.3
        
        # 3. Project-specific relevance (20% weight)
        project_score = self._calculate_project_relevance(use_case, project_description)
        score += project_score * 0.2
        
        # 4. Content quality (10% weight)
        quality_score = self._calculate_quality_score(use_case)
        score += quality_score * 0.1
        
        return min(score, 1.0)

    def _calculate_domain_relevance(self, use_case: Dict, domain: str) -> float:
        """Enhanced domain relevance calculation"""
        domain_terms = self._get_domain_search_terms(domain)
        text_content = f"{use_case.get('title', '')} {use_case.get('description', '')}".lower()
        
        # Primary domain terms (higher weight)
        primary_matches = sum(1 for term in domain_terms[:5] if term.lower() in text_content)
        
        # Secondary domain terms
        secondary_matches = sum(1 for term in domain_terms[5:10] if term.lower() in text_content)
        
        # Humanitarian context terms
        humanitarian_terms = [
            "humanitarian", "crisis", "emergency", "aid", "relief", "assistance",
            "vulnerable", "communities", "displaced", "refugees", "beneficiaries"
        ]
        humanitarian_matches = sum(1 for term in humanitarian_terms if term in text_content)
        
        # Calculate weighted score
        score = (primary_matches * 0.4) + (secondary_matches * 0.3) + (humanitarian_matches * 0.3)
        return min(score / 3, 1.0)

    def _calculate_ai_relevance(self, use_case: Dict) -> float:
        """Enhanced AI relevance calculation"""
        text_content = f"{use_case.get('title', '')} {use_case.get('description', '')}".lower()
        
        # Core AI terms (highest weight)
        core_ai_terms = [
            "artificial intelligence", "machine learning", "deep learning", "AI", "ML",
            "neural network", "algorithm", "model", "prediction", "classification"
        ]
        
        # Modern AI terms (high weight)
        modern_ai_terms = [
            "large language model", "LLM", "transformer", "GPT", "BERT",
            "natural language processing", "NLP", "computer vision", "generative AI"
        ]
        
        # Applied AI terms (medium weight)
        applied_ai_terms = [
            "automation", "intelligent", "smart", "predictive", "optimization",
            "pattern recognition", "data analysis", "analytics", "recommendation"
        ]
        
        core_matches = sum(1 for term in core_ai_terms if term in text_content)
        modern_matches = sum(1 for term in modern_ai_terms if term in text_content)
        applied_matches = sum(1 for term in applied_ai_terms if term in text_content)
        
        score = (core_matches * 0.5) + (modern_matches * 0.4) + (applied_matches * 0.1)
        return min(score / 2, 1.0)

    def _calculate_project_relevance(self, use_case: Dict, project_description: str) -> float:
        """Calculate project-specific relevance"""
        if not project_description:
            return 0.5
        
        project_keywords = [
            word.lower() for word in project_description.split() 
            if len(word) > 3 and word.lower() not in ['that', 'this', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said']
        ]
        
        text_content = f"{use_case.get('title', '')} {use_case.get('description', '')}".lower()
        matches = sum(1 for keyword in project_keywords[:10] if keyword in text_content)
        
        return min(matches / 5, 1.0)

    def _calculate_quality_score(self, use_case: Dict) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # Title quality
        title = use_case.get('title', '')
        if len(title) > 10:
            score += 0.2
        
        # Description quality
        description = use_case.get('description', '')
        if len(description) > 100:
            score += 0.3
        if len(description) > 300:
            score += 0.2
        
        # Source information
        if use_case.get('source_url'):
            score += 0.1
        if use_case.get('authors') or use_case.get('organization'):
            score += 0.1
        
        # Metadata completeness
        if use_case.get('type'):
            score += 0.1
        
        return min(score, 1.0)

    def _score_and_rank_use_cases(
        self, 
        use_cases: List[Dict], 
        project_description: str, 
        problem_domain: str
    ) -> List[Dict]:
        """Score and rank use cases by relevance"""
        # Sort by pre-calculated relevance score
        return sorted(
            use_cases, 
            key=lambda x: x.get('_relevance_score', 0), 
            reverse=True
        )

    async def _add_basic_educational_content(
        self, 
        use_cases: List[Dict], 
        project_description: str,
        problem_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Add basic educational content to use cases (simplified version)
        If enrichment fails, return use cases with empty educational fields
        """
        enriched_cases = []
        
        for use_case in use_cases:
            try:
                # Try to add basic educational content
                enriched = await self._add_simple_educational_fields(
                    use_case, project_description, problem_domain
                )
                enriched_cases.append(enriched)
                
            except Exception as e:
                logger.warning(f"Failed to enrich use case {use_case.get('title', 'Unknown')}: {e}")
                # Add empty educational fields
                use_case = self._add_empty_educational_placeholders(use_case)
                enriched_cases.append(use_case)
        
        return enriched_cases

    async def _add_simple_educational_fields(
        self, 
        use_case: Dict, 
        project_description: str,
        problem_domain: str
    ) -> Dict[str, Any]:
        """
        Add simple educational content using basic templates
        """
        # Add basic educational fields with simple content
        use_case.update({
            "how_it_works": f"This AI approach helps with {problem_domain} by analyzing data to provide insights and support decision-making.",
            "real_world_impact": f"Similar solutions have been used in humanitarian contexts to improve efficiency and outcomes in {problem_domain} work.",
            "similarity_to_project": "This use case shares similar goals with your project objectives.",
            "real_world_examples": "Check the source link for specific implementation examples and case studies.",
            "implementation_approach": "Consider starting with a pilot project to test the approach in your specific context.",
            "key_success_factors": ["Clear project goals", "Good quality data", "Stakeholder buy-in"],
            "resource_requirements": ["Technical expertise", "Data access", "Computing resources"],
            "decision_guidance": "This approach would be suitable if you have the necessary technical resources and data availability."
        })
        
        return use_case

    def _add_empty_educational_placeholders(self, use_case: Dict) -> Dict[str, Any]:
        """
        Add empty educational fields when enrichment fails
        """
        use_case.update({
            "how_it_works": "",
            "real_world_impact": "",
            "similarity_to_project": "",
            "real_world_examples": "",
            "implementation_approach": "",
            "key_success_factors": [],
            "resource_requirements": [],
            "decision_guidance": "",
            "challenges": []
        })
        
        return use_case

    def _get_domain_search_terms(self, domain: str) -> List[str]:
        """Get comprehensive search terms for a domain"""
        domain_mapping = {
            "health": [
                "health", "medical", "healthcare", "disease", "epidemic", "pandemic", 
                "hospital", "clinic", "patient", "diagnosis", "treatment", "medicine",
                "public health", "epidemiology", "medical diagnosis", "health monitoring",
                "telemedicine", "digital health", "health screening", "disease surveillance"
            ],
            "education": [
                "education", "learning", "school", "teaching", "training", "student", 
                "literacy", "curriculum", "educational technology", "e-learning", 
                "adaptive learning", "educational assessment", "personalized learning",
                "tutoring", "skill development", "knowledge transfer"
            ],
            "food_security": [
                "food security", "agriculture", "farming", "nutrition", "hunger", 
                "malnutrition", "crop", "harvest", "food systems", "supply chain", 
                "food distribution", "yield prediction", "precision agriculture",
                "crop monitoring", "livestock", "agricultural productivity"
            ],
            "water_sanitation": [
                "water", "sanitation", "WASH", "clean water", "hygiene", "water quality",
                "water management", "water monitoring", "water scarcity", "water treatment", 
                "water distribution", "sanitation systems", "waste management"
            ],
            "disaster_response": [
                "disaster", "emergency", "crisis", "disaster response", "emergency management",
                "early warning", "disaster prediction", "crisis management", "relief operations",
                "emergency coordination", "damage assessment", "evacuation", "preparedness"
            ],
            "migration_displacement": [
                "migration", "refugee", "displacement", "asylum", "population movement",
                "displacement tracking", "refugee management", "migration patterns", 
                "integration", "resettlement", "border management"
            ],
            "shelter_housing": [
                "shelter", "housing", "accommodation", "settlement", "camp management",
                "infrastructure", "construction", "urban planning", "temporary shelter",
                "housing allocation", "settlement planning"
            ],
            "protection": [
                "protection", "human rights", "safety", "security", "violence prevention",
                "child protection", "gender-based violence", "legal aid", "case management",
                "risk assessment", "safety monitoring"
            ]
        }
        
        return domain_mapping.get(domain, [domain.replace('_', ' ')])

    async def extract_problem_domain_with_llm(
        self,
        project_description: str, 
        project_context: Optional[str] = None
    ) -> str:
        """Use LLM to intelligently extract the primary humanitarian problem domain"""
        
        full_text = project_description
        if project_context:
            full_text += f"\n\nContext: {project_context}"
        
        prompt = f"""
        Analyze this humanitarian AI project and identify the primary problem domain.
        
        Project Text: "{full_text}"
        
        Classify into ONE of these specific humanitarian domains:
        - health: Medical care, disease prevention, health emergencies, epidemics, mental health
        - education: Learning, schools, literacy, educational access, skills training
        - food_security: Nutrition, hunger, agricultural production, food distribution, malnutrition
        - water_sanitation: Clean water access, sanitation systems, hygiene, water quality
        - shelter_housing: Temporary shelter, housing, displacement, settlements, infrastructure
        - protection: Human rights, gender-based violence, child protection, legal aid, safety
        - disaster_response: Emergency response, natural disasters, crisis management, early warning
        - livelihoods: Economic opportunities, employment, income generation, microfinance, markets
        - migration_displacement: Refugee services, migration patterns, displacement tracking, integration
        - logistics_supply: Supply chain, resource distribution, transportation, inventory management
        - general_humanitarian: Multi-sector or doesn't fit specific categories above
        
        Return ONLY the domain key (e.g., "health", "education", "food_security").
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            domain = response.strip().lower()
            
            valid_domains = {
                "health", "education", "food_security", "water_sanitation", 
                "shelter_housing", "protection", "disaster_response", 
                "livelihoods", "migration_displacement", "logistics_supply", 
                "general_humanitarian"
            }
            
            if domain in valid_domains:
                logger.info(f"LLM extracted domain: {domain}")
                return domain
            else:
                logger.warning(f"LLM returned unexpected domain '{domain}', using fallback")
                return self._fallback_domain_extraction(full_text)
                
        except Exception as e:
            logger.error(f"LLM domain extraction failed: {e}")
            return self._fallback_domain_extraction(full_text)

    def _fallback_domain_extraction(self, text: str) -> str:
        """Fallback domain extraction using keyword matching"""
        
        text_lower = text.lower()
        
        domain_keywords = {
            "health": [
                "health", "medical", "disease", "epidemic", "pandemic", "illness", "treatment", 
                "hospital", "clinic", "vaccination", "medicine", "healthcare", "mental health"
            ],
            "education": [
                "education", "school", "learning", "literacy", "teaching", "training", "classroom", 
                "student", "teacher", "curriculum", "skills", "knowledge"
            ],
            "food_security": [
                "food", "nutrition", "hunger", "malnutrition", "agriculture", "farming", "crop", 
                "harvest", "livestock", "food security", "famine"
            ],
            "water_sanitation": [
                "water", "sanitation", "hygiene", "toilet", "latrine", "clean water", "drinking water",
                "wash", "sewage", "waste"
            ],
            "shelter_housing": [
                "shelter", "housing", "accommodation", "settlement", "camp", "tent", "building",
                "construction", "infrastructure"
            ],
            "protection": [
                "protection", "safety", "security", "violence", "abuse", "rights", "legal", 
                "gender", "children", "vulnerable"
            ],
            "disaster_response": [
                "disaster", "emergency", "crisis", "earthquake", "flood", "hurricane", "drought",
                "tsunami", "wildfire", "response", "rescue", "relief"
            ],
            "migration_displacement": [
                "refugee", "migration", "displacement", "asylum", "border", "migrant", 
                "internally displaced", "resettlement"
            ],
            "livelihoods": [
                "livelihood", "income", "employment", "job", "economic", "business", "trade",
                "market", "microfinance", "poverty"
            ],
            "logistics_supply": [
                "logistics", "supply", "distribution", "transportation", "delivery", "warehouse",
                "inventory", "procurement"
            ]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            logger.info(f"Fallback extraction selected: {best_domain} (score: {domain_scores[best_domain]})")
            return best_domain
        
        logger.info("No specific domain detected, using general_humanitarian")
        return "general_humanitarian"
