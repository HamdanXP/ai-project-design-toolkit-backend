from typing import Optional
import logging
from core.llm_service import llm_service

logger = logging.getLogger(__name__)

class DomainExtractionService:
    """Service dedicated to extracting humanitarian problem domains from project descriptions"""
    
    async def extract_problem_domain(
        self,
        project_description: str, 
        project_context: Optional[str] = None
    ) -> str:
        """Use LLM to intelligently extract the primary humanitarian problem domain"""
        
        # Combine description and context for analysis
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
        
        Consider:
        - What is the PRIMARY problem being addressed?
        - What type of beneficiaries are mentioned?
        - What outcomes or impacts are expected?
        - What data sources or indicators are referenced?
        
        Return ONLY the domain key (e.g., "health", "education", "food_security").
        If unclear or multi-domain, choose the most prominent one or use "general_humanitarian".
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            domain = response.strip().lower()
            
            # Validate the response is one of our expected domains
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
                # If LLM returns something unexpected, try to map it
                domain_mapping = {
                    "health": ["medical", "healthcare", "disease", "epidemic", "mental"],
                    "education": ["learning", "school", "literacy", "training", "teaching"],
                    "food_security": ["food", "nutrition", "hunger", "agriculture", "farming"],
                    "water_sanitation": ["water", "sanitation", "hygiene", "wash"],
                    "shelter_housing": ["shelter", "housing", "settlement", "camp"],
                    "protection": ["protection", "rights", "violence", "safety", "legal"],
                    "disaster_response": ["disaster", "emergency", "crisis", "warning", "response"],
                    "livelihoods": ["livelihood", "economic", "employment", "income", "market"],
                    "migration_displacement": ["refugee", "migration", "displacement", "asylum"],
                    "logistics_supply": ["logistics", "supply", "distribution", "transportation"]
                }
                
                # Try to find a match
                for standard_domain, keywords in domain_mapping.items():
                    if any(keyword in domain for keyword in keywords):
                        logger.info(f"Mapped LLM response '{domain}' to: {standard_domain}")
                        return standard_domain
                
                # Fallback if no mapping found
                logger.warning(f"LLM returned unexpected domain '{domain}', using fallback")
                return "general_humanitarian"
                
        except Exception as e:
            logger.error(f"LLM domain extraction failed: {e}")
            
            # Fallback to keyword extraction
            return self._fallback_domain_extraction(full_text)

    def _fallback_domain_extraction(self, text: str) -> str:
        """Fallback domain extraction using improved keyword matching"""
        
        text_lower = text.lower()
        
        # Comprehensive keyword sets
        domain_keywords = {
            "health": [
                "health", "medical", "disease", "epidemic", "pandemic", "illness", "treatment", 
                "hospital", "clinic", "vaccination", "medicine", "healthcare", "mental health",
                "outbreak", "infection", "diagnosis", "patient", "doctor", "nurse"
            ],
            "education": [
                "education", "school", "learning", "literacy", "teaching", "training", "classroom", 
                "student", "teacher", "curriculum", "skills", "knowledge", "university", "college"
            ],
            "food_security": [
                "food", "nutrition", "hunger", "malnutrition", "agriculture", "farming", "crop", 
                "harvest", "livestock", "food security", "famine", "feeding", "dietary"
            ],
            "water_sanitation": [
                "water", "sanitation", "hygiene", "toilet", "latrine", "clean water", "drinking water",
                "wash", "sewage", "waste", "contamination"
            ],
            "shelter_housing": [
                "shelter", "housing", "accommodation", "settlement", "camp", "tent", "building",
                "construction", "infrastructure", "displacement camp"
            ],
            "protection": [
                "protection", "safety", "security", "violence", "abuse", "rights", "legal", 
                "gender", "children", "vulnerable", "trafficking", "exploitation"
            ],
            "disaster_response": [
                "disaster", "emergency", "crisis", "earthquake", "flood", "hurricane", "drought",
                "tsunami", "wildfire", "response", "rescue", "relief", "early warning"
            ],
            "migration_displacement": [
                "refugee", "migration", "displacement", "asylum", "border", "migrant", "internally displaced",
                "resettlement", "repatriation", "stateless"
            ],
            "livelihoods": [
                "livelihood", "income", "employment", "job", "economic", "business", "trade",
                "market", "microfinance", "poverty", "financial"
            ],
            "logistics_supply": [
                "logistics", "supply", "distribution", "transportation", "delivery", "warehouse",
                "inventory", "procurement", "supply chain"
            ]
        }
        
        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # Return domain with highest score
            best_domain = max(domain_scores, key=domain_scores.get)
            logger.info(f"Fallback extraction selected: {best_domain} (score: {domain_scores[best_domain]})")
            return best_domain
        
        # Ultimate fallback
        logger.info("No specific domain detected, using general_humanitarian")
        return "general_humanitarian"