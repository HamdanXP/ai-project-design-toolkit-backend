from typing import Optional, Dict, Any
import logging
from core.llm_service import llm_service

logger = logging.getLogger(__name__)

class ProjectAnalysisService:
    """Service for analyzing and extracting comprehensive information from project descriptions"""
    
    async def extract_project_info(
        self, 
        project_description: str, 
        project_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract comprehensive project information including domain, beneficiaries, and context"""
        
        # Combine description and context for analysis
        full_text = project_description
        if project_context:
            full_text += f"\n\nContext: {project_context}"
        
        prompt = f"""
        Analyze this humanitarian AI project description and extract key information:
        
        "{full_text}"
        
        Extract and return the following information in JSON format:
        {{
            "problem_domain": "primary domain from the list below",
            "target_beneficiaries": "specific groups this project aims to help or null if not clear",
            "geographic_context": "geographic area or context if mentioned or null if not specified",
            "urgency_level": "low, medium, high, or critical - or null if unclear",
            "ai_approach_hints": "any AI/ML approaches suggested in description or null if not mentioned"
        }}
        
        For problem_domain, classify into ONE of these specific humanitarian domains:
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
        
        For target_beneficiaries, be specific about WHO will benefit. Examples:
        - "refugees and displaced populations"
        - "children in emergency situations" 
        - "rural communities without healthcare access"
        - "elderly population in isolated areas"
        - "vulnerable women and girls"
        - "children under 5 in refugee camps"
        - "displaced families in conflict zones"
        
        IMPORTANT: 
        - Use null (not "null" string) if information cannot be determined
        - Only extract information that is clearly mentioned or strongly implied
        - Don't make assumptions or use generic defaults
        - Be specific rather than generic where possible
        
        Return ONLY valid JSON.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            
            # Clean and parse JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            import json
            extracted_info = json.loads(cleaned_response.strip())
            
            # Only validate problem_domain since it's critical
            extracted_info = self._validate_critical_fields(extracted_info, full_text)
            
            logger.info(f"Extracted project info: {extracted_info}")
            return extracted_info
            
        except Exception as e:
            logger.error(f"LLM project info extraction failed: {e}")
            # Only fallback for critical field (problem_domain)
            return {
                "problem_domain": self._extract_domain_only(full_text),
                "target_beneficiaries": None,
                "geographic_context": None,
                "urgency_level": None,
                "ai_approach_hints": None
            }
    
    async def extract_problem_domain(
        self, 
        project_description: str, 
        project_context: Optional[str] = None
    ) -> str:
        """Extract problem domain - keeping for backward compatibility"""
        project_info = await self.extract_project_info(project_description, project_context)
        return project_info.get("problem_domain", "general_humanitarian")
    
    async def extract_target_beneficiaries(
        self, 
        project_description: str, 
        project_context: Optional[str] = None
    ) -> Optional[str]:
        """Extract target beneficiaries - returns None if not extractable"""
        project_info = await self.extract_project_info(project_description, project_context)
        return project_info.get("target_beneficiaries")
    
    def _validate_critical_fields(self, extracted_info: Dict[str, Any], full_text: str) -> Dict[str, Any]:
        """Validate only critical fields that would break the system"""
        
        # Only validate problem domain since other services depend on it
        valid_domains = {
            "health", "education", "food_security", "water_sanitation", 
            "shelter_housing", "protection", "disaster_response", 
            "livelihoods", "migration_displacement", "logistics_supply", 
            "general_humanitarian"
        }
        
        problem_domain = extracted_info.get("problem_domain", "").lower()
        if problem_domain not in valid_domains:
            # Only fallback for problem_domain since it's critical
            logger.warning(f"Invalid domain '{problem_domain}', using extraction fallback")
            extracted_info["problem_domain"] = self._extract_domain_only(full_text)
        else:
            extracted_info["problem_domain"] = problem_domain
        
        # Don't modify other fields - let them be None if extraction failed
        return extracted_info
    
    def _extract_domain_only(self, text: str) -> str:
        """Fallback domain extraction using keyword matching - ONLY for critical problem_domain field"""
        
        text_lower = text.lower()
        
        # Comprehensive keyword sets for domain extraction only
        domain_keywords = {
            "health": [
                "health", "medical", "disease", "epidemic", "pandemic", "illness", "treatment", 
                "hospital", "clinic", "vaccination", "medicine", "healthcare", "mental health",
                "outbreak", "infection", "diagnosis", "patient", "doctor", "nurse", "malnutrition"
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
            logger.info(f"Keyword extraction selected domain: {best_domain} (score: {domain_scores[best_domain]})")
            return best_domain
        
        # Only fallback for critical field
        logger.warning("No domain keywords detected, using general_humanitarian")
        return "general_humanitarian"