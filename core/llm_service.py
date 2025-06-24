from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm = None
        self.embed_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM and embedding models"""
        try:
            self.llm = OpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=settings.llm_temperature,
                system_prompt="You are an AI assistant specialized in humanitarian AI project design. You help humanitarian professionals develop ethical, responsible AI solutions."
            )
            
            self.embed_model = OpenAIEmbedding(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
                embed_batch_size=settings.embedding_batch_size
            )
            
            # Set global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            
            logger.info("LLM services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM services: {e}")
            raise
    
    async def analyze_text(self, text: str, prompt: str) -> str:
        """Analyze text using LLM"""
        try:
            full_prompt = f"{prompt}\n\nText to analyze:\n{text}"
            response = await self.llm.acomplete(full_prompt)
            return str(response)
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise
    
    async def extract_project_info(self, description: str) -> dict:
        """Extract project title, context, and tags from description"""
        prompt = """
        Analyze the following project description and extract:
        1. A concise, professional title (max 60 characters)
        2. Key context and background information
        3. Relevant tags (5-8 tags related to humanitarian domains, AI techniques, target populations)
        
        Return the response in this exact JSON format:
        {
            "title": "extracted title",
            "context": "key context and background",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
        }
        """
        
        try:
            response = await self.analyze_text(description, prompt)
            # Parse JSON response (you might want to add JSON validation here)
            import json
            return json.loads(response)
        except Exception as e:
            logger.error(f"Project info extraction failed: {e}")
            # Return default values if extraction fails
            return {
                "title": "Humanitarian AI Project",
                "context": description[:200] + "..." if len(description) > 200 else description,
                "tags": ["humanitarian", "ai", "social-good"]
            }

# Global LLM service instance
llm_service = LLMService()