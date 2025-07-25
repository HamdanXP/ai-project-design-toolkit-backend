import asyncio
import logging
from typing import List, Dict, Any, Optional
from core.llm_service import llm_service
from utils.session_manager import session_manager
from config import settings
import json
import xml.etree.ElementTree as ET
import re

logger = logging.getLogger(__name__)

class UseCaseService:
    
    def __init__(self):
        pass

    async def search_ai_use_cases(
        self, 
        project_description: str, 
        problem_domain: str,
        technical_infrastructure: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        
        try:
            search_terms = await self._generate_search_terms(project_description, problem_domain)
            if not search_terms:
                logger.error("Failed to generate search terms")
                return []
            
            all_papers = await self._search_all_sources(search_terms)
            if not all_papers:
                logger.warning("No papers found from academic sources")
                return []
                
            await asyncio.sleep(0.5)
            relevant_papers = await self._filter_relevant_papers(
                all_papers, project_description, problem_domain, technical_infrastructure
            )
            
            if not relevant_papers:
                logger.warning("No relevant AI papers found")
                return []
            
            await asyncio.sleep(0.5)
            use_cases = await self._add_educational_content(
                relevant_papers, project_description, problem_domain, technical_infrastructure
            )
            
            final_count = min(len(use_cases), settings.max_use_cases_returned)
            final_use_cases = use_cases[:final_count]
            
            return final_use_cases
            
        except Exception as e:
            logger.error(f"Use case search failed: {e}")
            return []

    async def _generate_search_terms(
        self, 
        project_description: str, 
        problem_domain: str
    ) -> Optional[Dict[str, List[str]]]:
        
        prompt = f"""
        Create specific AI research search queries for this humanitarian project:
        
        Project: "{project_description}"
        Domain: {problem_domain}
        
        Generate 2 focused AI search queries per source that find papers about artificial intelligence, machine learning, or automated systems for this specific problem:
        {{
            "arxiv": ["AI/ML query 1 with specific technical terms", "AI/ML query 2 with implementation focus"],
            "semantic_scholar": ["machine learning query 1", "AI methodology query 2"], 
            "openalex": ["artificial intelligence query 1", "automated system query 2"]
        }}
        
        Requirements:
        - Every query must include AI/ML terms (artificial intelligence, machine learning, algorithm, model, etc.)
        - Keep queries focused on the problem domain without infrastructure specifics
        - Target papers that implement AI solutions for humanitarian problems
        - Use academic terminology that would appear in paper titles/abstracts
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            cleaned = self._clean_json_response(response)
            queries = json.loads(cleaned)
            return queries
        except Exception as e:
            logger.error(f"Search term generation failed: {e}")
            return None

    async def _search_all_sources(self, search_terms: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        
        tasks = []
        enabled_sources = []
        
        if settings.enable_arxiv and search_terms.get("arxiv"):
            tasks.append(self._search_source("arxiv", search_terms["arxiv"]))
            enabled_sources.append("arxiv")
            
        if settings.enable_semantic_scholar and search_terms.get("semantic_scholar"):
            tasks.append(self._search_source("semantic_scholar", search_terms["semantic_scholar"]))
            enabled_sources.append("semantic_scholar")
            
        if settings.enable_openalex and search_terms.get("openalex"):
            tasks.append(self._search_source("openalex", search_terms["openalex"]))
            enabled_sources.append("openalex")
        
        if not tasks:
            logger.error("No search tasks created")
            return []
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_papers = []
            for i, result in enumerate(results):
                source_name = enabled_sources[i]
                
                if isinstance(result, Exception):
                    logger.error(f"{source_name} search failed: {result}")
                elif isinstance(result, list):
                    all_papers.extend(result)
            
            seen_titles = set()
            unique_papers = []
            
            for paper in all_papers:
                title_normalized = re.sub(r'[^\w\s]', '', paper.get("title", "").lower())
                title_normalized = re.sub(r'\s+', ' ', title_normalized).strip()
                
                if title_normalized and title_normalized not in seen_titles:
                    seen_titles.add(title_normalized)
                    unique_papers.append(paper)
            
            return unique_papers
            
        except Exception as e:
            logger.error(f"Parallel search failed: {e}")
            return []

    async def _search_source(self, source: str, queries: List[str]) -> List[Dict[str, Any]]:
        
        papers = []
        max_per_query = 10
        
        for query in queries[:2]:
            try:
                if source == "arxiv":
                    batch = await self._search_arxiv(query, max_per_query)
                elif source == "semantic_scholar":
                    batch = await self._search_semantic_scholar(query, max_per_query)
                elif source == "openalex":
                    batch = await self._search_openalex(query, max_per_query)
                else:
                    continue
                    
                papers.extend(batch)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"{source} query failed: {e}")
                continue
        
        return papers

    async def _filter_relevant_papers(
        self, 
        papers: List[Dict[str, Any]], 
        project_description: str,
        problem_domain: str,
        technical_infrastructure: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        
        if not papers:
            return []
        
        ai_papers = self._filter_ai_papers(papers)
        
        if not ai_papers:
            logger.warning("No papers contain sufficient AI content")
            return []
            
        relevant_papers = await self._llm_relevance_filter(
            ai_papers[:20], project_description, problem_domain, technical_infrastructure
        )
        
        return relevant_papers

    def _filter_ai_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        strong_ai_terms = [
            "artificial intelligence", "machine learning", "deep learning", "neural network",
            "ai", "ml", "algorithm", "model", "prediction", "classification", 
            "recommendation", "automated", "intelligent", "learning", "training"
        ]
        
        ai_papers = []
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('description', '')}".lower()
            
            ai_score = sum(text.count(term) for term in strong_ai_terms)
            
            if ai_score >= 3 or any(term in text for term in ["artificial intelligence", "machine learning", "deep learning", "neural network"]):
                ai_papers.append(paper)
            
        return ai_papers

    async def _llm_relevance_filter(
        self, 
        papers: List[Dict[str, Any]], 
        project_description: str,
        problem_domain: str,
        technical_infrastructure: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        
        papers_text = ""
        for i, paper in enumerate(papers):
            papers_text += f"""
Paper {i+1}:
Title: {paper.get('title', '')[:150]}
Abstract: {paper.get('description', '')[:300]}
---
"""
        
        infrastructure_context = ""
        if technical_infrastructure:
            infrastructure_context = f"""
            
            Available Infrastructure:
            - Computing: {technical_infrastructure.get('computing_resources', 'unspecified')}
            - Storage: {technical_infrastructure.get('storage_data', 'unspecified')}
            - Connectivity: {technical_infrastructure.get('internet_connectivity', 'unspecified')}
            - Deployment: {technical_infrastructure.get('deployment_environment', 'unspecified')}
            
            Consider whether each AI approach could be adapted to work with this infrastructure.
            """
        
        prompt = f"""
        Rate these AI research papers for relevance to this humanitarian project:
        
        Project: "{project_description}"
        Domain: {problem_domain}
        {infrastructure_context}
        
        Papers:
        {papers_text}
        
        CRITERIA:
        - Rate 8-10: AI methods directly applicable to the project goals
        - Rate 6-7: AI techniques that could be adapted for the project
        - Rate 4-5: Relevant AI domain but requires adaptation
        - Rate 0-3: AI paper but not relevant to the project
        
        For infrastructure consideration: Most AI approaches can be adapted to different infrastructure setups, so focus primarily on problem relevance rather than exact infrastructure match.
        
        Return JSON array:
        [
            {{"paper_number": 1, "relevance_score": 7, "reason": "AI refugee matching relates to registration systems"}},
            ...
        ]
        
        Focus on practical AI applicability to the humanitarian goals.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            cleaned = self._clean_json_response(response)
            scores = json.loads(cleaned)
            
            relevant_papers = []
            for score_data in scores:
                paper_idx = score_data.get("paper_number", 0) - 1
                relevance = score_data.get("relevance_score", 0)
                
                if 0 <= paper_idx < len(papers) and relevance >= 6:
                    paper = papers[paper_idx].copy()
                    paper["relevance_score"] = relevance
                    paper["relevance_reason"] = score_data.get("reason", "")
                    relevant_papers.append(paper)
            
            relevant_papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return relevant_papers
            
        except Exception as e:
            logger.error(f"LLM relevance filtering failed: {e}")
            return []

    async def _add_educational_content(
        self, 
        papers: List[Dict[str, Any]], 
        project_description: str,
        problem_domain: str,
        technical_infrastructure: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        
        enhanced_papers = []
        batch_size = 3
        
        for i in range(0, len(papers[:8]), batch_size):
            batch = papers[i:i + batch_size]
            
            try:
                if i > 0:
                    await asyncio.sleep(0.5)
                    
                enhanced_batch = await self._enhance_paper_batch(
                    batch, project_description, problem_domain, technical_infrastructure
                )
                enhanced_papers.extend(enhanced_batch)
                
            except Exception as e:
                logger.error(f"Failed to enhance batch {i//batch_size + 1}: {e}")
                continue
        
        return enhanced_papers

    async def _enhance_paper_batch(
        self, 
        papers: List[Dict[str, Any]], 
        project_description: str,
        problem_domain: str,
        technical_infrastructure: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        
        papers_text = ""
        for i, paper in enumerate(papers):
            papers_text += f"""
Paper {i+1}:
Title: {paper.get('title', '')[:150]}
Abstract: {paper.get('description', '')[:300]}
Source: {paper.get('source', '')}
---
"""
        
        infrastructure_context = ""
        if technical_infrastructure:
            infrastructure_context = f"""
            
            Available Infrastructure:
            - Computing: {technical_infrastructure.get('computing_resources', 'unspecified')}
            - Storage: {technical_infrastructure.get('storage_data', 'unspecified')} 
            - Connectivity: {technical_infrastructure.get('internet_connectivity', 'unspecified')}
            - Deployment: {technical_infrastructure.get('deployment_environment', 'unspecified')}
            
            Provide implementation guidance that considers these infrastructure constraints.
            """
        
        prompt = f"""
        Create educational content for these {len(papers)} AI research papers for humanitarian practitioners:
        
        Project Context: "{project_description}" (Domain: {problem_domain})
        {infrastructure_context}
        
        Research Papers:
        {papers_text}
        
        For each paper, return JSON array with practical guidance:
        [
            {{
                "paper_number": 1,
                "how_it_works": "2-3 sentences explaining the AI approach in simple terms",
                "real_world_impact": "How this AI solution could help humanitarian work",
                "similarity_to_project": "Direct relevance to the user's specific project",
                "implementation_approach": "Practical steps for humanitarian implementation considering available infrastructure",
                "key_success_factors": ["3-4 critical success factors"],
                "resource_requirements": ["3-4 specific resource needs"],
                "challenges": ["2-3 implementation challenges"],
                "decision_guidance": "Should this approach be pursued given the project goals?"
            }},
            ...
        ]
        
        Focus on practical guidance for non-technical humanitarian professionals.
        Return ONLY the JSON array.
        """
        
        try:
            response = await llm_service.analyze_text("", prompt)
            cleaned_response = self._clean_json_response(response)
            educational_contents = json.loads(cleaned_response)
            
            enhanced_papers = []
            for i, paper in enumerate(papers):
                enhanced_paper = paper.copy()
                
                if i < len(educational_contents):
                    content = educational_contents[i]
                    enhanced_paper.update({
                        "id": f"paper_{hash(paper.get('title', '')) % 100000}",
                        "how_it_works": content.get("how_it_works", ""),
                        "real_world_impact": content.get("real_world_impact", ""),
                        "similarity_to_project": content.get("similarity_to_project", ""),
                        "implementation_approach": content.get("implementation_approach", ""),
                        "key_success_factors": content.get("key_success_factors", []),
                        "resource_requirements": content.get("resource_requirements", []),
                        "challenges": content.get("challenges", []),
                        "decision_guidance": content.get("decision_guidance", "")
                    })
                else:
                    enhanced_paper.update({
                        "id": f"paper_{hash(paper.get('title', '')) % 100000}",
                        "how_it_works": "This AI research provides technical insights for humanitarian applications.",
                        "real_world_impact": f"Could offer valuable approaches for {problem_domain.replace('_', ' ')} implementations.",
                        "similarity_to_project": "Identified as relevant to your project goals.",
                        "implementation_approach": "Review the source paper for implementation details."
                    })
                
                enhanced_papers.append(enhanced_paper)
            
            return enhanced_papers
            
        except Exception as e:
            logger.error(f"Batch educational enhancement failed: {e}")
            raise

    async def _search_arxiv(self, query: str, limit: int) -> List[Dict[str, Any]]:
        papers = []
        
        async with session_manager.get_session() as session:
            try:
                params = {
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance"
                }
                
                async with session.get(
                    "http://export.arxiv.org/api/query", 
                    params=params,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        parsed_papers = self._parse_arxiv_xml(xml_content)
                        
                        for paper in parsed_papers[:limit]:
                            if paper.get("title") and paper.get("summary"):
                                papers.append({
                                    "title": paper["title"],
                                    "description": paper["summary"][:500],
                                    "source": "arXiv",
                                    "source_url": paper.get("link", ""),
                                    "type": "academic_paper",
                                    "authors": paper.get("authors", []),
                                    "year": paper.get("published", "")[:4] if paper.get("published") else None,
                                    "venue": "arXiv preprint"
                                })
                    
            except Exception as e:
                logger.error(f"arXiv search failed: {e}")
        
        return papers

    async def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict[str, Any]]:
        papers = []
        
        headers = {}
        if settings.semantic_scholar_api_key:
            headers['x-api-key'] = settings.semantic_scholar_api_key
        
        async with session_manager.get_session() as session:
            try:
                params = {
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,url,year,authors,venue,citationCount,isOpenAccess"
                }
                
                async with session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params=params,
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for paper in data.get("data", [])[:limit]:
                            title = paper.get("title", "").strip()
                            abstract = paper.get("abstract", "")
                            
                            if title and abstract:
                                papers.append({
                                    "title": title,
                                    "description": abstract[:500],
                                    "source": "Semantic Scholar",
                                    "source_url": paper.get("url", ""),
                                    "type": "academic_paper",
                                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                                    "year": paper.get("year"),
                                    "venue": paper.get("venue", ""),
                                    "citation_count": paper.get("citationCount", 0),
                                    "open_access": paper.get("isOpenAccess", False)
                                })
                    
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed: {e}")
        
        return papers

    async def _search_openalex(self, query: str, limit: int) -> List[Dict[str, Any]]:
        papers = []
        
        async with session_manager.get_session() as session:
            try:
                params = {
                    "search": query,
                    "per-page": limit,
                    "sort": "relevance_score:desc"
                }
                
                async with session.get(
                    "https://api.openalex.org/works",
                    params=params,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for work in data.get("results", [])[:limit]:
                            title = work.get("title") 
                            if title:
                                title = title.strip()
                            
                            abstract = work.get("abstract") or work.get("abstract_inverted_index")
                            
                            if isinstance(abstract, dict):
                                words = []
                                for word, positions in list(abstract.items())[:100]:
                                    words.extend([word] * len(positions))
                                abstract = " ".join(words[:200])
                            elif abstract:
                                abstract = str(abstract)
                            
                            if title and abstract and len(title.strip()) > 10 and len(abstract.strip()) > 50:
                                paper_url = ""
                                primary_location = work.get("primary_location", {})
                                if primary_location and primary_location.get("landing_page_url"):
                                    paper_url = primary_location["landing_page_url"]
                                elif work.get("doi"):
                                    paper_url = f"https://doi.org/{work['doi']}"
                                
                                papers.append({
                                    "title": title,
                                    "description": abstract[:500],
                                    "source": "OpenAlex",
                                    "source_url": paper_url,
                                    "type": "academic_paper",
                                    "year": work.get("publication_year"),
                                    "citation_count": work.get("cited_by_count", 0)
                                })
                    
            except Exception as e:
                logger.error(f"OpenAlex search failed: {e}")
        
        return papers

    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        papers = []
        
        try:
            xml_content = xml_content.replace('xmlns="http://www.w3.org/2005/Atom"', '')
            root = ET.fromstring(xml_content)
            
            for entry in root.findall('.//entry'):
                paper = {}
                
                title_elem = entry.find('title')
                if title_elem is not None:
                    paper['title'] = re.sub(r'\s+', ' ', title_elem.text).strip()
                
                summary_elem = entry.find('summary')
                if summary_elem is not None:
                    paper['summary'] = re.sub(r'\s+', ' ', summary_elem.text).strip()
                
                link_elem = entry.find('link[@type="text/html"]')
                if link_elem is not None:
                    paper['link'] = link_elem.get('href')
                
                published_elem = entry.find('published')
                if published_elem is not None:
                    paper['published'] = published_elem.text.strip()
                
                if paper.get('title') and paper.get('summary'):
                    papers.append(paper)
                    
        except Exception as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
        
        return papers

    def _clean_json_response(self, response: str) -> str:
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