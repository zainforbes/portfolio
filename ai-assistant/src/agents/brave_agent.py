import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent
from src.core.state_schema import AssistantState
from src.core.message_types import AgentMessage, MessageTypes

@dataclass
class SearchResult:
    """Search result data structure"""
    title: str
    url: str
    description: str
    age: str = ""
    language: str = ""
    family_friendly: bool = True
    relevance_score: float = 0.0

@dataclass
class SearchSummary:
    """Search session summary"""
    query: str
    total_results: int
    top_domains: List[str]
    key_topics: List[str]
    search_timestamp: datetime
    suggested_refinements: List[str]

class BraveAgent(BaseAgent):
    """
    Specialized agent for web search using Brave Search API.
    Handles web searches, result analysis, and search optimization.
    """
    
    def __init__(self, gemini_mcp_client, agent_name: str = "BraveAgent"):
        capabilities = [
            "web_search",
            "search_result_analysis", 
            "query_optimization",
            "fact_checking",
            "research_synthesis",
            "domain_analysis",
            "content_summarization"
        ]
        
        super().__init__(gemini_mcp_client, agent_name, capabilities)
        
        # Search-specific prompt templates
        self.system_prompts.update({
            'search_analysis': """You are a search result analyzer. Your task is to:
1. Evaluate search result relevance and quality
2. Identify key themes and patterns
3. Suggest query refinements
4. Extract actionable insights
Always prioritize factual accuracy and source credibility.""",
            
            'query_optimization': """You are a search query optimizer. Help users:
1. Refine search queries for better results
2. Suggest alternative search terms
3. Identify information gaps
4. Recommend search strategies
Focus on improving search effectiveness."""
        })
        
        # Search quality thresholds
        self.quality_thresholds = {
            'min_results': 3,
            'max_results': 20,
            'relevance_threshold': 0.6
        }

    async def execute(self, state: AssistantState) -> AssistantState:
        """
        Execute search operations based on user request.
        
        Args:
            state: Current assistant state
            
        Returns:
            Updated state with search results
        """
        try:
            user_request = state.get('user_input', '')
            context = state.get('context', {})
            
            # Determine search action
            search_action = await self._classify_search_request(user_request)
            
            if search_action['action'] == 'web_search':
                return await self._handle_web_search(state, search_action)
            elif search_action['action'] == 'analyze_results':
                return await self._handle_result_analysis(state, search_action)
            elif search_action['action'] == 'optimize_query':
                return await self._handle_query_optimization(state, search_action)
            else:
                return await self._handle_general_search_help(state)
                
        except Exception as e:
            self.logger.error(f"BraveAgent execution failed: {e}")
            state['error_log'] = state.get('error_log', [])
            state['error_log'].append({
                'agent': self.agent_name,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            return state

    async def can_handle(self, request: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this agent can handle the search request.
        
        Args:
            request: User request text
            context: Additional context
            
        Returns:
            True if agent can handle the request
        """
        search_indicators = [
            'search', 'find', 'look up', 'research', 'investigate',
            'what is', 'who is', 'how to', 'when did', 'where is',
            'google', 'browse', 'web', 'internet', 'online',
            'information about', 'details on', 'facts about'
        ]
        
        request_lower = request.lower()
        return any(indicator in request_lower for indicator in search_indicators)

    async def _classify_search_request(self, request: str) -> Dict[str, Any]:
        """Classify the type of search request."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['search', 'find', 'look up', 'research']):
            return {
                'action': 'web_search',
                'query': self._extract_search_query(request),
                'intent': 'information_seeking'
            }
        elif any(word in request_lower for word in ['analyze', 'summarize', 'explain']):
            return {
                'action': 'analyze_results',
                'intent': 'result_analysis'
            }
        elif any(word in request_lower for word in ['optimize', 'improve', 'refine']):
            return {
                'action': 'optimize_query',
                'intent': 'query_improvement'
            }
        else:
            return {
                'action': 'web_search',
                'query': request,
                'intent': 'general_search'
            }

    def _extract_search_query(self, request: str) -> str:
        """Extract the actual search query from the request."""
        # Remove common command words
        command_words = ['search for', 'find', 'look up', 'research', 'google', 'search']
        query = request.lower()
        
        for word in command_words:
            query = query.replace(word, '')
        
        return query.strip()

    async def _handle_web_search(self, state: AssistantState, search_action: Dict[str, Any]) -> AssistantState:
        """Handle web search requests."""
        try:
            query = search_action['query']
            if not query:
                query = state.get('user_input', '')
            
            self.logger.info(f"Performing web search for: {query}")
            
            # Perform search using MCP tool
            search_results = await self.use_tool('search_web', {
                'query': query,
                'count': 10
            })
            
            if not search_results or 'results' not in search_results:
                # Fallback: try direct search through Gemini MCP client
                search_results = await self.gemini_mcp_client._search_web(query, 10)
            
            # Check if we got a fallback response instead of search results
            if search_results.get('search_unavailable'):
                response = f"**Search Results for '{query}'**\n\n"
                response += search_results.get('message', 'Search temporarily unavailable') + "\n\n"
                response += search_results.get('fallback_response', 'No additional information available.')
            else:
                # Process and analyze results
                processed_results = await self._process_search_results(search_results, query)
                
                # Generate response
                response = await self._generate_search_response(query, processed_results)
            
            # Update state
            state['final_response'] = response
            state['search_results'] = search_results.get('results', []) if not search_results.get('search_unavailable') else []
            state['search_query'] = query
            state['task_type'] = 'web_search'
            
            return state
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            state['final_response'] = f"I encountered an error while searching: {str(e)}"
            return state

    async def _process_search_results(self, search_results: Dict[str, Any], query: str) -> List[SearchResult]:
        """Process and enrich search results."""
        processed_results = []
        
        results_list = search_results.get('results', [])
        if not results_list:
            return processed_results
        
        for result in results_list:
            search_result = SearchResult(
                title=result.get('title', ''),
                url=result.get('url', ''),
                description=result.get('description', ''),
                age=result.get('age', ''),
                language=result.get('language', ''),
                family_friendly=result.get('family_friendly', True)
            )
            
            # Calculate relevance score
            search_result.relevance_score = await self._calculate_relevance_score(
                search_result, query
            )
            
            processed_results.append(search_result)
        
        # Sort by relevance score
        processed_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return processed_results

    async def _calculate_relevance_score(self, result: SearchResult, query: str) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        query_words = query.lower().split()
        
        # Check title relevance
        title_words = result.title.lower().split()
        title_matches = sum(1 for word in query_words if word in title_words)
        score += (title_matches / len(query_words)) * 0.4
        
        # Check description relevance
        desc_words = result.description.lower().split()
        desc_matches = sum(1 for word in query_words if word in desc_words)
        score += (desc_matches / len(query_words)) * 0.3
        
        # URL quality indicators
        if any(domain in result.url for domain in ['.edu', '.gov', '.org']):
            score += 0.2
        
        # Recent content bonus
        if result.age and any(term in result.age.lower() for term in ['hour', 'day', 'week']):
            score += 0.1
        
        return min(score, 1.0)

    async def _generate_search_response(self, query: str, results: List[SearchResult]) -> str:
        """Generate a comprehensive search response."""
        if not results:
            return f"I couldn't find any results for '{query}'. You might want to try different search terms."
        
        # Filter high-quality results
        quality_results = [r for r in results if r.relevance_score >= self.quality_thresholds['relevance_threshold']]
        
        if not quality_results:
            quality_results = results[:3]  # Show top 3 if none meet threshold
        
        # Create prompt for response generation
        results_summary = []
        for i, result in enumerate(quality_results[:5], 1):
            results_summary.append(f"{i}. {result.title}\n   {result.description}\n   {result.url}")
        
        prompt = f"""Answer this question directly and concisely: "{query}"

Based on these search results:
{chr(10).join(results_summary)}

Requirements:
- Provide a direct answer in 1-2 sentences
- Use the most relevant information
- Be factual and concise
- No explanations or elaborations"""
        
        response = await self.generate_response(prompt)
        
        # Return just the concise answer without metadata
        return response

    async def _handle_result_analysis(self, state: AssistantState, search_action: Dict[str, Any]) -> AssistantState:
        """Handle search result analysis requests."""
        search_results = state.get('search_results', [])
        
        if not search_results:
            state['final_response'] = "No search results to analyze. Please perform a search first."
            return state
        
        # Analyze search patterns
        analysis = await self._analyze_search_patterns(search_results)
        
        # Generate analysis response
        response = await self.generate_response(
            f"Analyze these search result patterns: {analysis}",
            context=self.system_prompts['search_analysis']
        )
        
        state['final_response'] = response
        state['search_analysis'] = analysis
        
        return state

    async def _analyze_search_patterns(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze patterns in search results."""
        if not results:
            return {}
        
        # Extract domains
        domains = [result.url.split('/')[2] if '/' in result.url else result.url for result in results]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Find common themes in titles
        all_titles = ' '.join([result.title for result in results])
        common_words = self._extract_common_words(all_titles)
        
        return {
            'total_results': len(results),
            'top_domains': sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'common_themes': common_words,
            'avg_relevance_score': sum(r.relevance_score for r in results) / len(results),
            'quality_sources': len([r for r in results if r.relevance_score > 0.7])
        }

    def _extract_common_words(self, text: str, min_length: int = 4) -> List[str]:
        """Extract common meaningful words from text."""
        import re
        from collections import Counter
        
        # Clean and split text
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out common stop words and short words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'car', 'way'}
        
        meaningful_words = [w for w in words if len(w) >= min_length and w not in stop_words]
        
        # Get most common words
        word_counts = Counter(meaningful_words)
        return [word for word, count in word_counts.most_common(10)]

    async def _handle_query_optimization(self, state: AssistantState, search_action: Dict[str, Any]) -> AssistantState:
        """Handle query optimization requests."""
        original_query = state.get('search_query', state.get('user_input', ''))
        
        # Generate optimization suggestions
        prompt = f"""Optimize this search query for better results: "{original_query}"

Provide:
1. Alternative query formulations
2. More specific terms to try
3. Related keywords to explore
4. Search strategies for this topic

Focus on improving search effectiveness and finding authoritative sources."""
        
        response = await self.generate_response(
            prompt,
            context=self.system_prompts['query_optimization']
        )
        
        state['final_response'] = response
        state['task_type'] = 'query_optimization'
        
        return state

    async def _handle_general_search_help(self, state: AssistantState) -> AssistantState:
        """Handle general search help requests."""
        response = """I can help you with web searches and research tasks:

" **Web Search**: Find information on any topic
" **Result Analysis**: Analyze search patterns and themes
" **Query Optimization**: Improve your search terms
" **Fact Checking**: Verify information from multiple sources
" **Research Synthesis**: Combine information from various sources

Just ask me to search for something, or say "search for [your topic]" to get started!"""
        
        state['final_response'] = response
        state['task_type'] = 'search_help'
        
        return state

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search-specific performance statistics."""
        base_stats = self.get_status()
        
        # Add search-specific metrics
        base_stats.update({
            'search_capabilities': self.capabilities,
            'quality_thresholds': self.quality_thresholds,
            'supported_search_types': [
                'web_search',
                'result_analysis', 
                'query_optimization',
                'research_synthesis'
            ]
        })
        
        return base_stats