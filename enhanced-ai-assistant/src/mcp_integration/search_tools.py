import os
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SearchTools:
    """Enhanced search tools integration using MCP client and Brave Search API."""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        # Set the API key from environment variable
        self.api_key = os.getenv("BRAVE_API_KEY")
        # Alternative attribute names in case your code uses different naming
        self.brave_api_key = self.api_key
        
        # Search configuration
        self.search_config = {
            'default_count': 10,
            'max_count': 20,
            'timeout': 10,
            'user_agent': 'AI-Assistant/1.0'
        }
        
        # Available search tools
        self.available_tools = [
            'web_search',
            'search_web',
            'brave_search',
            'search_summarize',
            'search_analyze'
        ]
        
        if not self.api_key:
            logger.error("BRAVE_API_KEY not found in environment variables")
        else:
            logger.info(f"Brave API key loaded (length: {len(self.api_key)})")
    
    async def start(self):
        """Start the search tools service."""
        if not self.api_key:
            raise ValueError("Brave API key required. Check your .env file contains BRAVE_API_KEY=your_key")
        
        # Test API connectivity
        try:
            test_result = await self.web_search("test", count=1)
            logger.info("SearchTools connectivity test successful")
        except Exception as e:
            logger.warning(f"SearchTools connectivity test failed: {e}")
        
        logger.info("SearchTools started successfully")
        return True
    
    async def web_search(self, query: str, count: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Perform web search using Brave Search API.
        
        Args:
            query: Search query string
            count: Number of results to return (default: 10)
            **kwargs: Additional search parameters
            
        Returns:
            Dict with search results and metadata
        """
        if not self.api_key:
            raise ValueError("Brave API key not available")
        
        # Validate and limit count
        count = min(count, self.search_config['max_count'])
        
        try:
            import httpx
            
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
                "User-Agent": self.search_config['user_agent']
            }
            
            # Build search parameters
            params = {
                "q": query,
                "count": count,
                "safesearch": kwargs.get('safesearch', 'moderate'),
                "freshness": kwargs.get('freshness', ''),
                "country": kwargs.get('country', ''),
                "search_lang": kwargs.get('search_lang', 'en')
            }
            
            # Remove empty parameters
            params = {k: v for k, v in params.items() if v}
            
            async with httpx.AsyncClient(timeout=self.search_config['timeout']) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # Parse Brave Search API response
                web_results = data.get("web", {}).get("results", [])
                for result in web_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "age": result.get("age", ""),
                        "language": result.get("language", ""),
                        "family_friendly": result.get("family_friendly", True),
                        "meta_url": result.get("meta_url", {}),
                        "thumbnail": result.get("thumbnail", {}),
                        "extra_snippets": result.get("extra_snippets", [])
                    })
                
                # Extract additional metadata
                query_context = data.get("query", {})
                web_context = data.get("web", {})
                
                search_result = {
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                    "query_altered": query_context.get("altered", False),
                    "original_query": query_context.get("original", query),
                    "spellcheck_off": query_context.get("spellcheck_off", False),
                    "family_friendly": web_context.get("family_friendly", True),
                    "search_timestamp": asyncio.get_event_loop().time()
                }
                
                logger.info(f"Web search completed: {len(results)} results for '{query}'")
                return search_result
                
        except ImportError:
            raise ImportError("httpx library required for web search. Install with: pip install httpx")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise
    
    async def search_web(self, query: str, count: int = 10) -> Dict[str, Any]:
        """Alias for web_search to match agent expectations."""
        return await self.web_search(query, count)
    
    async def brave_search(self, query: str, count: int = 10, **kwargs) -> Dict[str, Any]:
        """Direct Brave search method."""
        return await self.web_search(query, count, **kwargs)
    
    async def search_summarize(self, search_results: Dict[str, Any], max_results: int = 5) -> Dict[str, Any]:
        """
        Summarize search results into key insights.
        
        Args:
            search_results: Results from web_search
            max_results: Maximum results to include in summary
            
        Returns:
            Summarized search data
        """
        if not search_results or not search_results.get('results'):
            return {"error": "No search results to summarize"}
        
        results = search_results['results'][:max_results]
        
        # Extract key information
        domains = set()
        topics = []
        
        for result in results:
            # Extract domain
            if result.get('url'):
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(result['url']).netloc
                    domains.add(domain)
                except:
                    pass
            
            # Extract key topics from title and description
            text = f"{result.get('title', '')} {result.get('description', '')}"
            topics.extend(self._extract_keywords(text))
        
        # Count topic frequency
        from collections import Counter
        topic_counts = Counter(topics)
        
        summary = {
            "query": search_results.get('query', ''),
            "total_results": search_results.get('total_results', 0),
            "summarized_results": len(results),
            "top_domains": list(domains)[:5],
            "key_topics": [topic for topic, count in topic_counts.most_common(10)],
            "results_preview": [
                {
                    "title": r.get('title', ''),
                    "url": r.get('url', ''),
                    "description": r.get('description', '')[:200] + "..." if len(r.get('description', '')) > 200 else r.get('description', '')
                }
                for r in results[:3]
            ]
        }
        
        return summary
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        
        # Clean text and extract words
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter stop words and short words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'how', 
            'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 
            'man', 'car', 'way', 'with', 'this', 'that', 'from', 'they', 'she', 
            'been', 'than', 'what', 'when', 'where', 'will', 'more'
        }
        
        keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
        return keywords
    
    async def search_analyze(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search results for patterns and insights.
        
        Args:
            search_results: Results from web_search
            
        Returns:
            Analysis of search patterns
        """
        if not search_results or not search_results.get('results'):
            return {"error": "No search results to analyze"}
        
        results = search_results['results']
        
        # Domain analysis
        domains = {}
        for result in results:
            if result.get('url'):
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(result['url']).netloc
                    domains[domain] = domains.get(domain, 0) + 1
                except:
                    pass
        
        # Content analysis
        all_text = ' '.join([
            f"{r.get('title', '')} {r.get('description', '')}" 
            for r in results
        ])
        keywords = self._extract_keywords(all_text)
        
        from collections import Counter
        keyword_counts = Counter(keywords)
        
        # Quality analysis
        quality_indicators = {
            'has_description': sum(1 for r in results if r.get('description')),
            'has_age_info': sum(1 for r in results if r.get('age')),
            'family_friendly': sum(1 for r in results if r.get('family_friendly', True)),
            'avg_title_length': sum(len(r.get('title', '')) for r in results) / len(results) if results else 0,
            'avg_desc_length': sum(len(r.get('description', '')) for r in results) / len(results) if results else 0
        }
        
        analysis = {
            "query": search_results.get('query', ''),
            "total_results": len(results),
            "domain_distribution": dict(sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_keywords": [keyword for keyword, count in keyword_counts.most_common(15)],
            "quality_metrics": quality_indicators,
            "content_diversity": len(set(keywords)) / len(keywords) if keywords else 0,
            "analysis_timestamp": asyncio.get_event_loop().time()
        }
        
        return analysis
    
    def get_available_tools(self) -> List[str]:
        """Get list of available search tools."""
        return self.available_tools.copy()
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get current search configuration."""
        return self.search_config.copy()
    
    async def shutdown(self):
        """Shutdown the search tools."""
        logger.info("SearchTools shutdown completed")
        pass