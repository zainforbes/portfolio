# src/mcp_integration/custom_search_server.py
import asyncio
import json
import sys
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomSearchServer:
    """Enhanced MCP server for Brave Search API with comprehensive search tools."""
    
    def __init__(self, brave_api_key: str = None):
        self.brave_api_key = brave_api_key or os.getenv('BRAVE_API_KEY')
        self.server_info = {
            "name": "brave-search-mcp",
            "version": "1.0.0",
            "description": "Brave Search API MCP Server with advanced search capabilities"
        }
        
        # Search configuration
        self.search_config = {
            'max_results': 20,
            'default_results': 10,
            'timeout': 15,
            'user_agent': 'BraveSearchMCP/1.0'
        }
        
        # Define comprehensive tool set
        self.tools = [
            {
                "name": "web_search",
                "description": "Search the web using Brave Search API with advanced options",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of results to return (max 20)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 20
                        },
                        "safesearch": {
                            "type": "string",
                            "description": "Safe search setting",
                            "enum": ["off", "moderate", "strict"],
                            "default": "moderate"
                        },
                        "freshness": {
                            "type": "string",
                            "description": "Freshness of results",
                            "enum": ["", "pd", "pw", "pm", "py"],
                            "default": ""
                        },
                        "country": {
                            "type": "string",
                            "description": "Country code for localized results (e.g., 'US', 'GB')",
                            "default": ""
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_summarize",
                "description": "Summarize and analyze search results for key insights",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Original search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to include in summary",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_analyze",
                "description": "Analyze search patterns and extract insights from results",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to analyze"
                        },
                        "include_domains": {
                            "type": "boolean",
                            "description": "Include domain analysis",
                            "default": True
                        },
                        "include_keywords": {
                            "type": "boolean",
                            "description": "Include keyword extraction",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_health",
                "description": "Check the health and configuration of the search service",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        
        # Cache for storing recent search results
        self.search_cache = {}
        self.cache_max_size = 100
        
        logger.info(f"CustomSearchServer initialized with {len(self.tools)} tools")
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle MCP JSON-RPC requests with comprehensive error handling."""
        method = request.get("method")
        request_id = request.get("id")
        
        try:
            logger.debug(f"Handling request: {method}")
            
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "experimental": {
                                "search_caching": True,
                                "search_analytics": True
                            }
                        },
                        "serverInfo": self.server_info
                    }
                }
            
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": self.tools}
                }
            
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                logger.info(f"Executing tool: {tool_name}")
                
                # Route to appropriate tool handler
                if tool_name == "web_search":
                    result = await self._web_search(arguments)
                elif tool_name == "search_summarize":
                    result = await self._search_summarize(arguments)
                elif tool_name == "search_analyze":
                    result = await self._search_analyze(arguments)
                elif tool_name == "search_health":
                    result = await self._search_health(arguments)
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                # Format response consistently
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2, default=str)
                            }
                        ]
                    }
                }
            
            elif method == "notifications/initialized":
                # Handle initialization notification
                logger.info("Server initialized successfully")
                return None
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
    
    async def _web_search(self, args: Dict) -> Dict:
        """Perform enhanced web search using Brave API with caching and validation."""
        if not self.brave_api_key:
            raise ValueError("Brave API key not configured")
        
        query = args.get("query", "").strip()
        if not query:
            raise ValueError("Search query cannot be empty")
        
        count = min(args.get("count", self.search_config['default_results']), 
                   self.search_config['max_results'])
        safesearch = args.get("safesearch", "moderate")
        freshness = args.get("freshness", "")
        country = args.get("country", "")
        
        # Check cache first
        cache_key = f"{query}:{count}:{safesearch}:{freshness}:{country}"
        if cache_key in self.search_cache:
            logger.info(f"Returning cached results for: {query}")
            cached_result = self.search_cache[cache_key].copy()
            cached_result["cached"] = True
            return cached_result
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key,
            "User-Agent": self.search_config['user_agent']
        }
        
        # Build parameters
        params = {"q": query, "count": count}
        if safesearch != "moderate":
            params["safesearch"] = safesearch
        if freshness:
            params["freshness"] = freshness
        if country:
            params["country"] = country
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=self.search_config['timeout']) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # Parse comprehensive results
                if "web" in data and "results" in data["web"]:
                    for result in data["web"]["results"]:
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
                
                # Extract metadata
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
                    "search_timestamp": datetime.utcnow().isoformat(),
                    "cached": False,
                    "search_params": {
                        "count": count,
                        "safesearch": safesearch,
                        "freshness": freshness,
                        "country": country
                    }
                }
                
                # Cache result
                self._cache_result(cache_key, search_result)
                
                logger.info(f"Web search completed: {len(results)} results for '{query}'")
                return search_result
                
        except ImportError:
            raise ValueError("httpx library required for web search. Install with: pip install httpx")
        except Exception as e:
            logger.error(f"Search API error: {e}")
            raise ValueError(f"Search API error: {str(e)}")
    
    async def _search_summarize(self, args: Dict) -> Dict:
        """Summarize search results with enhanced analysis."""
        query = args.get("query", "").strip()
        max_results = min(args.get("max_results", 5), 10)
        
        if not query:
            raise ValueError("Query required for summarization")
        
        # Perform search if not cached
        search_result = await self._web_search({"query": query, "count": max_results * 2})
        results = search_result.get("results", [])[:max_results]
        
        if not results:
            return {"error": "No results to summarize", "query": query}
        
        # Extract domains and topics
        domains = set()
        all_text = []
        
        for result in results:
            # Domain extraction
            if result.get('url'):
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(result['url']).netloc
                    domains.add(domain)
                except:
                    pass
            
            # Text aggregation
            text = f"{result.get('title', '')} {result.get('description', '')}"
            all_text.append(text)
        
        # Keyword extraction
        keywords = self._extract_keywords(' '.join(all_text))
        
        summary = {
            "query": query,
            "search_timestamp": search_result.get("search_timestamp"),
            "total_results": search_result.get("total_results", 0),
            "summarized_results": len(results),
            "top_domains": list(domains)[:5],
            "key_topics": keywords[:10],
            "results_preview": [
                {
                    "title": r.get('title', ''),
                    "url": r.get('url', ''),
                    "description": (r.get('description', '')[:200] + "...") 
                                 if len(r.get('description', '')) > 200 
                                 else r.get('description', '')
                }
                for r in results[:3]
            ],
            "summary_metadata": {
                "avg_title_length": sum(len(r.get('title', '')) for r in results) / len(results),
                "family_friendly_ratio": sum(1 for r in results if r.get('family_friendly', True)) / len(results),
                "has_age_info": sum(1 for r in results if r.get('age')) / len(results)
            }
        }
        
        return summary
    
    async def _search_analyze(self, args: Dict) -> Dict:
        """Analyze search patterns with detailed insights."""
        query = args.get("query", "").strip()
        include_domains = args.get("include_domains", True)
        include_keywords = args.get("include_keywords", True)
        
        if not query:
            raise ValueError("Query required for analysis")
        
        # Perform comprehensive search
        search_result = await self._web_search({"query": query, "count": 15})
        results = search_result.get("results", [])
        
        if not results:
            return {"error": "No results to analyze", "query": query}
        
        analysis = {
            "query": query,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "total_results": len(results),
            "search_metadata": search_result.get("search_params", {})
        }
        
        # Domain analysis
        if include_domains:
            domains = {}
            for result in results:
                if result.get('url'):
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(result['url']).netloc
                        domains[domain] = domains.get(domain, 0) + 1
                    except:
                        pass
            
            analysis["domain_analysis"] = {
                "unique_domains": len(domains),
                "domain_distribution": dict(sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]),
                "domain_diversity": len(domains) / len(results) if results else 0
            }
        
        # Keyword analysis
        if include_keywords:
            all_text = ' '.join([
                f"{r.get('title', '')} {r.get('description', '')}" 
                for r in results
            ])
            keywords = self._extract_keywords(all_text)
            
            from collections import Counter
            keyword_counts = Counter(keywords)
            
            analysis["keyword_analysis"] = {
                "total_keywords": len(keywords),
                "unique_keywords": len(set(keywords)),
                "top_keywords": [{"keyword": k, "count": c} for k, c in keyword_counts.most_common(15)],
                "keyword_diversity": len(set(keywords)) / len(keywords) if keywords else 0
            }
        
        # Quality metrics
        analysis["quality_metrics"] = {
            "has_description": sum(1 for r in results if r.get('description')),
            "has_age_info": sum(1 for r in results if r.get('age')),
            "family_friendly": sum(1 for r in results if r.get('family_friendly', True)),
            "avg_title_length": sum(len(r.get('title', '')) for r in results) / len(results),
            "avg_description_length": sum(len(r.get('description', '')) for r in results) / len(results)
        }
        
        return analysis
    
    async def _search_health(self, args: Dict) -> Dict:
        """Check search service health and configuration."""
        health_status = {
            "service": "brave-search-mcp",
            "version": self.server_info["version"],
            "timestamp": datetime.utcnow().isoformat(),
            "api_key_configured": bool(self.brave_api_key),
            "api_key_length": len(self.brave_api_key) if self.brave_api_key else 0,
            "cache_size": len(self.search_cache),
            "cache_max_size": self.cache_max_size,
            "available_tools": len(self.tools),
            "configuration": self.search_config.copy()
        }
        
        # Test API connectivity
        if self.brave_api_key:
            try:
                test_result = await self._web_search({"query": "test", "count": 1})
                health_status["api_connectivity"] = "healthy"
                health_status["last_test_results"] = test_result.get("total_results", 0)
            except Exception as e:
                health_status["api_connectivity"] = "error"
                health_status["api_error"] = str(e)
        else:
            health_status["api_connectivity"] = "no_api_key"
        
        return health_status
    
    def _cache_result(self, cache_key: str, result: Dict):
        """Cache search result with size management."""
        if len(self.search_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = result.copy()
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'how', 
            'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 
            'man', 'car', 'way', 'with', 'this', 'that', 'from', 'they', 'she', 
            'been', 'than', 'what', 'when', 'where', 'will', 'more', 'said', 'each',
            'about', 'would', 'there', 'their', 'other', 'after', 'first', 'well',
            'also', 'just', 'being', 'over', 'years', 'into', 'through', 'during'
        }
        
        keywords = [word for word in words 
                   if len(word) >= min_length and word not in stop_words]
        return keywords
    
    async def run(self):
        """Run the enhanced MCP server with improved error handling."""
        logger.info("Starting Brave Search MCP Server...")
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    logger.info("Server shutting down...")
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                request = json.loads(line)
                response = await self.handle_request(request)
                
                if response is not None:  # Handle initialization notification
                    print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                }
                print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    server = CustomSearchServer()
    asyncio.run(server.run())