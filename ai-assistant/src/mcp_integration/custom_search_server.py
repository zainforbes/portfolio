# src/mcp_integration/custom_search_server.py
import asyncio
import json
import sys
import os
import requests
from typing import Dict, List, Any

class CustomSearchServer:
    def __init__(self, brave_api_key: str = None):
        self.brave_api_key = brave_api_key or os.getenv('BRAVE_API_KEY')
        self.tools = [
            {
                "name": "web_search",
                "description": "Search the web using Brave Search API",
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
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle MCP JSON-RPC requests"""
        method = request.get("method")
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "custom-brave-search",
                            "version": "1.0.0"
                        }
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
                
                if tool_name == "web_search":
                    result = await self._web_search(arguments)
                    # Format response like the filesystem server
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2)
                                }
                            ]
                        }
                    }
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"}
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
    
    async def _web_search(self, args: Dict) -> Dict:
        """Perform web search using Brave API"""
        if not self.brave_api_key:
            raise ValueError("Brave API key not configured")
        
        query = args.get("query", "")
        count = min(args.get("count", 10), 20)
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }
        params = {"q": query, "count": count}
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(url, headers=headers, params=params, timeout=10)
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "published": result.get("age", "")
                    })
            
            return {
                "query": query,
                "results": results,
                "total_results": len(results)
            }
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Search API error: {str(e)}")
    
    async def run(self):
        """Run the MCP server"""
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
                }
                print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    server = CustomSearchServer()
    asyncio.run(server.run())