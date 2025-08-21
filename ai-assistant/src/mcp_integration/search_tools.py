# src/mcp_integration/search_tools.py
import os
import sys
import json
from pathlib import Path
from .mcp_client import MCPClient
from typing import List, Dict

class SearchTools:
    def __init__(self, mcp_client: MCPClient, api_key: str = None):
        self.client = mcp_client
        self.server_name = "search"
        self.api_key = api_key or os.getenv('BRAVE_API_KEY')
    
    async def start(self):
        """Start custom search MCP server"""
        if not self.api_key:
            raise ValueError("Brave API key required")
        
        # Path to our custom search server
        search_server_path = Path(__file__).parent / "custom_search_server.py"
        
        command = [sys.executable, str(search_server_path)]
        
        # Set environment variable for API key
        env = os.environ.copy()
        env['BRAVE_API_KEY'] = self.api_key
        
        success = await self.client.start_server(
            self.server_name, 
            command,
            {'env': env}
        )
        
        if success:
            await self.client.list_tools(self.server_name)
        return success
    
    def _extract_search_results(self, response: Dict) -> List[Dict]:
        """Extract search results from MCP response"""
        if 'result' not in response:
            return []
        
        result = response['result']
        
        # Handle MCP content format
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            if isinstance(content, list) and len(content) > 0:
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        try:
                            search_data = json.loads(item.get('text', '{}'))
                            return search_data.get('results', [])
                        except json.JSONDecodeError:
                            pass
        
        return []
    
    async def web_search(self, query: str, count: int = 10) -> List[Dict]:
        """Perform web search"""
        result = await self.client.call_tool(
            self.server_name,
            "web_search",
            {"query": query, "count": count}
        )
        
        return self._extract_search_results(result)