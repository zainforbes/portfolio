import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
import sys
import os
import base64

from mcp_integration.mcp_client import MCPClient

class GmailTools:
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
        self.server_name = "gmail"
    
    async def start(self):
        """Start Gmail MCP server"""
        gmail_server_path = Path(__file__).parent / "gmail_server.py"
        command = [sys.executable, str(gmail_server_path)]
        
        success = await self.client.start_server(self.server_name, command)
        if success:
            await self.client.list_tools(self.server_name)
        return success
    
    def _extract_gmail_response(self, response: Dict) -> Dict:
        """Extract Gmail response from MCP format"""
        if 'result' not in response:
            return {}
        
        result = response['result']
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            if isinstance(content, list) and len(content) > 0:
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        try:
                            return json.loads(item.get('text', '{}'))
                        except json.JSONDecodeError:
                            pass
        return {}
    
    async def list_messages(self, query: str = "", max_results: int = 10) -> Dict:
        """List Gmail messages"""
        result = await self.client.call_tool(
            self.server_name,
            "list_messages",
            {"query": query, "max_results": max_results}
        )
        return self._extract_gmail_response(result)
    
    async def read_message(self, message_id: str) -> Dict:
        """Read a Gmail message"""
        result = await self.client.call_tool(
            self.server_name,
            "read_message",
            {"message_id": message_id}
        )
        return self._extract_gmail_response(result)
    
    async def send_message(self, to: str, subject: str, body: str) -> Dict:
        """Send a Gmail message"""
        result = await self.client.call_tool(
            self.server_name,
            "send_message",
            {"to": to, "subject": subject, "body": body}
        )
        return self._extract_gmail_response(result)