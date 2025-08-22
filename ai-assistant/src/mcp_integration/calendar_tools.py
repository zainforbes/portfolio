import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
import sys
import os
import base64

from src.mcp_integration.mcp_client import MCPClient


class CalendarTools:
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
        self.server_name = "calendar"
    
    async def start(self):
        """Start Calendar MCP server"""
        calendar_server_path = Path(__file__).parent / "calendar_server.py"
        command = [sys.executable, str(calendar_server_path)]
        
        success = await self.client.start_server(self.server_name, command)
        if success:
            await self.client.list_tools(self.server_name)
        return success
    
    def _extract_calendar_response(self, response: Dict) -> Dict:
        """Extract Calendar response from MCP format"""
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
    
    async def list_events(self, days_ahead: int = 7) -> Dict:
        """List calendar events"""
        result = await self.client.call_tool(
            self.server_name,
            "list_events",
            {"days_ahead": days_ahead}
        )
        return self._extract_calendar_response(result)
    
    async def create_event(self, title: str, start_time: str, end_time: str, description: str = "") -> Dict:
        """Create a calendar event"""
        result = await self.client.call_tool(
            self.server_name,
            "create_event",
            {
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "description": description
            }
        )
        return self._extract_calendar_response(result)
