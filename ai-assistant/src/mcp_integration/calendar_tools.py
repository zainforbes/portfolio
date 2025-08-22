import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
import sys
import os
import base64

from src.mcp_integration.gemini_mcp_client import GeminiMCPClient


class CalendarTools:
    def __init__(self, mcp_client: GeminiMCPClient):
        self.client = mcp_client
        self.server_name = "calendar"
    
    async def start(self):
        """Initialize Calendar tools via GeminiMCPClient"""
        # Calendar functionality is built into GeminiMCPClient, so just verify it's initialized
        try:
            await self.client.initialize()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize calendar tools: {e}")
            return False
    
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
        result = await self.client._list_calendar_events(days_ahead)
        return result
    
    async def create_event(self, title: str, start_time: str, end_time: str, description: str = "") -> Dict:
        """Create a calendar event"""
        result = await self.client._create_calendar_event(title, start_time, end_time, description)
        return result
