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

class GmailTools:
    def __init__(self, mcp_client: GeminiMCPClient):
        self.client = mcp_client
        self.server_name = "gmail"
    
    async def start(self):
        """Initialize Gmail tools via GeminiMCPClient"""
        # Gmail functionality is built into GeminiMCPClient, so just verify it's initialized
        try:
            await self.client.initialize()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize gmail tools: {e}")
            return False
    
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
        result = await self.client._list_emails(query, max_results)
        return result
    
    async def read_message(self, message_id: str) -> Dict:
        """Read a Gmail message"""
        result = await self.client._read_email(message_id)
        return result
    
    async def send_message(self, to: str, subject: str, body: str) -> Dict:
        """Send a Gmail message"""
        result = await self.client._send_email(to, subject, body)
        return result