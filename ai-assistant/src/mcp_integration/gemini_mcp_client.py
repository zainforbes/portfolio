import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path

# Import our MCP tools
from mcp_integration.mcp_client import MCPClient
from mcp_integration.filesystem_tools import FilesystemTools
from mcp_integration.search_tools import SearchTools
from mcp_integration.gmail_tools import GmailTools
from mcp_integration.calendar_tools import CalendarTools

class GeminiMCPClient:
    def __init__(self, api_key: str = None, brave_api_key: str = None):
        # Initialize Gemini
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize MCP
        self.mcp_client = MCPClient()
        self.filesystem_tools = FilesystemTools(self.mcp_client)
        self.search_tools = SearchTools(self.mcp_client, brave_api_key)
        self.gmail_tools = GmailTools(self.mcp_client)
        self.calendar_tools = CalendarTools(self.mcp_client)
        
        # Available tools for Gemini
        self.available_tools = {
            "search_web": self._search_web,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "list_emails": self._list_emails,
            "read_email": self._read_email,
            "send_email": self._send_email,
            "list_calendar_events": self._list_calendar_events,
            "create_calendar_event": self._create_calendar_event
        }
    
    async def initialize(self, allowed_directories: List[str] = None):
        """Initialize all MCP servers"""
        print("Initializing MCP servers...")
        
        # Initialize filesystem
        success = await self.filesystem_tools.start(allowed_directories)
        if success:
            print("✓ Filesystem server initialized")
        else:
            print("✗ Filesystem server failed to start")
        
        # Initialize search
        try:
            success = await self.search_tools.start()
            if success:
                print("✓ Search server initialized")
        except Exception as e:
            print(f"✗ Search server failed: {e}")
        
        # Initialize Gmail
        try:
            success = await self.gmail_tools.start()
            if success:
                print("✓ Gmail server initialized")
        except Exception as e:
            print(f"✗ Gmail server failed: {e}")
        
        # Initialize Calendar
        try:
            success = await self.calendar_tools.start()
            if success:
                print("✓ Calendar server initialized")
        except Exception as e:
            print(f"✗ Calendar server failed: {e}")
        
        print("MCP initialization complete!")
    
    def _create_tool_prompt(self) -> str:
        """Create a prompt describing available tools for Gemini"""
        tools_description = """
You have access to these tools via function calls:

FILESYSTEM TOOLS:
- read_file(filepath) - Read contents of a file
- write_file(filepath, content) - Write content to a file
- list_directory(dirpath) - List directory contents

SEARCH TOOLS:
- search_web(query, count=10) - Search the web

EMAIL TOOLS:
- list_emails(query="", max_results=10) - List Gmail messages
- read_email(message_id) - Read a specific email
- send_email(to, subject, body) - Send an email

CALENDAR TOOLS:
- list_calendar_events(days_ahead=7) - List upcoming calendar events
- create_calendar_event(title, start_time, end_time, description) - Create calendar event

When you need to use any of these tools, format your response as:
TOOL_CALL: tool_name(arguments)

For example:
TOOL_CALL: search_web("latest AI news")
TOOL_CALL: read_file("/path/to/file.txt")
TOOL_CALL: send_email("user@example.com", "Subject", "Email body")

Always explain what you're doing before calling tools and interpret the results for the user.
        """
        return tools_description
    
    async def chat(self, message: str, context: str = "") -> str:
        """Chat with Gemini using MCP tools"""
        # Prepare the full prompt
        full_prompt = f"""
{self._create_tool_prompt()}

Previous context: {context}

User message: {message}

Response:
        """
        
        # Get initial response from Gemini
        response = self.model.generate_content(full_prompt)
        response_text = response.text
        
        # Check if Gemini wants to use any tools
        if "TOOL_CALL:" in response_text:
            response_text = await self._process_tool_calls(response_text)
        
        return response_text
    
    async def _process_tool_calls(self, response_text: str) -> str:
        """Process tool calls in Gemini's response"""
        lines = response_text.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip().startswith("TOOL_CALL:"):
                # Extract the tool call
                tool_call = line.strip().replace("TOOL_CALL:", "").strip()
                
                try:
                    # Parse the tool call (simple parsing for now)
                    if "(" in tool_call and tool_call.endswith(")"):
                        tool_name = tool_call.split("(")[0].strip()
                        args_str = tool_call[tool_call.find("(")+1:-1]
                        
                        # Execute the tool call
                        result = await self._execute_tool(tool_name, args_str)
                        processed_lines.append(f"Tool Result: {result}")
                    else:
                        processed_lines.append(f"Invalid tool call format: {tool_call}")
                        
                except Exception as e:
                    processed_lines.append(f"Tool execution error: {str(e)}")
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    async def _execute_tool(self, tool_name: str, args_str: str) -> str:
        """Execute a specific tool call"""
        if tool_name not in self.available_tools:
            return f"Unknown tool: {tool_name}"
        
        try:
            # Simple argument parsing (you might want to make this more robust)
            args = self._parse_tool_args(args_str)
            result = await self.available_tools[tool_name](**args)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def _parse_tool_args(self, args_str: str) -> Dict:
        """Parse tool arguments from string"""
        args = {}
        
        # Handle simple cases
        if not args_str.strip():
            return args
        
        # Split by commas (basic parsing - you might want to improve this)
        parts = [part.strip() for part in args_str.split(',')]
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().strip('"\'')
                value = value.strip().strip('"\'')
                
                # Try to convert to appropriate type
                if value.isdigit():
                    args[key] = int(value)
                elif value.lower() in ['true', 'false']:
                    args[key] = value.lower() == 'true'
                else:
                    args[key] = value
            else:
                # Positional argument
                value = part.strip().strip('"\'')
                if len(args) == 0:
                    args['query'] = value  # Default for search
                elif 'filepath' not in args:
                    args['filepath'] = value  # Default for file operations
        
        return args
    
    # Tool implementation methods
    async def _search_web(self, query: str, count: int = 10) -> Dict:
        """Search the web"""
        return await self.search_tools.web_search(query, count)
    
    async def _read_file(self, filepath: str) -> str:
        """Read a file"""
        return await self.filesystem_tools.read_file(filepath)
    
    async def _write_file(self, filepath: str, content: str) -> bool:
        """Write to a file"""
        return await self.filesystem_tools.write_file(filepath, content)
    
    async def _list_directory(self, dirpath: str) -> List[Dict]:
        """List directory contents"""
        return await self.filesystem_tools.list_directory(dirpath)
    
    async def _list_emails(self, query: str = "", max_results: int = 10) -> Dict:
        """List emails"""
        return await self.gmail_tools.list_messages(query, max_results)
    
    async def _read_email(self, message_id: str) -> Dict:
        """Read an email"""
        return await self.gmail_tools.read_message(message_id)
    
    async def _send_email(self, to: str, subject: str, body: str) -> Dict:
        """Send an email"""
        return await self.gmail_tools.send_message(to, subject, body)
    
    async def _list_calendar_events(self, days_ahead: int = 7) -> Dict:
        """List calendar events"""
        return await self.calendar_tools.list_events(days_ahead)
    
    async def _create_calendar_event(self, title: str, start_time: str, end_time: str, description: str = "") -> Dict:
        """Create calendar event"""
        return await self.calendar_tools.create_event(title, start_time, end_time, description)
    
    async def cleanup(self):
        """Cleanup MCP connections"""
        await self.mcp_client.shutdown()