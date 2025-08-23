# src/mcp_integration/calendar_server.py
import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from typing import Dict, List, Any

SCOPES = ['https://www.googleapis.com/auth/calendar']

class CalendarServer:
    def __init__(self):
        self.service = None
        self.tools = [
            {
                "name": "list_events",
                "description": "List Calendar events",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days_ahead": {"type": "integer", "description": "Days to look ahead", "default": 7},
                        "max_results": {"type": "integer", "description": "Maximum results", "default": 10}
                    }
                }
            },
            {
                "name": "create_event",
                "description": "Create a Calendar event",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title"},
                        "start_time": {"type": "string", "description": "Start time (ISO format)"},
                        "end_time": {"type": "string", "description": "End time (ISO format)"},
                        "description": {"type": "string", "description": "Event description", "default": ""}
                    },
                    "required": ["title", "start_time", "end_time"]
                }
            }
        ]
    
    async def _authenticate(self):
        """Authenticate with Calendar API"""
        creds = None
        # Ensure config directory exists
        os.makedirs('config', exist_ok=True)
        token_path = 'config/calendar_token.json'
        credentials_path = os.getenv("GOOGLE_CLIENT_SECRET") or 'config/oauth_credentials.json'
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    raise ValueError("Calendar credentials.json file not found")
                
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: flow.run_local_server(port=0)
                )
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('calendar', 'v3', credentials=creds)
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle MCP JSON-RPC requests"""
        method = request.get("method")
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                await self._authenticate()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "calendar-server", "version": "1.0.0"}
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
                
                if tool_name == "list_events":
                    result = await self._list_events(arguments)
                elif tool_name == "create_event":
                    result = await self._create_event(arguments)
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                    }
                }
            
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
    
    async def _list_events(self, args: Dict) -> Dict:
        """List Calendar events"""
        days_ahead = args.get("days_ahead", 7)
        max_results = args.get("max_results", 10)
        
        now = datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'
        
        loop = asyncio.get_event_loop()
        events_result = await loop.run_in_executor(
            None,
            lambda: self.service.events().list(
                calendarId='primary', 
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results, 
                singleEvents=True,
                orderBy='startTime'
            ).execute()
        )
        
        events = events_result.get('items', [])
        event_list = []
        
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            event_list.append({
                "id": event['id'],
                "title": event.get('summary', ''),
                "start_time": start,
                "end_time": end,
                "description": event.get('description', ''),
                "location": event.get('location', '')
            })
        
        return {"events": event_list, "total": len(event_list)}
    
    async def _create_event(self, args: Dict) -> Dict:
        """Create a Calendar event"""
        title = args.get("title")
        start_time = args.get("start_time")
        end_time = args.get("end_time")
        description = args.get("description", "")
        
        event = {
            'summary': title,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.service.events().insert(
                calendarId='primary', body=event
            ).execute()
        )
        
        return {
            "event_id": result['id'],
            "status": "created",
            "html_link": result.get('htmlLink', '')
        }
    
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
    server = CalendarServer()
    asyncio.run(server.run())