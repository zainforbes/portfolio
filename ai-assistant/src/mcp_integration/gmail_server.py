import asyncio
import json
import sys
import os
import base64
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from typing import Dict, List, Any

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

class GmailServer:
    def __init__(self):
        self.service = None
        self.tools = [
            {
                "name": "list_messages",
                "description": "List Gmail messages",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Gmail search query", "default": ""},
                        "max_results": {"type": "integer", "description": "Maximum results", "default": 10}
                    }
                }
            },
            {
                "name": "read_message",
                "description": "Read a specific Gmail message",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string", "description": "Gmail message ID"}
                    },
                    "required": ["message_id"]
                }
            },
            {
                "name": "send_message",
                "description": "Send a Gmail message",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"}
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        ]
    
    async def _authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        token_path = 'data/token.json'
        credentials_path = 'secrets/oauth_credentials.json'
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    raise ValueError("Gmail credentials.json file not found")
                
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = await asyncio.get_event_loop().run_in_executor(
                    None, flow.run_local_server, {'port': 0}
                )
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
    
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
                        "serverInfo": {"name": "gmail-server", "version": "1.0.0"}
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
                
                if tool_name == "list_messages":
                    result = await self._list_messages(arguments)
                elif tool_name == "read_message":
                    result = await self._read_message(arguments)
                elif tool_name == "send_message":
                    result = await self._send_message(arguments)
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
    
    async def _list_messages(self, args: Dict) -> Dict:
        """List Gmail messages"""
        query = args.get("query", "")
        max_results = args.get("max_results", 10)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results
            ).execute()
        )
        
        messages = result.get('messages', [])
        message_list = []
        
        for msg in messages:
            # Get basic info for each message
            msg_detail = await loop.run_in_executor(
                None,
                lambda: self.service.users().messages().get(
                    userId='me', id=msg['id'], format='metadata',
                    metadataHeaders=['From', 'Subject', 'Date']
                ).execute()
            )
            
            headers = {h['name']: h['value'] for h in msg_detail.get('payload', {}).get('headers', [])}
            
            message_list.append({
                "id": msg['id'],
                "from": headers.get('From', ''),
                "subject": headers.get('Subject', ''),
                "date": headers.get('Date', '')
            })
        
        return {"messages": message_list, "total": len(message_list)}
    
    async def _read_message(self, args: Dict) -> Dict:
        """Read a specific Gmail message"""
        message_id = args.get("message_id")
        
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: self.service.users().messages().get(
                userId='me', id=message_id, format='full'
            ).execute()
        )
        
        headers = {h['name']: h['value'] for h in message.get('payload', {}).get('headers', [])}
        
        # Extract body
        body = self._extract_message_body(message.get('payload', {}))
        
        return {
            "id": message_id,
            "from": headers.get('From', ''),
            "to": headers.get('To', ''),
            "subject": headers.get('Subject', ''),
            "date": headers.get('Date', ''),
            "body": body
        }
    
    def _extract_message_body(self, payload):
        """Extract message body from Gmail payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    break
        elif payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body
    
    async def _send_message(self, args: Dict) -> Dict:
        """Send a Gmail message"""
        to = args.get("to")
        subject = args.get("subject")
        body = args.get("body")
        
        message = f"To: {to}\nSubject: {subject}\n\n{body}"
        raw = base64.urlsafe_b64encode(message.encode()).decode()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.service.users().messages().send(
                userId='me', body={'raw': raw}
            ).execute()
        )
        
        return {"message_id": result['id'], "status": "sent"}
    
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
    server = GmailServer()
    asyncio.run(server.run())