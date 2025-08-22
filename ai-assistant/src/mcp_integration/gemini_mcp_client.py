import asyncio
import sys
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText
import google.generativeai as genai

# Import MCP server implementations directly
# These will be used to provide MCP functionality without the old MCPClient layer

class GeminiMCPClient:
    def __init__(self, api_key: str = None, brave_api_key: str = None):
        # Initialize Gemini
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required")
        
        genai.configure(api_key=self.api_key)
        # Use a more commonly available Gemini model
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            # Fallback to other available models
            try:
                self.model = genai.GenerativeModel('gemini-pro')
            except:
                self.model = genai.GenerativeModel('models/gemini-pro')
        
        # Initialize SearchTools for Brave search integration
        self.search_tools = None
        
        # Google API credentials and services
        self.gmail_service = None
        self.calendar_service = None
        self.credentials = None
        
        # Gmail and Calendar API scopes
        self.SCOPES = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.send',
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/calendar.readonly',
            'https://www.googleapis.com/auth/calendar.events'
        ]
        
        # Initialize MCP functionality directly within the Gemini MCP client
        # We'll implement the MCP services internally instead of using separate tools
        
        # Available tools for Gemini
        self.available_tools = {
            "search_web": self._search_web,
            "brave_search": self._brave_search,
            "search_summarize": self._search_summarize,
            "search_analyze": self._search_analyze,
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
        """Initialize Gemini MCP client - now handles all MCP functionality internally"""
        print("Initializing Gemini MCP client...")
        
        # Initialize Gemini model
        try:
            # Test Gemini connection
            test_response = self.model.generate_content("Hello")
            print("[OK] Gemini connection established")
        except Exception as e:
            print(f"[FAIL] Gemini connection failed: {e}")
            raise
        
        # Initialize Google APIs
        await self._setup_google_credentials()
        
        # Initialize SearchTools
        await self._setup_search_tools()
        
        print("Gemini MCP client initialization complete!")
    
    async def _setup_search_tools(self):
        """Initialize search tools for Brave search integration"""
        try:
            from src.mcp_integration.search_tools import SearchTools
            self.search_tools = SearchTools(self)
            await self.search_tools.start()
            print("[OK] Search tools initialized")
        except Exception as e:
            print(f"[WARNING] Search tools initialization failed: {e}")
            self.search_tools = None
    
    async def _setup_google_credentials(self):
        """Set up Google API credentials for Gmail and Calendar"""
        try:
            # Load existing credentials
            token_path = "config/gmail_token.pickle"
            credentials_path = "config/credentials.json"
            
            creds = None
            
            # Check if token file exists
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            # If there are no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    # Refresh expired credentials
                    try:
                        creds.refresh(Request())
                        print("[OK] Google credentials refreshed")
                    except Exception as e:
                        print(f"[WARN] Credential refresh failed: {e}")
                        creds = None
                
                if not creds:
                    # Get new credentials
                    if os.path.exists(credentials_path):
                        flow = InstalledAppFlow.from_client_secrets_file(
                            credentials_path, self.SCOPES)
                        creds = flow.run_local_server(port=0)
                        print("[OK] New Google credentials obtained")
                    else:
                        print(f"[FAIL] Credentials file not found: {credentials_path}")
                        return
                
                # Save credentials for next run
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                    print("[OK] Google credentials saved")
            
            self.credentials = creds
            
            # Build Gmail and Calendar services
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            
            print("[OK] Gmail API service initialized")
            print("[OK] Calendar API service initialized")
            
        except Exception as e:
            print(f"[FAIL] Google API setup failed: {e}")
            print("Gmail and Calendar functionality will be limited")
    
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
        
        try:
            # Get initial response from Gemini
            response = self.model.generate_content(full_prompt)
            
            # Check if response has valid content
            if not response.candidates:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your request."
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason and finish_reason != 1:  # 1 is STOP (successful completion)
                    if finish_reason == 3:  # SAFETY
                        return "I cannot provide a response to that request due to safety guidelines."
                    elif finish_reason == 4:  # RECITATION
                        return "I cannot provide that response due to content policy restrictions."
                    elif finish_reason == 5:  # LENGTH
                        return "My response was cut short due to length limits. Please try asking a more specific question."
                    else:
                        # Handle unknown finish reasons (like 12)
                        return f"I encountered an issue generating a response (finish_reason: {finish_reason}). Please try again."
            
            # Try to get the text content safely
            try:
                response_text = response.text
            except Exception as text_error:
                # If response.text fails, try to extract from parts directly
                if candidate.content and candidate.content.parts:
                    response_text = ""
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                    if not response_text:
                        return "I generated a response but couldn't extract the text content. Please try again."
                else:
                    return f"I encountered an error accessing the response content: {str(text_error)}"
            
            # Check if Gemini wants to use any tools
            if "TOOL_CALL:" in response_text:
                response_text = await self._process_tool_calls(response_text)
            
            return response_text
            
        except Exception as e:
            print(f"[ERROR] Gemini chat error: {e}")
            return f"I encountered an error while processing your request: {str(e)}. Please try again."
    
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
            
            # Format result appropriately
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return str(result)
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
    
    # MCP Service Implementations
    async def _search_web(self, query: str, count: int = 10) -> Dict[str, Any]:
        """Search web using Brave Search API"""
        try:
            if not self.search_tools:
                return {"error": "Search tools not initialized"}
            
            result = await self.search_tools.web_search(query, count)
            return result
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    async def _brave_search(self, query: str, count: int = 10, **kwargs) -> Dict[str, Any]:
        """Enhanced Brave search with additional parameters"""
        try:
            if not self.search_tools:
                return {"error": "Search tools not initialized"}
            
            result = await self.search_tools.brave_search(query, count, **kwargs)
            return result
        except Exception as e:
            return {"error": f"Brave search failed: {str(e)}"}
    
    async def _search_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Summarize search results"""
        try:
            if not self.search_tools:
                return {"error": "Search tools not initialized"}
            
            # First get search results
            search_results = await self.search_tools.web_search(query, max_results * 2)
            # Then summarize them
            summary = await self.search_tools.search_summarize(search_results, max_results)
            return summary
        except Exception as e:
            return {"error": f"Search summarization failed: {str(e)}"}
    
    async def _search_analyze(self, query: str, include_domains: bool = True, include_keywords: bool = True) -> Dict[str, Any]:
        """Analyze search patterns and results"""
        try:
            if not self.search_tools:
                return {"error": "Search tools not initialized"}
            
            # First get search results
            search_results = await self.search_tools.web_search(query, 15)
            # Then analyze them
            analysis = await self.search_tools.search_analyze(search_results)
            return analysis
        except Exception as e:
            return {"error": f"Search analysis failed: {str(e)}"}
    
    async def _read_file(self, filepath: str) -> str:
        """Read file from filesystem"""
        try:
            import os
            from pathlib import Path
            
            # Security check - only allow reading files in the project directory
            project_root = Path(__file__).parent.parent.parent
            file_path = Path(filepath)
            
            # If relative path, make it relative to project root
            if not file_path.is_absolute():
                file_path = project_root / file_path
            
            # Security: Ensure path is within project directory
            try:
                file_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                return f"Access denied: File {filepath} is outside the project directory"
            
            # Check if file exists
            if not file_path.exists():
                return f"File not found: {filepath}"
            
            if not file_path.is_file():
                return f"Path is not a file: {filepath}"
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return f"File content from {filepath}:\n\n{content}"
                
        except Exception as e:
            return f"File reading failed: {str(e)}"
    
    async def _write_file(self, filepath: str, content: str) -> str:
        """Write file to filesystem"""
        try:
            import os
            from pathlib import Path
            
            # Security check - only allow writing files in the project directory
            project_root = Path(__file__).parent.parent.parent
            file_path = Path(filepath)
            
            # If relative path, make it relative to project root
            if not file_path.is_absolute():
                file_path = project_root / file_path
            
            # Security: Ensure path is within project directory
            try:
                file_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                return f"Access denied: File {filepath} is outside the project directory"
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"File successfully written to {filepath} ({len(content)} characters)"
                
        except Exception as e:
            return f"File writing failed: {str(e)}"
    
    async def _list_directory(self, dirpath: str) -> str:
        """List directory contents"""
        try:
            import os
            from pathlib import Path
            
            # Security check - only allow listing directories in the project directory
            project_root = Path(__file__).parent.parent.parent
            dir_path = Path(dirpath)
            
            # If relative path, make it relative to project root
            if not dir_path.is_absolute():
                dir_path = project_root / dir_path
            
            # Security: Ensure path is within project directory
            try:
                dir_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                return f"Access denied: Directory {dirpath} is outside the project directory"
            
            # Check if directory exists
            if not dir_path.exists():
                return f"Directory not found: {dirpath}"
            
            if not dir_path.is_dir():
                return f"Path is not a directory: {dirpath}"
            
            # List directory contents
            contents = []
            for item in sorted(dir_path.iterdir()):
                item_type = "DIR" if item.is_dir() else "FILE"
                size = ""
                if item.is_file():
                    try:
                        size = f" ({item.stat().st_size} bytes)"
                    except:
                        size = ""
                contents.append(f"  {item_type}: {item.name}{size}")
            
            if not contents:
                return f"Directory {dirpath} is empty"
            
            return f"Directory listing for {dirpath}:\n\n" + "\n".join(contents)
                
        except Exception as e:
            return f"Directory listing failed: {str(e)}"
    
    async def _list_emails(self, query: str = "", max_results: int = 10) -> Dict[str, Any]:
        """List Gmail emails using Gmail API"""
        try:
            if not self.gmail_service:
                return {"error": "Gmail service not initialized. Please check authentication."}
            
            # Search for messages
            search_query = query if query else "is:unread"
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=search_query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return {"messages": [], "count": 0, "query": search_query}
            
            # Get details for each message
            email_list = []
            for msg in messages:
                msg_detail = self.gmail_service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                
                headers = {h['name'].lower(): h['value'] for h in msg_detail.get('payload', {}).get('headers', [])}
                
                email_list.append({
                    'id': msg['id'],
                    'subject': headers.get('subject', 'No Subject'),
                    'from': headers.get('from', 'Unknown Sender'),
                    'date': headers.get('date', ''),
                    'snippet': msg_detail.get('snippet', '')[:100] + '...'
                })
            
            return {
                "messages": email_list,
                "count": len(email_list),
                "query": search_query
            }
            
        except HttpError as e:
            error_msg = f"Gmail API error: {e}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Email listing failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
    
    async def _read_email(self, message_id: str) -> Dict[str, Any]:
        """Read specific Gmail email using Gmail API"""
        try:
            if not self.gmail_service:
                return {"error": "Gmail service not initialized. Please check authentication."}
            
            # Get the message
            message = self.gmail_service.users().messages().get(
                userId='me',
                id=message_id
            ).execute()
            
            # Extract headers
            headers = {}
            if 'payload' in message:
                for header in message['payload'].get('headers', []):
                    headers[header['name'].lower()] = header['value']
            
            # Extract body
            body = ""
            if 'payload' in message:
                body = self._extract_email_body(message['payload'])
            
            return {
                'id': message_id,
                'subject': headers.get('subject', 'No Subject'),
                'from': headers.get('from', 'Unknown Sender'),
                'to': headers.get('to', ''),
                'date': headers.get('date', ''),
                'body': body,
                'snippet': message.get('snippet', ''),
                'thread_id': message.get('threadId', '')
            }
            
        except HttpError as e:
            error_msg = f"Gmail API error: {e}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Email reading failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
    
    def _extract_email_body(self, payload) -> str:
        """Extract email body from Gmail API payload"""
        body = ""
        
        if 'parts' in payload:
            # Multi-part message
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
                elif part['mimeType'] == 'text/html':
                    if 'data' in part['body'] and not body:  # Fallback to HTML if no plain text
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        else:
            # Single part message
            if payload.get('mimeType') == 'text/plain' and 'data' in payload.get('body', {}):
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
            elif payload.get('mimeType') == 'text/html' and 'data' in payload.get('body', {}):
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body
    
    async def _send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Send Gmail email using Gmail API"""
        try:
            if not self.gmail_service:
                return {"error": "Gmail service not initialized. Please check authentication."}
            
            # Create message
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            # Send message
            send_result = self.gmail_service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            return {
                "message_id": send_result.get('id'),
                "thread_id": send_result.get('threadId'),
                "to": to,
                "subject": subject,
                "status": "sent"
            }
            
        except HttpError as e:
            error_msg = f"Gmail API error: {e}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Email sending failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
    
    async def _list_calendar_events(self, days_ahead: int = 7) -> Dict[str, Any]:
        """List calendar events using Calendar API"""
        try:
            if not self.calendar_service:
                return {"error": "Calendar service not initialized. Please check authentication."}
            
            # Calculate time range
            now = datetime.utcnow()
            time_min = now.isoformat() + 'Z'
            time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'
            
            # Get events
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=20,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return {"events": [], "count": 0, "days_ahead": days_ahead}
            
            # Format events
            event_list = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                event_list.append({
                    'id': event.get('id'),
                    'summary': event.get('summary', 'No Title'),
                    'description': event.get('description', ''),
                    'start': start,
                    'end': end,
                    'location': event.get('location', ''),
                    'attendees': [a.get('email') for a in event.get('attendees', [])],
                    'creator': event.get('creator', {}).get('email', ''),
                    'html_link': event.get('htmlLink', '')
                })
            
            return {
                "events": event_list,
                "count": len(event_list),
                "days_ahead": days_ahead,
                "time_range": {
                    "start": time_min,
                    "end": time_max
                }
            }
            
        except HttpError as e:
            error_msg = f"Calendar API error: {e}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Calendar listing failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
    
    async def _create_calendar_event(self, title: str, start_time: str, end_time: str, description: str = "") -> Dict[str, Any]:
        """Create calendar event using Calendar API"""
        try:
            if not self.calendar_service:
                return {"error": "Calendar service not initialized. Please check authentication."}
            
            # Parse time strings - assuming ISO format or basic formats
            try:
                # Try to parse various time formats
                if 'T' not in start_time:
                    # Assume it's a simple time like "2 PM" or "14:00"
                    # For now, use today's date with the specified time
                    today = datetime.now().date()
                    start_time = f"{today}T{self._parse_time_string(start_time)}"
                    end_time = f"{today}T{self._parse_time_string(end_time)}"
                
                # Ensure timezone info
                if start_time.endswith('Z') or '+' in start_time[-6:]:
                    pass  # Already has timezone
                else:
                    start_time += ':00'
                    end_time += ':00'
            except Exception as e:
                return {"error": f"Time parsing error: {str(e)}. Please use ISO format (YYYY-MM-DDTHH:MM:SS) or simple time (2 PM, 14:00)."}
            
            # Create event object
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'America/New_York',  # Adjust timezone as needed
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': 'America/New_York',  # Adjust timezone as needed
                },
            }
            
            # Create the event
            created_event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return {
                "event_id": created_event.get('id'),
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
                "html_link": created_event.get('htmlLink'),
                "status": "created"
            }
            
        except HttpError as e:
            error_msg = f"Calendar API error: {e}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Calendar event creation failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {"error": error_msg}
    
    def _parse_time_string(self, time_str: str) -> str:
        """Parse various time string formats to HH:MM format"""
        time_str = time_str.lower().strip()
        
        # Handle PM/AM format
        if 'pm' in time_str:
            hour = int(time_str.replace('pm', '').strip())
            if hour != 12:
                hour += 12
            return f"{hour:02d}:00"
        elif 'am' in time_str:
            hour = int(time_str.replace('am', '').strip())
            if hour == 12:
                hour = 0
            return f"{hour:02d}:00"
        elif ':' in time_str:
            # Already in HH:MM format
            return time_str
        else:
            # Assume it's just an hour
            try:
                hour = int(time_str)
                return f"{hour:02d}:00"
            except ValueError:
                return "12:00"  # Default fallback
    
    async def cleanup(self):
        """Cleanup Gemini MCP client"""
        print("Gemini MCP client cleanup complete")