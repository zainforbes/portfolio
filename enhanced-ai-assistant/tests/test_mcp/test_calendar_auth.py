#!/usr/bin/env python3
"""
Calendar authentication test using the same credentials as Gmail.
Tests both real Google Calendar API authentication and the MCP server.
"""

import asyncio
import os
import sys
import pickle
from pathlib import Path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add src to path - go up two levels from tests/test_mcp to reach src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Load environment variables
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar"]

class CalendarClient:
    def __init__(self):
        self.creds = None
        self.token_path = "config/calendar_token.pickle"
        self.secret_file = os.getenv("GOOGLE_CLIENT_SECRET") or "config/credentials.json"

        # Ensure config directory exists
        os.makedirs("config", exist_ok=True)
        
        # Load saved token
        if os.path.exists(self.token_path):
            with open(self.token_path, "rb") as token:
                self.creds = pickle.load(token)

        # If no creds, go through auth flow
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.secret_file, SCOPES
                )
                self.creds = flow.run_local_server(port=0)

            # Save token for next time
            with open(self.token_path, "wb") as token:
                pickle.dump(self.creds, token)

        # Build Calendar service
        self.service = build("calendar", "v3", credentials=self.creds)

    def list_events(self, max_results=5, days_ahead=7):
        """List upcoming calendar events."""
        now = datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'
        
        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        return events_result.get('items', [])
    
    def get_calendar_list(self):
        """Get list of user's calendars."""
        calendar_list = self.service.calendarList().list().execute()
        return calendar_list.get('items', [])
    
    def create_event(self, title, start_time, end_time, description=""):
        """Create a new calendar event."""
        event = {
            'summary': title,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }
        
        result = self.service.events().insert(
            calendarId='primary', body=event
        ).execute()
        
        return result

from src.mcp_integration.calendar_server import CalendarServer

async def test_real_calendar_client():
    """Test the real Calendar client with actual authentication"""
    print("Testing Real Calendar Client...")
    
    try:
        print("Initializing Calendar client with OAuth...")
        calendar_client = CalendarClient()
        
        print("SUCCESS: Calendar authentication successful!")
        
        # Test calendar list
        print("Testing calendar list...")
        calendars = calendar_client.get_calendar_list()
        print(f"Found {len(calendars)} calendars")
        
        if calendars:
            primary_calendar = next((cal for cal in calendars if cal.get('primary')), calendars[0])
            print(f"Primary calendar: {primary_calendar.get('summary', 'Unknown')}")
        
        # Test event listing
        print("Testing event listing...")
        events = calendar_client.list_events(max_results=3)
        print(f"Found {len(events)} upcoming events")
        
        if events:
            print("First few events:")
            for event in events[:3]:
                start = event['start'].get('dateTime', event['start'].get('date'))
                title = event.get('summary', 'No title')
                print(f"   - {title} at {start}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error with real Calendar client: {e}")
        return False

async def test_mcp_calendar_server():
    """Test the MCP Calendar server"""
    print("\nTesting MCP Calendar Server...")
    
    try:
        calendar_server = CalendarServer()
        
        # Test initialization (which includes authentication)
        print("Testing MCP Calendar initialization...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        init_response = await calendar_server.handle_request(init_request)
        print(f"Init response: {init_response.get('result', {}).get('serverInfo', 'No server info')}")
        
        if 'error' in init_response:
            print(f"ERROR: MCP Calendar initialization failed: {init_response['error']}")
            return False
        
        print("SUCCESS: MCP Calendar initialization successful!")
        
        # Test tools list
        print("Testing MCP tools list...")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        tools_response = await calendar_server.handle_request(tools_request)
        tools = tools_response.get('result', {}).get('tools', [])
        print(f"Available tools: {[tool['name'] for tool in tools]}")
        
        # Test list events
        print("Testing MCP list events...")
        list_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "list_events",
                "arguments": {"days_ahead": 7, "max_results": 3}
            }
        }
        
        list_response = await calendar_server.handle_request(list_request)
        if 'error' in list_response:
            print(f"ERROR: List events failed: {list_response['error']}")
            return False
        
        print("SUCCESS: MCP list events working!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error testing MCP Calendar server: {e}")
        return False

async def main():
    """Run all Calendar tests"""
    print("Testing Calendar Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Real Calendar client with actual authentication
    real_client_test = await test_real_calendar_client()
    
    # Test 2: MCP Calendar server
    mcp_server_test = await test_mcp_calendar_server()
    
    print("\n" + "=" * 50)
    print("Stats: Test Results:")
    print(f"Real Calendar Client: {'SUCCESS: PASS' if real_client_test else 'ERROR: FAIL'}")
    print(f"MCP Calendar Server: {'SUCCESS: PASS' if mcp_server_test else 'ERROR: FAIL'}")
    
    if real_client_test and mcp_server_test:
        print("\nSUCCESS: All Calendar tests passed!")
        print("   SUCCESS: Real authentication working")
        print("   SUCCESS: MCP server implementation ready")
    elif real_client_test:
        print("\nWARNING: Real Calendar authentication works, but MCP server has issues.")
    elif mcp_server_test:
        print("\nWARNING: MCP server works, but real authentication failed.")
        print("   Check your Google credentials and setup.")
    else:
        print("\nERROR: All tests failed. Check your setup:")
        print("   - GOOGLE_CLIENT_SECRET environment variable")
        print("   - Google credentials file")
        print("   - Calendar API enabled in Google Cloud Console")
    
    print("\nCleanup: Test completed!")

if __name__ == "__main__":
    # Set up environment
    from dotenv import load_dotenv
    load_dotenv()
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'MISSING'}")
    credentials_path = os.getenv("GOOGLE_CLIENT_SECRET") or "config/oauth_credentials.json"
    print(f"Calendar credentials: {'FOUND' if os.path.exists(credentials_path) else 'MISSING'}")
    print(f"Google Client Secret: {'SET' if os.getenv('GOOGLE_CLIENT_SECRET') else 'MISSING'}")
    
    # Run tests
    asyncio.run(main())