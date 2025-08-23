#!/usr/bin/env python3
"""
Gmail authentication test using the provided reference code.
Tests both real Gmail API authentication and the MCP server.
"""

import asyncio
import os
import sys
import pickle
from pathlib import Path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# Add src to path - go up two levels from tests/test_mcp to reach src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Load environment variables
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

class GmailClient:
    def __init__(self):
        self.creds = None
        self.token_path = "config/gmail_token.pickle"
        self.secret_file = os.getenv("GOOGLE_CLIENT_SECRET") or "secrets/oauth_credentials.json"

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

        # Build Gmail service
        self.service = build("gmail", "v1", credentials=self.creds)

    def list_messages(self, max_results=5):
        """Fetch message metadata (IDs only for now)."""
        results = (
            self.service.users()
            .messages()
            .list(userId="me", maxResults=max_results)
            .execute()
        )
        return results.get("messages", [])
    
    def get_message_details(self, message_id: str):
        msg = self.service.users().messages().get(
            userId="me", id=message_id, format="metadata", metadataHeaders=["Subject"]
        ).execute()
        headers = msg.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No subject")
        snippet = msg.get("snippet", "")
        return f"Email: {subject} - {snippet[:80]}..."

from mcp_integration.gmail_server import GmailMCPServer

async def test_real_gmail_client():
    """Test the real Gmail client with actual authentication"""
    print("Testing Real Gmail Client...")
    
    try:
        print("Initializing Gmail client with OAuth...")
        gmail_client = GmailClient()
        
        print("SUCCESS: Gmail authentication successful!")
        
        # Test basic operations
        print("Testing message listing...")
        messages = gmail_client.list_messages(max_results=3)
        print(f"Found {len(messages)} messages")
        
        if messages:
            print("Getting details for first message...")
            first_msg = gmail_client.get_message_details(messages[0]['id'])
            print(f"   {first_msg}")
        
        # Test profile
        print("Getting Gmail profile...")
        profile = gmail_client.service.users().getProfile(userId='me').execute()
        print(f"Email: {profile['emailAddress']}")
        print(f"Total messages: {profile['messagesTotal']}")
        print(f"Total threads: {profile['threadsTotal']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error with real Gmail client: {e}")
        return False

async def test_mcp_gmail_server():
    """Test the MCP Gmail server (mock implementation)"""
    print("\nTesting MCP Gmail Server...")
    
    try:
        gmail_server = GmailMCPServer()
        
        # Test authentication (mock)
        print("Email: Testing MCP authentication...")
        auth_result = await gmail_server.authenticate("dummy_credentials.json")
        print(f"Auth result: {auth_result}")
        
        if auth_result.get('success'):
            print("SUCCESS: MCP authentication successful (mock)!")
            
            # Test listing messages (mock)
            print("Messages: Testing MCP message listing...")
            messages = await gmail_server.list_messages(max_results=3)
            print(f"Messages: Found {len(messages.get('messages', []))} mock messages")
            
            # Test getting a specific message (mock)
            if messages.get('messages'):
                print("Email: Testing MCP get message...")
                msg_id = messages['messages'][0]['id']
                message = await gmail_server.get_message(msg_id)
                print(f"   Subject: {message.get('subject', 'No subject')}")
                print(f"   From: {message.get('from', 'Unknown')}")
            
            # Test profile (mock)
            print("Profile: Testing MCP profile...")
            profile = await gmail_server.get_profile()
            print(f"Email: {profile.get('emailAddress', 'Unknown')}")
            print(f"Stats: Total messages: {profile.get('messagesTotal', 0)}")
            
            return True
        else:
            print("ERROR: MCP authentication failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Error testing MCP Gmail server: {e}")
        return False

async def main():
    """Run all Gmail tests"""
    print("Testing Gmail Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Real Gmail client with actual authentication
    real_client_test = await test_real_gmail_client()
    
    # Test 2: MCP Gmail server (mock implementation)
    mcp_server_test = await test_mcp_gmail_server()
    
    print("\n" + "=" * 50)
    print("Stats: Test Results:")
    print(f"Real Gmail Client: {'SUCCESS: PASS' if real_client_test else 'ERROR: FAIL'}")
    print(f"MCP Gmail Server: {'SUCCESS: PASS' if mcp_server_test else 'ERROR: FAIL'}")
    
    if real_client_test and mcp_server_test:
        print("\nSUCCESS: All Gmail tests passed!")
        print("   SUCCESS: Real authentication working")
        print("   SUCCESS: MCP server implementation ready")
    elif real_client_test:
        print("\nWARNING: Real Gmail authentication works, but MCP server has issues.")
    elif mcp_server_test:
        print("\nWARNING: MCP server works, but real authentication failed.")
        print("   Check your Google credentials and setup.")
    else:
        print("\nERROR: All tests failed. Check your setup:")
        print("   - GOOGLE_CLIENT_SECRET environment variable")
        print("   - Google credentials.json file")
        print("   - Gmail API enabled in Google Cloud Console")
    
    print("\nCleanup: Test completed!")

if __name__ == "__main__":
    # Set up environment
    from dotenv import load_dotenv
    load_dotenv()
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'MISSING'}")
    credentials_path = os.getenv("GOOGLE_CLIENT_SECRET") or "secrets/oauth_credentials.json"
    print(f"Gmail credentials: {'FOUND' if os.path.exists(credentials_path) else 'MISSING'}")
    print(f"Google Client Secret: {'SET' if os.getenv('GOOGLE_CLIENT_SECRET') else 'MISSING'}")
    
    # Run tests
    asyncio.run(main())