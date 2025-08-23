#!/usr/bin/env python3
"""
Gmail integration test to verify MCP integration works.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / '..' / '..'))

from src.mcp_integration.gmail_server import GmailServer
from src.mcp_integration.gmail_tools import GmailTools
from src.mcp_integration.gemini_mcp_client import GeminiMCPClient

async def test_gmail_server_direct():
    """Test Gmail server directly"""
    print("=Testing Gmail Server directly...")
    
    try:
        gmail_server = GmailServer()
        
        # Test authentication
        print("=� Attempting Gmail authentication...")
        auth_result = await gmail_server.authenticate()
        
        if auth_result and auth_result.get('success'):
            print(" Gmail authentication successful!")
            
            # Test listing messages
            print("=� Testing message listing...")
            messages = await gmail_server.list_messages(max_results=5)
            print(f"=� Found {len(messages.get('messages', []))} messages")
            
            # Test profile info
            print("=d Testing profile info...")
            profile = await gmail_server.get_profile()
            print(f"=� Email: {profile.get('emailAddress', 'Unknown')}")
            print(f"=� Total messages: {profile.get('messagesTotal', 0)}")
            
            return True
            
        else:
            print("L Gmail authentication failed")
            return False
            
    except Exception as e:
        print(f"L Error testing Gmail server: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gmail_tools():
    """Test Gmail tools with MCP client"""
    print("\n=� Testing Gmail Tools with MCP client...")
    
    try:
        mcp_client = GeminiMCPClient()
        gmail_tools = GmailTools(mcp_client)
        
        print("=� Starting Gmail MCP server...")
        success = await gmail_tools.start()
        
        if success:
            print(" Gmail MCP server started successfully!")
            
            # Test listing emails
            print("=� Testing email listing via MCP...")
            result = await gmail_tools.list_emails(max_results=5)
            print(f"=� MCP Result: {result}")
            
            return True
        else:
            print("L Failed to start Gmail MCP server")
            return False
            
    except Exception as e:
        print(f"L Error testing Gmail tools: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all Gmail tests"""
    print(">� Gmail Integration Test Suite")
    print("=" * 50)
    
    # Check environment first
    from dotenv import load_dotenv
    load_dotenv()
    
    print(f"=� Working directory: {os.getcwd()}")
    print(f"= Google API Key: {' Set' if os.getenv('GOOGLE_API_KEY') else 'L Missing'}")
    print(f"= Gmail credentials: {' Found' if os.path.exists('config/credentials.json') else 'L Missing'}")
    print()
    
    # Test 1: Direct server test
    server_test = await test_gmail_server_direct()
    
    # Test 2: MCP tools test  
    tools_test = await test_gmail_tools()
    
    print("\n" + "=" * 50)
    print("=� Test Results:")
    print(f"Gmail Server Direct: {' PASS' if server_test else 'L FAIL'}")
    print(f"Gmail Tools MCP: {' PASS' if tools_test else 'L FAIL'}")
    
    if server_test and tools_test:
        print("\n<� All Gmail tests passed! Integration is working.")
    else:
        print("\n� Some tests failed. Check your credentials and setup.")

if __name__ == "__main__":
    asyncio.run(main())