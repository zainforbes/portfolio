#!/usr/bin/env python3
"""
Test script for Gmail and Calendar integration
"""
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from src.mcp_integration.gemini_mcp_client import GeminiMCPClient

async def test_integration():
    """Test Gmail and Calendar integration"""
    print("Testing Gmail and Calendar Integration")
    print("="*50)
    
    try:
        # Initialize client
        client = GeminiMCPClient()
        await client.initialize()
        
        print("\n[EMAIL] Testing Gmail functionality...")
        
        # Test list emails
        emails = await client._list_emails(query="", max_results=5)
        if isinstance(emails, dict) and "error" not in emails:
            print(f"[OK] Successfully retrieved {emails.get('count', 0)} emails")
            for email in emails.get('messages', [])[:3]:
                print(f"  Email: {email.get('subject', 'No Subject')[:50]}...")
        else:
            print(f"[FAIL] Gmail test failed: {emails}")
        
        print("\n[CALENDAR] Testing Calendar functionality...")
        
        # Test list calendar events
        events = await client._list_calendar_events(days_ahead=7)
        if isinstance(events, dict) and "error" not in events:
            print(f"[OK] Successfully retrieved {events.get('count', 0)} calendar events")
            for event in events.get('events', [])[:3]:
                print(f"  Event: {event.get('summary', 'No Title')} - {event.get('start', '')}")
        else:
            print(f"[FAIL] Calendar test failed: {events}")
        
        print("\n[AI] Testing AI chat with tools...")
        
        # Test AI chat that uses tools
        response = await client.chat("List my unread emails please")
        print(f"[OK] AI Response: {response[:200]}...")
        
        print("\nIntegration test complete!")
        
    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'client' in locals():
            await client.cleanup()

if __name__ == "__main__":
    asyncio.run(test_integration())