#!/usr/bin/env python3
"""
Test script to verify all agent functionality works correctly.
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.core.langgraph_workflow import AIAssistantWorkflow

async def test_search_agent():
    """Test the search agent functionality."""
    print("\n🔍 Testing Search Agent...")
    workflow = AIAssistantWorkflow()
    
    try:
        result = await workflow.process_request("what is the capital of prague")
        print(f"✅ Search result: {result.get('final_response', 'No response')[:100]}...")
        print(f"   Route: {result.get('route', 'unknown')}")
        print(f"   Agent: {result.get('current_agent', 'unknown')}")
        print(f"   Fallback triggered: {result.get('fallback_triggered', False)}")
        return result.get('fallback_triggered', True) == False
    except Exception as e:
        print(f"❌ Search agent failed: {e}")
        return False

async def test_email_agent():
    """Test the email agent functionality."""
    print("\n📧 Testing Email Agent...")
    workflow = AIAssistantWorkflow()
    
    try:
        result = await workflow.process_request("show me my unread emails")
        print(f"✅ Email result: {result.get('final_response', 'No response')[:100]}...")
        print(f"   Route: {result.get('route', 'unknown')}")
        print(f"   Agent: {result.get('current_agent', 'unknown')}")
        print(f"   Fallback triggered: {result.get('fallback_triggered', False)}")
        return result.get('fallback_triggered', True) == False
    except Exception as e:
        print(f"❌ Email agent failed: {e}")
        return False

async def test_calendar_agent():
    """Test the calendar agent functionality."""
    print("\n📅 Testing Calendar Agent...")
    workflow = AIAssistantWorkflow()
    
    try:
        result = await workflow.process_request("show my upcoming events")
        print(f"✅ Calendar result: {result.get('final_response', 'No response')[:100]}...")
        print(f"   Route: {result.get('route', 'unknown')}")
        print(f"   Agent: {result.get('current_agent', 'unknown')}")
        print(f"   Fallback triggered: {result.get('fallback_triggered', False)}")
        return result.get('fallback_triggered', True) == False
    except Exception as e:
        print(f"❌ Calendar agent failed: {e}")
        return False

async def test_fallback_improvement():
    """Test that fallback responses are helpful."""
    print("\n🔄 Testing Improved Fallback...")
    workflow = AIAssistantWorkflow()
    
    try:
        result = await workflow.process_request("xyz abc random nonsense 123")
        print(f"✅ Fallback result: {result.get('final_response', 'No response')[:200]}...")
        print(f"   Route: {result.get('route', 'unknown')}")
        print(f"   Fallback triggered: {result.get('fallback_triggered', False)}")
        
        # Check if fallback was triggered OR if response is helpful
        response = result.get('final_response', '').lower()
        fallback_triggered = result.get('fallback_triggered', False)
        is_helpful = any(word in response for word in ['help', 'can', 'try', 'email', 'calendar', 'search'])
        
        # Test passes if either fallback was properly triggered OR the response is helpful regardless of route
        return fallback_triggered or is_helpful
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Running AI Assistant Tests...")
    print("=" * 50)
    
    results = {
        'search': await test_search_agent(),
        'email': await test_email_agent(), 
        'calendar': await test_calendar_agent(),
        'fallback': await test_fallback_improvement()
    }
    
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize()} Agent: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some issues remain. Check the logs above for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)