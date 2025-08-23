#!/usr/bin/env python3
"""
Test script for Brave Search Agent integration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.brave_agent import BraveAgent
from src.mcp_integration.gemini_mcp_client import GeminiMCPClient
from src.mcp_integration.search_tools import SearchTools
from src.core.state_schema import make_initial_state


async def test_search_tools():
    """Test SearchTools directly"""
    print("=== Testing SearchTools directly ===")
    
    try:
        # Create a mock MCP client for SearchTools
        search_tools = SearchTools(None)
        
        # Test basic search
        print("Testing web search...")
        results = await search_tools.web_search("artificial intelligence", count=3)
        print(f"Search completed: {results.get('total_results', 0)} results")
        
        if results.get('results'):
            for i, result in enumerate(results['results'][:2], 1):
                print(f"  {i}. {result.get('title', 'No title')}")
                print(f"     {result.get('url', 'No URL')}")
        
        return True
    except Exception as e:
        print(f"SearchTools test failed: {e}")
        return False


async def test_brave_agent():
    """Test BraveAgent"""
    print("\n=== Testing BraveAgent ===")
    
    try:
        # Initialize Gemini MCP client
        gemini_client = GeminiMCPClient()
        await gemini_client.initialize()
        
        # Create BraveAgent
        brave_agent = BraveAgent(gemini_client)
        
        # Test can_handle method
        test_queries = [
            "search for Python tutorials",
            "find information about machine learning",
            "what is quantum computing",
            "schedule a meeting"  # Should not be handled
        ]
        
        print("Testing can_handle method:")
        for query in test_queries:
            can_handle = await brave_agent.can_handle(query)
            print(f"  '{query}' -> {can_handle}")
        
        # Test actual search execution
        print("\nTesting search execution...")
        state = make_initial_state("search for latest AI news", {})
        
        result_state = await brave_agent.execute(state)
        
        if 'response' in result_state:
            print("Search agent executed successfully")
            print(f"Response length: {len(result_state['response'])} characters")
        else:
            print("No response generated")
            return False
        
        return True
    except Exception as e:
        print(f"BraveAgent test failed: {e}")
        return False


async def test_workflow_integration():
    """Test full workflow integration"""
    print("\n=== Testing Workflow Integration ===")
    
    try:
        from src.core.langgraph_workflow import AIAssistantWorkflow
        
        # Initialize workflow
        workflow = AIAssistantWorkflow()
        await workflow.initialize_servers()
        
        # Test search routing
        print("Testing search routing...")
        
        search_queries = [
            "search for Python tutorials",
            "find information about climate change",
            "what is the latest news about AI"
        ]
        
        for query in search_queries:
            print(f"\nTesting query: '{query}'")
            
            # Create initial state
            initial_state = make_initial_state(query, {})
            
            # Run workflow
            result = await workflow.run(initial_state)
            
            if result.get('final_response'):
                print(f"Workflow completed successfully")
                print(f"Agent used: {result.get('current_agent', 'unknown')}")
                print(f"Response preview: {result['final_response'][:100]}...")
            else:
                print("No final response generated")
        
        return True
    except Exception as e:
        print(f"Workflow integration test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("Brave Search Integration Test Suite")
    print("=" * 50)
    
    # Test individual components
    search_tools_ok = await test_search_tools()
    brave_agent_ok = await test_brave_agent()
    workflow_ok = await test_workflow_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"SearchTools:      {'PASS' if search_tools_ok else 'FAIL'}")
    print(f"BraveAgent:       {'PASS' if brave_agent_ok else 'FAIL'}")
    print(f"Workflow:         {'PASS' if workflow_ok else 'FAIL'}")
    
    all_passed = search_tools_ok and brave_agent_ok and workflow_ok
    print(f"\nOverall:          {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)