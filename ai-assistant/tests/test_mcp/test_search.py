#!/usr/bin/env python3
"""
Simple test case for search_tools.py
Run this to verify your MCP integration and Brave API are working correctly.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_search_tools():
    """Test the SearchTools functionality with basic diagnostics."""
    
    print("🔍 Testing SearchTools Integration")
    print("=" * 50)
    
    # Step 1: Check environment variables
    print("1. Checking environment variables...")
    brave_key = os.getenv("BRAVE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if brave_key:
        print(f"   ✅ BRAVE_API_KEY: Found (length: {len(brave_key)})")
    else:
        print("   ❌ BRAVE_API_KEY: Not found")
        return
    
    if google_key:
        print(f"   ✅ GOOGLE_API_KEY: Found (length: {len(google_key)})")
    else:
        print("   ⚠️  GOOGLE_API_KEY: Not found (only needed for Gemini)")
    
    # Step 2: Test MCP Client import
    print("\n2. Testing MCP imports...")
    try:
        from src.mcp_integration.mcp_client import MCPClient
        from src.mcp_integration.search_tools import SearchTools
        print("   ✅ MCP imports successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return
    
    # Step 3: Initialize clients
    print("\n3. Initializing clients...")
    try:
        client = MCPClient()
        search = SearchTools(client)
        print("   ✅ Clients initialized")
    except Exception as e:
        print(f"   ❌ Client initialization failed: {e}")
        return
    
    # Step 4: Start search tools
    print("\n4. Starting search tools...")
    try:
        await search.start()
        print("   ✅ Search tools started successfully")
    except Exception as e:
        print(f"   ❌ Search tools start failed: {e}")
        if client:
            await client.shutdown()
        return
    
    # Step 5: Perform a simple search
    print("\n5. Testing web search...")
    try:
        results = await search.web_search('Python programming tutorial')
        print(f"   ✅ Search completed, got {len(results)} results")
        
        if results:
            print("\n   📋 First result:")
            first_result = results[0]
            print(f"      Title: {first_result.get('title', 'N/A')}")
            print(f"      URL: {first_result.get('url', 'N/A')}")
            print(f"      Description: {first_result.get('description', 'N/A')[:100]}...")
        else:
            print("   ⚠️  No results returned")
    
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    # Step 6: Cleanup
    print("\n6. Cleaning up...")
    try:
        await client.shutdown()
        print("   ✅ Cleanup completed")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    print("\n🎉 Test completed!")

# Standalone test function for manual verification
async def quick_api_test():
    """Quick test to verify API key is accessible."""
    load_dotenv()
    
    brave_key = os.getenv("BRAVE_API_KEY")
    if not brave_key:
        print("❌ BRAVE_API_KEY not found in environment")
        print("   Check your .env file contains: BRAVE_API_KEY=your_key")
        return False
    
    print(f"✅ BRAVE_API_KEY found (starts with: {brave_key[:8]}...)")
    return True

if __name__ == "__main__":
    print("Choose test:")
    print("1. Quick API key check")
    print("2. Full search tools test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(quick_api_test())
    else:
        asyncio.run(test_search_tools())