"""
Test file to verify all functionality step by step:
1. Gemini MCP client
2. Enhanced agents
3. MCP integrations
4. Overall system functionality
"""
import asyncio
import os
from datetime import datetime

async def test_gemini_mcp_client():
    """Test basic Gemini MCP client functionality"""
    print("\n=== Testing Gemini MCP Client ===")
    
    try:
        from src.mcp_integration.gemini_mcp_client import GeminiMCPClient
        
        # Test instantiation
        client = GeminiMCPClient()
        print("✓ GeminiMCPClient created successfully")
        
        # Test basic chat
        response = await client.chat("Hello, can you respond with 'Gemini is working'?")
        print(f"✓ Basic chat response: {response[:100]}...")
        
        # Test initialization
        await client.initialize()
        print("✓ GeminiMCPClient initialized successfully")
        
        return client
        
    except Exception as e:
        print(f"✗ GeminiMCPClient failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_search_agent(gemini_client):
    """Test Enhanced Search Agent"""
    print("\n=== Testing Enhanced Search Agent ===")
    
    try:
        from src.agents.enhanced_search_agent import EnhancedSearchAgent
        
        # Create agent
        search_agent = EnhancedSearchAgent(gemini_client, "TestSearchAgent")
        print("✓ EnhancedSearchAgent created successfully")
        
        # Test basic search functionality
        from src.core.enhanced_state_schema import make_enhanced_initial_state
        
        test_state = make_enhanced_initial_state("Search for AI news", "test_user")
        result = await search_agent.execute_with_full_pipeline(test_state)
        
        print(f"✓ Search agent executed: {result.get('final_response', 'No response')[:100]}...")
        
        return search_agent
        
    except Exception as e:
        print(f"✗ EnhancedSearchAgent failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_email_agent(gemini_client):
    """Test Enhanced Email Agent"""
    print("\n=== Testing Enhanced Email Agent ===")
    
    try:
        from src.agents.enhanced_email_agent import EnhancedEmailAgent
        
        # Create agent
        email_agent = EnhancedEmailAgent(gemini_client, "TestEmailAgent")
        print("✓ EnhancedEmailAgent created successfully")
        
        # Test basic email functionality (without actually sending)
        from src.core.enhanced_state_schema import make_enhanced_initial_state
        
        test_state = make_enhanced_initial_state("Help me compose an email about project updates", "test_user")
        result = await email_agent.execute_with_full_pipeline(test_state)
        
        print(f"✓ Email agent executed: {result.get('final_response', 'No response')[:100]}...")
        
        return email_agent
        
    except Exception as e:
        print(f"✗ EnhancedEmailAgent failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_calendar_agent(gemini_client):
    """Test Enhanced Calendar Agent"""
    print("\n=== Testing Enhanced Calendar Agent ===")
    
    try:
        from src.agents.enhanced_calendar_agent import EnhancedCalendarAgent
        
        # Create agent
        calendar_agent = EnhancedCalendarAgent(gemini_client, "TestCalendarAgent")
        print("✓ EnhancedCalendarAgent created successfully")
        
        # Test basic calendar functionality
        from src.core.enhanced_state_schema import make_enhanced_initial_state
        
        test_state = make_enhanced_initial_state("Check my calendar for next week", "test_user")
        result = await calendar_agent.execute_with_full_pipeline(test_state)
        
        print(f"✓ Calendar agent executed: {result.get('final_response', 'No response')[:100]}...")
        
        return calendar_agent
        
    except Exception as e:
        print(f"✗ EnhancedCalendarAgent failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_orchestrator(gemini_client, agents):
    """Test Enhanced Orchestrator"""
    print("\n=== Testing Enhanced Orchestrator ===")
    
    try:
        from src.core.enhanced_orchestrator import EnhancedOrchestrator
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(gemini_client)
        print("✓ EnhancedOrchestrator created successfully")
        
        # Register agents
        agent_capabilities = {
            'email': ['email_composition', 'email_search'],
            'calendar': ['calendar_scheduling', 'calendar_search'],
            'search': ['web_search', 'research_analysis']
        }
        
        for agent_name, capabilities in agent_capabilities.items():
            if agent_name in agents and agents[agent_name]:
                orchestrator.register_agent(agent_name, agents[agent_name], capabilities)
                print(f"✓ Registered {agent_name} agent")
        
        # Test orchestration
        from src.core.enhanced_state_schema import make_enhanced_initial_state
        
        test_state = make_enhanced_initial_state("Search for information about AI trends", "test_user")
        result = await orchestrator.orchestrate(test_state)
        
        print(f"✓ Orchestrator executed: {result.get('final_response', 'No response')[:100]}...")
        
        return orchestrator
        
    except Exception as e:
        print(f"✗ EnhancedOrchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_keyword_routing(orchestrator):
    """Test keyword-based routing"""
    print("\n=== Testing Keyword Routing ===")
    
    test_cases = [
        ("Send an email to john@example.com", "email"),
        ("Schedule a meeting for tomorrow", "calendar"), 
        ("Search for latest AI news", "search"),
        ("Hey Gemini, what's the weather?", "general")
    ]
    
    for query, expected_type in test_cases:
        try:
            from src.core.enhanced_state_schema import make_enhanced_initial_state
            
            test_state = make_enhanced_initial_state(query, "test_user")
            result = await orchestrator.orchestrate(test_state)
            
            route = result.get('route', 'unknown')
            print(f"✓ '{query}' -> routed to: {route}")
            
        except Exception as e:
            print(f"✗ Routing failed for '{query}': {e}")

async def main():
    """Main test function"""
    print("Starting Enhanced AI Assistant Functionality Tests")
    print("=" * 60)
    
    # Test Gemini MCP Client
    gemini_client = await test_gemini_mcp_client()
    if not gemini_client:
        print("\n❌ Cannot proceed without working Gemini client")
        return
    
    # Test individual agents
    agents = {}
    agents['search'] = await test_search_agent(gemini_client)
    agents['email'] = await test_email_agent(gemini_client)
    agents['calendar'] = await test_calendar_agent(gemini_client)
    
    # Test orchestrator
    orchestrator = await test_orchestrator(gemini_client, agents)
    
    if orchestrator:
        # Test routing
        await test_keyword_routing(orchestrator)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"- Gemini Client: {'✓' if gemini_client else '✗'}")
    print(f"- Search Agent: {'✓' if agents.get('search') else '✗'}")
    print(f"- Email Agent: {'✓' if agents.get('email') else '✗'}")
    print(f"- Calendar Agent: {'✓' if agents.get('calendar') else '✗'}")
    print(f"- Orchestrator: {'✓' if orchestrator else '✗'}")
    
    if all([gemini_client, agents.get('search'), agents.get('email'), agents.get('calendar'), orchestrator]):
        print("\n🎉 All core components are working! The chatbot should run correctly.")
    else:
        print("\n⚠️  Some components failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())