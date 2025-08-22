# src/langgraph_workflow.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional

from src.core.state_schema import AssistantState, make_initial_state
from src.core.orchestrator import CoreOrchestrator
from src.agents.email_agent import EmailAgent
from src.agents.calendar_agent import CalendarAgent
from src.agents.brave_agent import BraveAgent

from src.mcp_integration.gemini_mcp_client import GeminiMCPClient

class AIAssistantWorkflow:
    """
    LangGraph workflow for the AI Assistant system.
    Manages agent routing, state transitions, and orchestration.
    """
    
    def __init__(self):
        # Initialize core components - using only Gemini MCP client
        self.gemini_mcp_client = GeminiMCPClient()
        self.orchestrator = CoreOrchestrator(self.gemini_mcp_client)
        
        # Flag to track initialization
        self.initialized = False
        
        # Initialize agents with Gemini MCP client
        self.agents = {
            'email': EmailAgent(self.gemini_mcp_client),
            'calendar': CalendarAgent(self.gemini_mcp_client),
            'search': BraveAgent(self.gemini_mcp_client),
        }
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    async def initialize_servers(self):
        """Initialize MCP servers if not already done."""
        if not self.initialized:
            try:
                # Initialize Gemini MCP client with all MCP services
                await self.gemini_mcp_client.initialize()
                self.initialized = True
                print("[SUCCESS] MCP servers initialized successfully")
            except Exception as e:
                print(f"[ERROR] MCP server initialization failed: {e}")
                raise
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges."""
        
        # Create the graph
        workflow = StateGraph(AssistantState)
        
        # Add nodes
        workflow.add_node("route_request", self._route_request)
        workflow.add_node("email_agent", self._execute_email_agent)
        workflow.add_node("calendar_agent", self._execute_calendar_agent)
        workflow.add_node("search_agent", self._execute_search_agent)
        workflow.add_node("orchestrator", self._execute_orchestrator)
        workflow.add_node("verify_response", self._verify_response)
        workflow.add_node("fallback_handler", self._handle_fallback)
        
        # Set entry point
        workflow.set_entry_point("route_request")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "route_request",
            self._route_decision,
            {
                "email": "email_agent",
                "calendar": "calendar_agent",
                "search": "search_agent",
                "orchestrator": "orchestrator",
                "fallback": "fallback_handler"
            }
        )
        
        # Add edges from agents to verification
        for agent in ["email_agent", "calendar_agent", "search_agent", "orchestrator"]: 
            workflow.add_edge(agent, "verify_response")
        
        # Add conditional edges from verification
        workflow.add_conditional_edges(
            "verify_response",
            self._verification_decision,
            {
                "success": END,
                "retry": "route_request",
                "fallback": "fallback_handler"
            }
        )
        
        # Fallback always ends
        workflow.add_edge("fallback_handler", END)
        
        return workflow.compile(checkpointer=None, interrupt_before=None, debug=False)
    
    # Node Functions
    async def _route_request(self, state: AssistantState) -> AssistantState:
        """Route the user request to appropriate agent."""
        try:
            # Use orchestrator for intelligent routing
            route_result = await self.orchestrator.route_request(state['user_input'])
            
            # Update state with routing information
            state['route'] = route_result['route']
            state['route_confidence'] = route_result['confidence']
            state['route_reason'] = route_result['reason']
            state['task_type'] = route_result.get('task_type', 'general')
            
            return state
            
        except Exception as e:
            # Handle routing errors
            error_log = state.get('error_log', [])
            error_log.append({
                'stage': 'routing',
                'error': str(e),
                'timestamp': self._get_timestamp()
            })
            state['error_log'] = error_log
            state['route'] = 'fallback'
            return state
    
    async def _execute_email_agent(self, state: AssistantState) -> AssistantState:
        """Execute the email agent."""
        state['current_agent'] = 'email'
        return await self.agents['email'].execute_with_tracking(state)
    
    async def _execute_calendar_agent(self, state: AssistantState) -> AssistantState:
        """Execute the calendar agent."""
        state['current_agent'] = 'calendar'
        return await self.agents['calendar'].execute_with_tracking(state)
    
    async def _execute_search_agent(self, state: AssistantState) -> AssistantState:
        """Execute the search agent."""
        state['current_agent'] = 'search'
        return await self.agents['search'].execute_with_tracking(state)
    
    async def _execute_orchestrator(self, state: AssistantState) -> AssistantState:
        """Execute the orchestrator for complex multi-agent tasks."""
        state['current_agent'] = 'orchestrator'
        try:
            # Use orchestrator to handle complex requests that may require multiple agents
            orchestrator_result = await self.orchestrator.process_complex_request(
                state['user_input'],
                state.get('task_type', 'general')
            )
            
            state['final_response'] = orchestrator_result.get('response', '')
            state['orchestrator_metadata'] = orchestrator_result.get('metadata', {})
            
            return state
            
        except Exception as e:
            # Handle orchestrator execution errors
            error_log = state.get('error_log', [])
            error_log.append({
                'stage': 'orchestrator_execution',
                'error': str(e),
                'timestamp': self._get_timestamp()
            })
            state['error_log'] = error_log
            state['route'] = 'fallback'
            return state
    
    async def _verify_response(self, state: AssistantState) -> AssistantState:
        """Verify the quality and completeness of the agent response."""
        try:
            # Use orchestrator for response verification
            verification_result = await self.orchestrator.verify_response(
                state['user_input'],
                state.get('final_response', ''),
                state.get('current_agent', '')
            )
            
            # Update verification scores
            state['verification_scores'] = verification_result
            
            return state
            
        except Exception as e:
            # Handle verification errors
            error_log = state.get('error_log', [])
            error_log.append({
                'stage': 'verification',
                'error': str(e),
                'timestamp': self._get_timestamp()
            })
            state['error_log'] = error_log
            
            # Default to fallback on verification errors
            state['verification_scores'] = {'quality': 0.0, 'completeness': 0.0}
            return state
    
    async def _handle_fallback(self, state: AssistantState) -> AssistantState:
        """Handle fallback scenarios when other agents fail."""
        try:
            # Use orchestrator for fallback response
            fallback_response = await self.orchestrator.handle_fallback(
                state['user_input'],
                state.get('error_log', [])
            )
            
            state['final_response'] = fallback_response
            state['fallback_triggered'] = True
            state['current_agent'] = 'fallback'
            
            return state
            
        except Exception as e:
            # Last resort fallback
            state['final_response'] = "I apologize, but I'm unable to process your request at the moment. Please try again later."
            state['fallback_triggered'] = True
            state['current_agent'] = 'fallback'
            
            error_log = state.get('error_log', [])
            error_log.append({
                'stage': 'fallback',
                'error': str(e),
                'timestamp': self._get_timestamp()
            })
            state['error_log'] = error_log
            
            return state
    
    # Decision Functions
    def _route_decision(self, state: AssistantState) -> str:
        """Determine which agent should handle the request."""
        route = state.get('route', 'fallback')
        confidence = state.get('route_confidence', 0.0)
        
        # Require minimum confidence for routing (lowered for testing)
        if confidence < 0.3:
            return 'fallback'
        
        # Map routes to agent nodes
        route_mapping = {
            'email': 'email',
            'calendar': 'calendar',
            'search': 'search',
            'multi_agent': 'orchestrator',
            'complex': 'orchestrator'
        }
        
        return route_mapping.get(route, 'fallback')
    
    def _verification_decision(self, state: AssistantState) -> str:
        """Determine next step based on verification results."""
        verification_scores = state.get('verification_scores', {})
        retry_count = state.get('retry_count', 0)
        
        # Check if response quality is acceptable
        quality_score = verification_scores.get('quality', 0.0)
        completeness_score = verification_scores.get('completeness', 0.0)
        
        # Success criteria - lower the threshold to avoid infinite retries
        if quality_score >= 0.5 and completeness_score >= 0.5:
            return 'success'
        
        # Retry logic (max 1 retry to prevent infinite loops)
        if retry_count < 1 and quality_score >= 0.3:
            state['retry_count'] = retry_count + 1
            return 'retry'
        
        # Fallback for poor quality or too many retries
        return 'fallback'
    
    # Utility Functions
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    # Public Interface
    async def process_request(self, user_input: str, user: Optional[str] = None) -> AssistantState:
        """
        Process a user request through the complete workflow.
        
        Args:
            user_input: The user's request
            user: Optional user identifier
            
        Returns:
            Final state with response and metadata
        """
        # Ensure servers are initialized
        await self.initialize_servers()
        
        # Create initial state
        initial_state = make_initial_state(user_input, user)
        
        # Execute the workflow with recursion limit
        final_state = await self.workflow.ainvoke(
            initial_state,
            config={"recursion_limit": 10}
        )
        
        return final_state
    
    async def stream_process(self, user_input: str, user: Optional[str] = None):
        """
        Stream the workflow execution for real-time updates.
        
        Args:
            user_input: The user's request
            user: Optional user identifier
            
        Yields:
            State updates throughout the workflow execution
        """
        # Create initial state
        initial_state = make_initial_state(user_input, user)
        
        # Stream the workflow execution with recursion limit
        async for state_update in self.workflow.astream(
            initial_state,
            config={"recursion_limit": 10}
        ):
            yield state_update
    
    def get_workflow_graph(self) -> str:
        """Get a visual representation of the workflow graph."""
        try:
            return self.workflow.get_graph().draw_mermaid()
        except:
            return "Workflow graph visualization not available"
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all system components.
        
        Returns:
            Health status of all components
        """
        health_status = {
            'workflow': 'healthy',
            'orchestrator': 'unknown',
            'agents': {},
            'clients': {}
        }
        
        try:
            # Check orchestrator
            orchestrator_status = await self.orchestrator.health_check()
            health_status['orchestrator'] = 'healthy' if orchestrator_status else 'unhealthy'
        except:
            health_status['orchestrator'] = 'unhealthy'
        
        # Check agents
        for agent_name, agent in self.agents.items():
            try:
                agent_status = agent.get_status()
                health_status['agents'][agent_name] = 'healthy' if agent_status['status'] == 'active' else 'unhealthy'
            except:
                health_status['agents'][agent_name] = 'unhealthy'
        
        # Check clients
        try:
            health_status['clients']['gemini_mcp'] = 'healthy' if self.gemini_mcp_client else 'unhealthy'
        except:
            health_status['clients']['gemini_mcp'] = 'unhealthy'
        
        return health_status


# Convenience function for easy usage
async def process_user_request(user_input: str, user: Optional[str] = None) -> str:
    """
    Simplified interface to process a user request and get a response.
    
    Args:
        user_input: The user's request
        user: Optional user identifier
        
    Returns:
        The assistant's response
    """
    workflow = AIAssistantWorkflow()
    result_state = await workflow.process_request(user_input, user)
    return result_state.get('final_response', 'No response generated')


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_workflow():
        """Test the workflow with sample requests."""
        workflow = AIAssistantWorkflow()
        
        test_requests = [
            "Help me organize my emails",
            "Schedule a meeting for tomorrow at 2 PM",
            "What are my priority tasks for today?",
            "Send an email to john@company.com about the project update"
        ]
        
        print("🚀 Testing AI Assistant Workflow\n")
        
        for request in test_requests:
            print(f"📝 Request: {request}")
            
            try:
                result = await workflow.process_request(request)
                print(f"✅ Response: {result.get('final_response', 'No response')}")
                print(f"🔀 Route: {result.get('route', 'unknown')} (confidence: {result.get('route_confidence', 0):.2f})")
                print(f"🤖 Agent: {result.get('current_agent', 'unknown')}")
                
                if result.get('error_log'):
                    print(f"⚠️ Errors: {len(result['error_log'])}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 50)
        
        # Health check
        print("\n🏥 System Health Check:")
        health = await workflow.health_check()
        for component, status in health.items():
            if isinstance(status, dict):
                print(f"  {component}:")
                for sub_component, sub_status in status.items():
                    print(f"    {sub_component}: {sub_status}")
            else:
                print(f"  {component}: {status}")
    
    # Run the test
    asyncio.run(test_workflow())