# src/langgraph_workflow.py
from langgraph.graph import StateGraph, END
from typing import Dict, List, Any, Optional

from src.core.state_schema import AssistantState, make_initial_state
from src.core.orchestrator import CoreOrchestrator
from src.agents.enhanced_email_agent import EnhancedEmailAgent
from src.agents.calendar_agent import CalendarAgent
from src.agents.brave_agent import BraveAgent
from src.mcp_integration.gemini_mcp_client import GeminiMCPClient

class AIAssistantWorkflow:
    def __init__(self):
        self.gemini_mcp_client = GeminiMCPClient()
        self.orchestrator = CoreOrchestrator(self.gemini_mcp_client)
        self.initialized = False

        self.agents = {
            'email': EnhancedEmailAgent(self.gemini_mcp_client),
            'calendar': CalendarAgent(self.gemini_mcp_client),
            'search': BraveAgent(self.gemini_mcp_client),
            }


        self.workflow = self._build_workflow()

    async def initialize_servers(self):
        if not self.initialized:
            await self.gemini_mcp_client.initialize()
            self.initialized = True
            print("[SUCCESS] MCP servers initialized successfully")

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AssistantState)

        workflow.add_node("route_request", self._route_request)
        workflow.add_node("email_agent", self._execute_email_agent)
        workflow.add_node("calendar_agent", self._execute_calendar_agent)
        workflow.add_node("search_agent", self._execute_search_agent)
        workflow.add_node("orchestrator", self._execute_orchestrator)
        workflow.add_node("verify_response", self._verify_response)
        workflow.add_node("summarize_with_gemini", self._summarize_with_gemini)
        workflow.add_node("fallback_handler", self._handle_fallback)

        workflow.set_entry_point("route_request")

        workflow.add_conditional_edges("route_request", self._route_decision, {
            "email": "email_agent",
            "calendar": "calendar_agent",
            "search": "search_agent",
            "orchestrator": "orchestrator",
            "fallback": "fallback_handler"
        })

        # ✅ Route ALL agents to orchestrator
        workflow.add_edge("email_agent", "orchestrator")
        workflow.add_edge("calendar_agent", "orchestrator")
        workflow.add_edge("search_agent", "orchestrator")
        workflow.add_edge("orchestrator", "verify_response")

        workflow.add_conditional_edges("verify_response", self._verification_decision, {
            "success": "summarize_with_gemini",
            "retry": "route_request",
            "fallback": "fallback_handler"
        })

        workflow.add_edge("summarize_with_gemini", END)
        workflow.add_edge("fallback_handler", END)

        return workflow.compile(checkpointer=None, interrupt_before=None, debug=False)

    async def _route_request(self, state: AssistantState) -> AssistantState:
        try:
            # Pass conversation history to the orchestrator for context-aware routing
            conversation_history = state.get('conversation_history', [])
            route_result = await self.orchestrator.route_request(state['user_input'], conversation_history)
            state.update({
                'route': route_result['route'],
                'route_confidence': route_result['confidence'],
                'route_reason': route_result['reason'],
                'task_type': route_result.get('task_type', 'general'),
                'user_intent': route_result.get('user_intent', ''),
                'next_action': route_result.get('next_action', ''),
                'parameters': route_result.get('parameters', {})
            })
            return state
        except Exception as e:
            state.update({
                'route': 'fallback',
                'error_log': state.get('error_log', []) + [{'stage': 'routing', 'error': str(e)}]
            })
            return state

    async def _execute_email_agent(self, state: AssistantState) -> AssistantState:
        state['current_agent'] = 'email'
        # Pass LLM analysis parameters to the email agent
        if 'parameters' not in state:
            state['parameters'] = {}
        if 'user_intent' not in state:
            state['user_intent'] = ''
        if 'next_action' not in state:
            state['next_action'] = ''
        return await self.agents['email'].execute_with_tracking(state)

    async def _execute_calendar_agent(self, state: AssistantState) -> AssistantState:
        state['current_agent'] = 'calendar'
        return await self.agents['calendar'].execute_with_tracking(state)

    async def _execute_search_agent(self, state: AssistantState) -> AssistantState:
        state['current_agent'] = 'search'
        return await self.agents['search'].execute_with_tracking(state)

    async def _execute_orchestrator(self, state: AssistantState) -> AssistantState:
        state['current_agent'] = 'orchestrator'
        try:
            # Check if this is an escalation or collaboration request from an agent
            if state.get('escalation_request') or state.get('agent_help_request'):
                result = await self.orchestrator.handle_agent_escalation(state)
                state['orchestrator_metadata'] = result.get('metadata', {})
                state['agent_messages'] = state.get('agent_messages', []) + [{'role': 'system', 'content': result.get('response', '')}]
                state['final_response'] = result.get('response', '')
                return state
            
            # Check if this request should be handled by a specific agent
            route = state.get('route', '')
            if route in ['email', 'calendar', 'search']:
                # This request came FROM an agent, just coordinate the response
                agent_messages = state.get('agent_messages', [])
                final_response = state.get('final_response', '')
                
                if final_response:
                    state['agent_messages'] = agent_messages + [{'role': 'system', 'content': final_response}]
                else:
                    state['agent_messages'] = agent_messages + [{'role': 'system', 'content': 'Agent completed task'}]
                return state
            else:
                # Handle complex orchestrator requests
                result = await self.orchestrator.process_complex_request(state['user_input'], state.get('task_type', 'general'))
                state['orchestrator_metadata'] = result.get('metadata', {})
                state['agent_messages'] = [{'role': 'system', 'content': result.get('response', '')}]
                return state
        except Exception as e:
            state.update({
                'route': 'fallback',
                'error_log': state.get('error_log', []) + [{'stage': 'orchestrator_execution', 'error': str(e)}]
            })
            return state

    async def _verify_response(self, state: AssistantState) -> AssistantState:
        try:
            result = await self.orchestrator.verify_response(state['user_input'], state.get('final_response', ''), state.get('current_agent', ''))
            state['verification_scores'] = result
            return state
        except Exception as e:
            state.update({
                'verification_scores': {'quality': 0.0, 'completeness': 0.0},
                'error_log': state.get('error_log', []) + [{'stage': 'verification', 'error': str(e)}]
            })
            return state

    async def _summarize_with_gemini(self, state: AssistantState) -> AssistantState:
        try:
            content = "\n".join([m['content'] for m in state.get('agent_messages', []) if m.get('content')])
            user_input = state.get('user_input', '')
            
            # For calendar requests, always format through Gemini
            if state.get('current_agent') == 'calendar':
                # Let Gemini format the calendar data properly
                pass
            # Check if we already have a final response from other agents
            elif state.get('final_response') and len(state['final_response'].strip()) > 5:
                return state  # Keep existing concise response
            
            # Different prompts for different agent types
            if state.get('current_agent') == 'calendar':
                prompt = f"""Format the calendar information to answer: "{user_input}"

Calendar data: {content}

Format each event as:
Event Name: [event title]
Date: [day, month date]
Time: [start time] - [end time]
Attendees: [attendee list or "None"]

Show all events in this format."""
            else:
                prompt = f"""Provide a direct, concise answer to: "{user_input}"
                
Based on: {content}

Requirements:
- Answer in 1-2 sentences maximum
- Be direct and factual
- No explanations or elaborations
- Just the core answer"""
            final = await self.gemini_mcp_client.chat(prompt)
            state['final_response'] = final
        except Exception as e:
            state['final_response'] = "Task completed but summarization failed."
        return state

    async def _handle_fallback(self, state: AssistantState) -> AssistantState:
        try:
            response = await self.orchestrator.handle_fallback(state['user_input'], state.get('error_log', []))
            state.update({
                'final_response': response,
                'fallback_triggered': True,
                'current_agent': 'fallback'
            })
            return state
        except Exception as e:
            state.update({
                'final_response': "Sorry, I couldn't process your request.",
                'fallback_triggered': True,
                'current_agent': 'fallback',
                'error_log': state.get('error_log', []) + [{'stage': 'fallback', 'error': str(e)}]
            })
            return state

    def _route_decision(self, state: AssistantState) -> str:
        if state.get('route_confidence', 0.0) < 0.3:
            return 'fallback'
        return {
            'email': 'email',
            'calendar': 'calendar',
            'search': 'search',
            'multi_agent': 'orchestrator',
            'complex': 'orchestrator',
            'orchestrator': 'orchestrator'
        }.get(state.get('route', ''), 'fallback')

    def _verification_decision(self, state: AssistantState) -> str:
        scores = state.get('verification_scores', {})
        final_response = state.get('final_response', '')
        
        # If there's a final response and no major errors, consider it successful
        if final_response and len(final_response.strip()) > 10:
            # Check if response indicates an error
            error_indicators = ['error', 'failed', 'unable to', 'cannot', 'sorry']
            has_errors = any(indicator in final_response.lower() for indicator in error_indicators)
            
            if not has_errors:
                return 'success'
        
        # Original verification logic with more lenient thresholds
        if scores.get('quality', 0.0) >= 0.3 and scores.get('completeness', 0.0) >= 0.3:
            return 'success'
        if state.get('retry_count', 0) < 1 and scores.get('quality', 0.0) >= 0.2:
            state['retry_count'] = state.get('retry_count', 0) + 1
            return 'retry'
        return 'fallback'

    async def process_request(self, user_input: str, user: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> AssistantState:
        await self.initialize_servers()
        initial_state = make_initial_state(user_input, user)
        
        # Preserve conversation history if provided
        if conversation_history:
            initial_state['conversation_history'] = conversation_history + [{"user": user or "me", "text": user_input}]
        
        return await self.workflow.ainvoke(initial_state, config={"recursion_limit": 10})

    async def stream_process(self, user_input: str, user: Optional[str] = None):
        initial_state = make_initial_state(user_input, user)
        async for update in self.workflow.astream(initial_state, config={"recursion_limit": 10}):
            yield update

    def get_workflow_graph(self) -> str:
        try:
            return self.workflow.get_graph().draw_mermaid()
        except:
            return "Graph unavailable"

    async def health_check(self) -> Dict[str, Any]:
        result = {'workflow': 'healthy', 'orchestrator': 'unknown', 'agents': {}, 'clients': {}}
        try:
            result['orchestrator'] = 'healthy' if await self.orchestrator.health_check() else 'unhealthy'
        except:
            result['orchestrator'] = 'unhealthy'
        for name, agent in self.agents.items():
            try:
                result['agents'][name] = 'healthy' if agent.get_status()['status'] == 'active' else 'unhealthy'
            except:
                result['agents'][name] = 'unhealthy'
        result['clients']['gemini_mcp'] = 'healthy' if self.gemini_mcp_client else 'unhealthy'
        return result
