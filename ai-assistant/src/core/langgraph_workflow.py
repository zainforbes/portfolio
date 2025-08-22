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
    def __init__(self):
        self.gemini_mcp_client = GeminiMCPClient()
        self.orchestrator = CoreOrchestrator(self.gemini_mcp_client)
        self.initialized = False

        self.agents = {
            'email': EmailAgent(self.gemini_mcp_client),
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
            route_result = await self.orchestrator.route_request(state['user_input'])
            state.update({
                'route': route_result['route'],
                'route_confidence': route_result['confidence'],
                'route_reason': route_result['reason'],
                'task_type': route_result.get('task_type', 'general')
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
            final = await self.gemini_mcp_client.generate_response(
                content,
                context="Summarize the assistant's findings and return a final answer."
            )
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
        if scores.get('quality', 0.0) >= 0.5 and scores.get('completeness', 0.0) >= 0.5:
            return 'success'
        if state.get('retry_count', 0) < 1 and scores.get('quality', 0.0) >= 0.3:
            state['retry_count'] = state.get('retry_count', 0) + 1
            return 'retry'
        return 'fallback'

    async def process_request(self, user_input: str, user: Optional[str] = None) -> AssistantState:
        await self.initialize_servers()
        initial_state = make_initial_state(user_input, user)
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
