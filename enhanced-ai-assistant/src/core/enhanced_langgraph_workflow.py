from langgraph.graph import StateGraph, END
from typing import Dict, List, Any, Optional
import logging
import time
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.core.enhanced_state_schema import (
    EnhancedAssistantState, make_enhanced_initial_state, TaskResult,
    update_resource_metrics, record_agent_decision, safe_get_resource_metric
)
from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.core.enhanced_message_types import MessageType, MessageFactory
from src.mcp_integration.gemini_mcp_client import GeminiMCPClient

# Import enhanced agents
from src.agents.enhanced_email_agent import EnhancedEmailAgent
from src.agents.enhanced_calendar_agent import EnhancedCalendarAgent  
from src.agents.enhanced_search_agent import EnhancedSearchAgent

class EnhancedAIAssistantWorkflow:
    """
    Next-generation AI Assistant Workflow with advanced capabilities:
    
    🧠 Intelligent Multi-Agent Orchestration
    🤝 Dynamic Collaboration & Escalation
    🎯 Autonomous Decision Making
    🔄 Error Recovery & Fallback Chains
    ⚡ Performance Optimization & Caching
    📝 Advanced Prompt Engineering
    🛡️ Hallucination Mitigation & Verification
    💾 Resource Management & Rate Limiting
    """
    
    def __init__(self):
        self.logger = logging.getLogger("enhanced_workflow")
        
        # Initialize MCP client
        self.gemini_mcp_client = GeminiMCPClient()
        
        # Initialize enhanced orchestrator
        self.orchestrator = EnhancedOrchestrator(self.gemini_mcp_client)
        
        # Initialize enhanced agents
        self.agents = self._initialize_enhanced_agents()
        
        # Register agents with orchestrator
        self._register_agents_with_orchestrator()
        
        # Build enhanced workflow graph
        self.workflow = self._build_enhanced_workflow()
        
        # Try to compile the workflow for compatibility
        try:
            self.compiled_workflow = self.workflow.compile()
            self.logger.info("✅ Workflow compiled successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Workflow compilation failed: {e}, will use fallback execution")
            self.compiled_workflow = None
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'successful_completions': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'error_rate': 0.0
        }
        
        self.initialized = False
    
    def _initialize_enhanced_agents(self) -> Dict[str, Any]:
        """Initialize all enhanced agents with advanced capabilities"""
        return {
            'email': EnhancedEmailAgent(
                self.gemini_mcp_client, 
                agent_name="EnhancedEmailAgent"
            ),
            'calendar': EnhancedCalendarAgent(
                self.gemini_mcp_client,
                agent_name="EnhancedCalendarAgent"
            ),
            'search': EnhancedSearchAgent(
                self.gemini_mcp_client,
                agent_name="EnhancedSearchAgent"
            )
        }
    
    def _register_agents_with_orchestrator(self) -> None:
        """Register all agents with the enhanced orchestrator"""
        agent_capabilities = {
            'email': [
                'email_composition', 'email_search', 'email_summarization',
                'email_classification', 'inbox_management', 'conversation_state_management'
            ],
            'calendar': [
                'calendar_scheduling', 'calendar_search', 'calendar_analysis',
                'availability_checking', 'conflict_resolution', 'schedule_optimization'
            ],
            'search': [
                'web_search', 'research_analysis', 'fact_verification',
                'information_synthesis', 'source_validation'
            ]
        }
        
        for agent_name, capabilities in agent_capabilities.items():
            self.orchestrator.register_agent(
                agent_name, 
                self.agents[agent_name], 
                capabilities
            )
    
    def _build_enhanced_workflow(self) -> StateGraph:
        """Build enhanced LangGraph workflow with advanced routing"""
        workflow = StateGraph(EnhancedAssistantState)
        
        # === CORE WORKFLOW NODES ===
        workflow.add_node("initialize_request", self._initialize_request)
        workflow.add_node("intelligent_routing", self._intelligent_routing)
        workflow.add_node("orchestrated_execution", self._orchestrated_execution)
        workflow.add_node("agent_collaboration", self._agent_collaboration)
        workflow.add_node("quality_verification", self._quality_verification)
        workflow.add_node("error_recovery", self._error_recovery)
        workflow.add_node("performance_optimization", self._performance_optimization)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # === SPECIALIZED AGENT NODES ===
        workflow.add_node("enhanced_email_agent", self._execute_enhanced_email_agent)
        workflow.add_node("enhanced_calendar_agent", self._execute_enhanced_calendar_agent)
        workflow.add_node("enhanced_search_agent", self._execute_enhanced_search_agent)
        
        # === ENTRY POINT ===
        workflow.set_entry_point("initialize_request")
        
        # === ENHANCED ROUTING LOGIC ===
        workflow.add_conditional_edges(
            "initialize_request",
            self._route_initial_decision,
            {
                "orchestrated": "intelligent_routing",
                "direct_email": "enhanced_email_agent",
                "direct_calendar": "enhanced_calendar_agent",
                "direct_search": "enhanced_search_agent",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "intelligent_routing",
            self._route_execution_decision,
            {
                "orchestrated": "orchestrated_execution",
                "email": "enhanced_email_agent",
                "calendar": "enhanced_calendar_agent", 
                "search": "enhanced_search_agent",
                "collaboration": "agent_collaboration",
                "error": "error_recovery"
            }
        )
        
        # === AGENT EXECUTION ROUTING ===
        for agent_node in ["enhanced_email_agent", "enhanced_calendar_agent", "enhanced_search_agent"]:
            workflow.add_conditional_edges(
                agent_node,
                self._route_post_execution,
                {
                    "success": "quality_verification",
                    "collaboration_needed": "agent_collaboration",
                    "escalation_needed": "orchestrated_execution",
                    "error": "error_recovery",
                    "complete": "finalize_response"
                }
            )
        
        # === ADVANCED WORKFLOW ROUTING ===
        workflow.add_conditional_edges(
            "orchestrated_execution",
            self._route_orchestration_result,
            {
                "success": "quality_verification",
                "collaboration": "agent_collaboration",
                "error": "error_recovery",
                "complete": "finalize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "agent_collaboration",
            self._route_collaboration_result,
            {
                "continue": "orchestrated_execution",
                "complete": "quality_verification",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "quality_verification",
            self._route_verification_result,
            {
                "passed": "performance_optimization",
                "failed": "error_recovery",
                "retry": "intelligent_routing"
            }
        )
        
        workflow.add_conditional_edges(
            "error_recovery",
            self._route_recovery_result,
            {
                "retry": "intelligent_routing",
                "fallback": "performance_optimization",
                "escalate": "orchestrated_execution",
                "complete": "finalize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "performance_optimization",
            self._route_optimization_result,
            {
                "complete": "finalize_response",
                "optimize_more": "intelligent_routing"
            }
        )
        
        # === TERMINAL NODE ===
        workflow.add_edge("finalize_response", END)
        
        return workflow
    
    async def initialize_servers(self):
        """Initialize all MCP servers and prepare the system"""
        if not self.initialized:
            try:
                await self.gemini_mcp_client.initialize()
                self.logger.info("✅ Enhanced AI Assistant Workflow initialized successfully")
                self.initialized = True
            except Exception as e:
                self.logger.error(f"❌ Initialization failed: {e}")
                raise
    
    async def process_request(self, user_input: str, user: str = None) -> EnhancedAssistantState:
        """Process user request through enhanced workflow"""
        start_time = time.time()
        
        try:
            # Create enhanced initial state
            initial_state = make_enhanced_initial_state(user_input, user)
            
            # Update performance metrics
            self.performance_metrics['total_requests'] += 1
            
            # Execute workflow - try different methods based on LangGraph version
            result_state = None
            
            try:
                # First try to use the compiled workflow if available
                if self.compiled_workflow:
                    if hasattr(self.compiled_workflow, 'ainvoke'):
                        result_state = await self.compiled_workflow.ainvoke(initial_state)
                    elif hasattr(self.compiled_workflow, 'invoke'):
                        result_state = self.compiled_workflow.invoke(initial_state)
                    else:
                        raise Exception("Compiled workflow has no invoke method")
                        
                # Fallback to raw workflow
                elif hasattr(self.workflow, 'ainvoke'):
                    result_state = await self.workflow.ainvoke(initial_state)
                elif hasattr(self.workflow, 'invoke'):
                    # Try sync invoke if async not available
                    result_state = self.workflow.invoke(initial_state)
                elif hasattr(self.workflow, 'run'):
                    # Try the run method for older versions
                    result_state = self.workflow.run(initial_state)
                else:
                    # Try to compile on the fly
                    try:
                        compiled_workflow = self.workflow.compile()
                        if hasattr(compiled_workflow, 'ainvoke'):
                            result_state = await compiled_workflow.ainvoke(initial_state)
                        elif hasattr(compiled_workflow, 'invoke'):
                            result_state = compiled_workflow.invoke(initial_state)
                        else:
                            raise Exception("No available execution method")
                    except Exception:
                        # Last resort: manual execution through orchestrator
                        result_state = await self.orchestrator.orchestrate(initial_state)
                        
            except Exception as workflow_error:
                self.logger.warning(f"Workflow execution method failed: {workflow_error}, falling back to direct orchestrator")
                result_state = await self.orchestrator.orchestrate(initial_state)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * (self.performance_metrics['total_requests'] - 1) + 
                 processing_time) / self.performance_metrics['total_requests']
            )
            
            if result_state.get('final_response'):
                self.performance_metrics['successful_completions'] += 1
            
            return result_state
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            
            # Create error response
            error_state = make_enhanced_initial_state(user_input, user)
            error_state['final_response'] = "I encountered an unexpected error. Please try again."
            error_state['error_log'] = [{
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'level': 'workflow'
            }]
            
            return error_state
    
    # === WORKFLOW NODE IMPLEMENTATIONS ===
    
    async def _initialize_request(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Initialize and preprocess the request"""
        start_time = time.time()
        
        try:
            # Add session information
            if 'session_context' not in state:
                state['session_context'] = {}
            
            state['session_context']['request_start_time'] = start_time
            state['session_context']['workflow_version'] = 'enhanced_v1'
            
            # Initialize resource tracking
            update_resource_metrics(state, 0, 0, 0, 0)
            
            # Log request
            self.logger.info(f"Processing request: {state.get('user_input', '')[:100]}...")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Request initialization failed: {e}")
            state['error_log'] = [{'error': str(e), 'node': 'initialize_request'}]
            return state
    
    async def _intelligent_routing(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Enhanced intelligent routing with orchestrator"""
        try:
            # Use orchestrator for intelligent routing
            routed_state = await self.orchestrator.orchestrate(state)
            return routed_state
            
        except Exception as e:
            self.logger.error(f"Intelligent routing failed: {e}")
            state['route'] = 'error'
            return state
    
    async def _orchestrated_execution(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Execute complex orchestrated tasks"""
        try:
            # Handle complex multi-agent coordination
            result_state = await self.orchestrator.orchestrate(state)
            return result_state
            
        except Exception as e:
            self.logger.error(f"Orchestrated execution failed: {e}")
            state['route'] = 'error'
            return state
    
    async def _agent_collaboration(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Handle inter-agent collaboration"""
        try:
            collaboration_requests = state.get('collaboration_requests', [])
            
            for request in collaboration_requests:
                # Process collaboration request
                target_agent = self.agents.get(request.target_agent)
                if target_agent:
                    # Execute collaboration
                    collab_state = {
                        'user_input': request.context.get('task_description', ''),
                        'collaboration_context': request.context,
                        'requesting_agent': request.requesting_agent
                    }
                    
                    result = await target_agent.execute_with_full_pipeline(collab_state)
                    
                    # Merge results back
                    collaboration_results = state.get('collaboration_results', [])
                    if not isinstance(collaboration_results, list):
                        collaboration_results = []
                    collaboration_results.append({
                        'request_id': id(request),
                        'result': result,
                        'success': result.get('final_response') is not None
                    })
                    state['collaboration_results'] = collaboration_results
            
            return state
            
        except Exception as e:
            self.logger.error(f"Agent collaboration failed: {e}")
            return state
    
    async def _quality_verification(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Perform quality verification and hallucination detection"""
        try:
            final_response = state.get('final_response', '')
            
            if not final_response:
                state['verification_passed'] = False
                return state
            
            # Perform verification checks
            verification_results = []
            
            # 1. Response completeness check
            completeness_score = self._check_response_completeness(final_response, state)
            verification_results.append(('completeness', completeness_score))
            
            # 2. Context alignment check
            context_score = self._check_context_alignment(final_response, state)
            verification_results.append(('context_alignment', context_score))
            
            # 3. Hallucination detection
            hallucination_score = self._detect_hallucinations(final_response, state)
            verification_results.append(('hallucination_risk', 1.0 - hallucination_score))
            
            # Calculate overall verification score
            overall_score = sum(score for _, score in verification_results) / len(verification_results)
            
            state['verification_score'] = overall_score
            state['verification_details'] = verification_results
            state['verification_passed'] = overall_score > 0.7
            
            return state
            
        except Exception as e:
            self.logger.error(f"Quality verification failed: {e}")
            state['verification_passed'] = False
            return state
    
    async def _error_recovery(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Handle errors with intelligent recovery strategies"""
        try:
            error_log = state.get('error_log', [])
            retry_count = state.get('retry_count', 0)
            
            if retry_count >= 3:
                # Max retries reached, use graceful degradation
                state['final_response'] = "I encountered persistent issues. Please try rephrasing your request or contact support."
                state['recovery_strategy'] = 'graceful_degradation'
                return state
            
            # Determine recovery strategy based on error types
            error_types = [error.get('error', '') for error in error_log]
            
            if any('rate limit' in error.lower() for error in error_types):
                # Rate limit recovery - wait and retry
                state['recovery_strategy'] = 'rate_limit_retry'
                state['retry_count'] = retry_count + 1
                await asyncio.sleep(2)  # Brief wait
                
            elif any('network' in error.lower() or 'timeout' in error.lower() for error in error_types):
                # Network recovery - retry with exponential backoff
                state['recovery_strategy'] = 'network_retry'
                state['retry_count'] = retry_count + 1
                await asyncio.sleep(2 ** retry_count)
                
            else:
                # Generic recovery - try fallback approach
                state['recovery_strategy'] = 'fallback'
                state['fallback_triggered'] = True
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error recovery failed: {e}")
            state['final_response'] = "I encountered multiple errors. Please try again."
            return state
    
    async def _performance_optimization(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Apply performance optimizations and caching"""
        try:
            # Update cache hit rate
            cache_hits = safe_get_resource_metric(state, 'cache_hits', 0)
            total_requests = safe_get_resource_metric(state, 'api_calls', 1)
            cache_hit_rate = cache_hits / max(total_requests, 1)
            
            self.performance_metrics['cache_hit_rate'] = (
                (self.performance_metrics['cache_hit_rate'] * 0.9) + (cache_hit_rate * 0.1)
            )
            
            # Optimization flags
            optimizations_applied = []
            
            # Token usage optimization
            token_count = safe_get_resource_metric(state, 'token_count', 0)
            if token_count > 5000:
                optimizations_applied.append('high_token_usage_detected')
                optimization_flags = state.get('optimization_flags', [])
                if not isinstance(optimization_flags, list):
                    optimization_flags = []
                optimization_flags.append('reduce_token_usage')
                state['optimization_flags'] = optimization_flags
            
            # Response time optimization
            processing_time = safe_get_resource_metric(state, 'processing_time', 0)
            if processing_time > 10.0:
                optimizations_applied.append('slow_response_detected')
                optimization_flags = state.get('optimization_flags', [])
                if not isinstance(optimization_flags, list):
                    optimization_flags = []
                optimization_flags.append('optimize_response_time')
                state['optimization_flags'] = optimization_flags
            
            if optimizations_applied:
                self.logger.info(f"Applied optimizations: {optimizations_applied}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
            return state
    
    async def _finalize_response(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Finalize the response and clean up"""
        try:
            # Ensure we have a final response
            if not state.get('final_response'):
                state['final_response'] = "I was unable to process your request. Please try again."
            
            # Add performance metadata (optional, for debugging)
            if self.logger.isEnabledFor(logging.DEBUG):
                processing_time = safe_get_resource_metric(state, 'processing_time', 0)
                token_count = safe_get_resource_metric(state, 'token_count', 0)
                
                debug_info = f"\n\n[Debug: {processing_time:.2f}s, {token_count} tokens]"
                # Don't add debug info to production responses
                # state['final_response'] += debug_info
            
            # Clean up sensitive information
            if 'error_log' in state:
                # Keep error log for debugging but sanitize sensitive data
                state['error_log'] = [
                    {k: v for k, v in error.items() if k not in ['api_key', 'credentials']}
                    for error in state['error_log']
                ]
            
            return state
            
        except Exception as e:
            self.logger.error(f"Response finalization failed: {e}")
            state['final_response'] = "Response finalization failed."
            return state
    
    # === AGENT EXECUTION NODES ===
    
    async def _execute_enhanced_email_agent(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Execute enhanced email agent"""
        try:
            return await self.agents['email'].execute_with_full_pipeline(state)
        except Exception as e:
            self.logger.error(f"Enhanced email agent failed: {e}")
            state['route'] = 'error'
            return state
    
    async def _execute_enhanced_calendar_agent(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Execute enhanced calendar agent"""
        try:
            return await self.agents['calendar'].execute_with_full_pipeline(state)
        except Exception as e:
            self.logger.error(f"Enhanced calendar agent failed: {e}")
            state['route'] = 'error'
            return state
    
    async def _execute_enhanced_search_agent(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Execute enhanced search agent"""
        try:
            return await self.agents['search'].execute_with_full_pipeline(state)
        except Exception as e:
            self.logger.error(f"Enhanced search agent failed: {e}")
            state['route'] = 'error'
            return state
    
    # === ROUTING DECISION FUNCTIONS ===
    
    def _route_initial_decision(self, state: EnhancedAssistantState) -> str:
        """Route initial decision based on request analysis"""
        user_input = state.get('user_input', '').lower()
        
        # Check for obvious single-agent requests
        if 'email' in user_input and not any(word in user_input for word in ['schedule', 'search', 'find']):
            return "direct_email"
        elif 'calendar' in user_input or 'schedule' in user_input:
            return "direct_calendar"
        elif 'search' in user_input or 'find' in user_input:
            return "direct_search"
        
        # Default to orchestrated for complex requests
        return "orchestrated"
    
    def _route_execution_decision(self, state: EnhancedAssistantState) -> str:
        """Route execution based on orchestrator decision"""
        route = state.get('route', 'orchestrated')
        
        # Check for collaboration needs
        if state.get('collaboration_requests'):
            return "collaboration"
        
        # Check for escalation needs  
        if state.get('escalation_requests'):
            return "orchestrated"
        
        return route
    
    def _route_post_execution(self, state: EnhancedAssistantState) -> str:
        """Route after agent execution"""
        # Check for collaboration requests
        if state.get('collaboration_requests'):
            return "collaboration_needed"
        
        # Check for escalation requests
        if state.get('escalation_requests'):
            return "escalation_needed"
        
        # Check for errors
        if state.get('error_log'):
            return "error"
        
        # Check if response is complete
        if state.get('final_response'):
            return "success"
        
        return "complete"
    
    def _route_orchestration_result(self, state: EnhancedAssistantState) -> str:
        """Route orchestration results"""
        if state.get('collaboration_requests'):
            return "collaboration"
        elif state.get('error_log'):
            return "error"
        elif state.get('final_response'):
            return "success"
        else:
            return "complete"
    
    def _route_collaboration_result(self, state: EnhancedAssistantState) -> str:
        """Route collaboration results"""
        collaboration_results = state.get('collaboration_results', [])
        
        if any(not result['success'] for result in collaboration_results):
            return "error"
        elif state.get('final_response'):
            return "complete"
        else:
            return "continue"
    
    def _route_verification_result(self, state: EnhancedAssistantState) -> str:
        """Route verification results"""
        verification_passed = state.get('verification_passed', False)
        retry_count = state.get('retry_count', 0)
        
        if verification_passed:
            return "passed"
        elif retry_count < 2:
            return "retry"
        else:
            return "failed"
    
    def _route_recovery_result(self, state: EnhancedAssistantState) -> str:
        """Route recovery results"""
        recovery_strategy = state.get('recovery_strategy', 'fallback')
        retry_count = state.get('retry_count', 0)
        
        if recovery_strategy in ['rate_limit_retry', 'network_retry'] and retry_count < 3:
            return "retry"
        elif recovery_strategy == 'fallback':
            return "fallback"
        elif recovery_strategy == 'escalate':
            return "escalate"
        else:
            return "complete"
    
    def _route_optimization_result(self, state: EnhancedAssistantState) -> str:
        """Route optimization results"""
        optimization_flags = state.get('optimization_flags', [])
        
        if 'retry_with_optimization' in optimization_flags:
            return "optimize_more"
        else:
            return "complete"
    
    # === UTILITY METHODS ===
    
    def _check_response_completeness(self, response: str, state: EnhancedAssistantState) -> float:
        """Check if response is complete and addresses the user input"""
        if len(response) < 10:
            return 0.0
        
        user_input = state.get('user_input', '').lower()
        response_lower = response.lower()
        
        # Check if response addresses key terms from input
        user_words = set(user_input.split())
        response_words = set(response_lower.split())
        
        overlap = len(user_words.intersection(response_words))
        return min(overlap / max(len(user_words), 1), 1.0)
    
    def _check_context_alignment(self, response: str, state: EnhancedAssistantState) -> float:
        """Check if response aligns with conversation context"""
        conversation_history = state.get('conversation_history', [])
        
        if len(conversation_history) <= 1:
            return 0.8  # No context to check against
        
        # Simple alignment check - could be enhanced with LLM
        recent_messages = [msg.get('text', '') for msg in conversation_history[-3:]]
        context_text = ' '.join(recent_messages).lower()
        
        response_lower = response.lower()
        
        # Check for context continuity
        context_words = set(context_text.split())
        response_words = set(response_lower.split())
        
        overlap = len(context_words.intersection(response_words))
        return min(overlap / max(len(context_words), 1), 1.0)
    
    def _detect_hallucinations(self, response: str, state: EnhancedAssistantState) -> float:
        """Detect potential hallucinations in response"""
        risk_score = 0.0
        response_lower = response.lower()
        
        # Check for uncertain language
        uncertain_phrases = ['i think', 'maybe', 'probably', 'might be', 'could be']
        for phrase in uncertain_phrases:
            if phrase in response_lower:
                risk_score += 0.1
        
        # Check for specific claims without evidence
        definitive_phrases = ['definitely', 'certainly', 'always', 'never', 'all']
        for phrase in definitive_phrases:
            if phrase in response_lower:
                risk_score += 0.05
        
        return min(1.0 - risk_score, 1.0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            'workflow_initialized': self.initialized,
            'agents_healthy': True,
            'orchestrator_healthy': True,
            'performance_metrics': self.performance_metrics
        }
        
        # Check agent health
        try:
            for agent_name, agent in self.agents.items():
                # Simple health check - could be enhanced
                health_status[f'{agent_name}_agent'] = hasattr(agent, 'execute_with_full_pipeline')
        except Exception as e:
            health_status['agents_healthy'] = False
            health_status['agent_error'] = str(e)
        
        return health_status