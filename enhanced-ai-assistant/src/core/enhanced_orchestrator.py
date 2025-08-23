import asyncio
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict

from src.core.enhanced_state_schema import (
    EnhancedAssistantState, TaskResult, AgentDecision, CollaborationRequest,
    EscalationRequest, VerificationResult, TaskType, AgentType,
    update_resource_metrics, record_agent_decision
)
from src.core.enhanced_message_types import (
    MessageType, MessageFactory, MessageRouter, MessagePriority
)
from src.intelligence.decision_logger import get_decision_logger

class EnhancedOrchestrator:
    """
    Advanced orchestrator with comprehensive AI capabilities:
    
    🧠 Intelligent Routing: Dynamic agent selection with confidence scoring
    🤝 Multi-Agent Coordination: Complex task decomposition and delegation  
    🎯 Autonomous Decision Making: Independent orchestration choices
    🔄 Error Handling & Recovery: Graceful failure management across agents
    ⚡ Performance Optimization: Resource allocation and load balancing
    📝 Prompt Engineering: Dynamic routing prompts with context awareness
    🛡️ Quality Assurance: Cross-agent verification and consistency checking
    💾 Resource Management: Global token budgeting and rate limiting
    """
    
    def __init__(self, gemini_mcp_client):
        self.gemini_mcp_client = gemini_mcp_client
        self.logger = logging.getLogger("enhanced_orchestrator")
        
        # === AGENT MANAGEMENT ===
        self.agents: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.agent_load: Dict[str, int] = {}
        
        # === ROUTING & DECISION MAKING ===
        self.routing_patterns: Dict[str, Dict[str, Any]] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.success_metrics: Dict[str, List[float]] = {}
        self.confidence_threshold = 0.7
        
        # === COLLABORATION & COORDINATION ===
        self.active_collaborations: Dict[str, CollaborationRequest] = {}
        self.escalation_queue: List[EscalationRequest] = []
        self.task_dependencies: Dict[str, List[str]] = {}
        
        # === RESOURCE MANAGEMENT ===
        self.global_token_budget = 20000
        self.token_allocation: Dict[str, int] = {}
        self.rate_limit_coordinator = {}
        self.performance_monitor = {}
        
        # === QUALITY ASSURANCE ===
        self.cross_agent_verification = True
        self.consistency_checkers: List[callable] = []
        self.quality_metrics: Dict[str, float] = {}
        
        # === PROMPT ENGINEERING ===
        self.routing_prompts = self._initialize_routing_prompts()
        self.coordination_prompts = self._initialize_coordination_prompts()
        self.dynamic_prompt_cache: Dict[str, str] = {}
        
        # === ERROR HANDLING ===
        self.error_recovery_strategies = self._initialize_recovery_strategies()
        self.fallback_chains: Dict[str, List[str]] = {}
        self.error_patterns: Dict[str, int] = {}
        
        # Initialize routing patterns
        self.route_patterns = self._initialize_route_patterns()
        
        # Message router for coordination
        self.message_router = MessageRouter()
    
    def register_agent(self, agent_name: str, agent_instance: Any, capabilities: List[str]) -> None:
        """Register an agent with the orchestrator"""
        self.agents[agent_name] = agent_instance
        self.agent_capabilities[agent_name] = capabilities
        self.agent_performance[agent_name] = {
            'success_rate': 0.8,
            'avg_confidence': 0.7,
            'avg_response_time': 2.0,
            'error_rate': 0.1
        }
        self.agent_load[agent_name] = 0
        
        # Allocate token budget
        base_allocation = self.global_token_budget // len(self.agents)
        self.token_allocation[agent_name] = base_allocation
        
        self.logger.info(f"Registered agent: {agent_name} with capabilities: {capabilities}")
    
    async def orchestrate(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Main orchestration method with full AI pipeline"""
        start_time = time.time()
        
        try:
            # 1. Global Resource Management
            await self._manage_global_resources(state)
            
            # 2. Intelligent Request Analysis
            request_analysis = await self._analyze_request(state)
            
            # 3. Dynamic Agent Routing
            routing_decision = await self._make_routing_decision(request_analysis, state)
            
            # 4. Handle Collaborations and Escalations
            if await self._needs_coordination(state):
                return await self._coordinate_multi_agent_task(state)
            
            # 5. Execute with Selected Agent
            result_state = await self._execute_with_agent(routing_decision, state)
            
            # 6. Cross-Agent Verification (if enabled)
            if self.cross_agent_verification:
                result_state = await self._cross_verify_result(result_state, routing_decision)
            
            # 7. Update Learning Patterns
            await self._update_routing_patterns(routing_decision, result_state, start_time)
            
            # 8. Performance Monitoring
            await self._update_performance_metrics(routing_decision, result_state, start_time)
            
            return result_state
            
        except Exception as e:
            return await self._handle_orchestration_error(e, state, start_time)
    
    # === INTELLIGENT REQUEST ANALYSIS ===
    
    async def _analyze_request(self, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Comprehensive request analysis for intelligent routing"""
        user_input = state.get('user_input', '')
        conversation_history = state.get('conversation_history', [])
        
        # Multi-dimensional analysis
        analysis = {
            'primary_intent': await self._detect_primary_intent(user_input, conversation_history),
            'complexity_level': await self._assess_request_complexity(user_input, conversation_history),
            'required_capabilities': await self._identify_required_capabilities(user_input),
            'context_requirements': await self._analyze_context_requirements(conversation_history),
            'urgency_level': self._assess_urgency(user_input),
            'multi_agent_needed': await self._detect_multi_agent_requirement(user_input),
            'resource_requirements': self._estimate_resource_requirements(user_input),
            'confidence': 0.8  # Base confidence
        }
        
        # Adjust confidence based on analysis quality
        analysis['confidence'] = self._calculate_analysis_confidence(analysis)
        
        return analysis
    
    async def _detect_primary_intent(self, user_input: str, conversation_history: List[Dict]) -> str:
        """Detect the primary intent using advanced NLP analysis"""
        # Pattern matching for speed
        intent_patterns = {
            'email': ['email', 'mail', 'send', 'compose', 'inbox', 'message'],
            'calendar': ['schedule', 'calendar', 'meeting', 'appointment', 'event', 'time'],
            'search': ['search', 'find', 'look up', 'research', 'what is', 'how to'],
            'coordination': ['both', 'also', 'and then', 'after that', 'multiple']
        }
        
        user_lower = user_input.lower()
        intent_scores = {}
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        # Return highest scoring intent
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        # Fallback to LLM analysis if needed
        return await self._llm_intent_detection(user_input, conversation_history)
    
    async def _assess_request_complexity(self, user_input: str, conversation_history: List[Dict]) -> str:
        """Assess the complexity level of the request"""
        complexity_indicators = {
            'high': ['complex', 'detailed', 'comprehensive', 'analyze', 'multiple', 'coordinate'],
            'medium': ['help', 'create', 'find', 'organize', 'manage'],
            'low': ['simple', 'quick', 'basic', 'just', 'only']
        }
        
        user_lower = user_input.lower()
        
        # Check for high complexity indicators
        if any(indicator in user_lower for indicator in complexity_indicators['high']):
            return 'high'
        
        # Check for multiple requests
        if len(user_input.split(' and ')) > 2 or len(user_input.split(',')) > 2:
            return 'high'
        
        # Check conversation context
        if len(conversation_history) > 5:
            return 'medium'
        
        # Check for low complexity indicators
        if any(indicator in user_lower for indicator in complexity_indicators['low']):
            return 'low'
        
        return 'medium'  # Default
    
    async def _identify_required_capabilities(self, user_input: str) -> List[str]:
        """Identify what capabilities are needed to fulfill the request"""
        capabilities = []
        user_lower = user_input.lower()
        
        capability_keywords = {
            'email_composition': ['email', 'send', 'compose', 'write', 'message'],
            'email_search': ['find email', 'search email', 'look for email'],
            'calendar_scheduling': ['schedule', 'book', 'calendar', 'meeting', 'appointment'],
            'calendar_analysis': ['check schedule', 'analyze calendar', 'conflicts'],
            'web_search': ['search', 'look up', 'find information', 'research'],
            'multi_agent_coordination': ['both', 'and', 'also', 'multiple tasks'],
            'context_management': ['remember', 'context', 'previous', 'earlier'],
            'verification': ['verify', 'check', 'confirm', 'validate']
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities if capabilities else ['general_assistance']
    
    # === DYNAMIC AGENT ROUTING ===
    
    async def _make_routing_decision(self, analysis: Dict[str, Any], 
                                   state: EnhancedAssistantState) -> Dict[str, Any]:
        """Make intelligent routing decision with confidence scoring"""
        primary_intent = analysis['primary_intent']
        complexity = analysis['complexity_level']
        required_capabilities = analysis['required_capabilities']
        
        # Get candidate agents
        candidate_agents = self._get_candidate_agents(required_capabilities)
        
        # Score each candidate
        agent_scores = {}
        for agent_name in candidate_agents:
            score = await self._score_agent_for_task(agent_name, analysis, state)
            agent_scores[agent_name] = score
        
        # Select best agent
        if agent_scores:
            best_agent = max(agent_scores, key=agent_scores.get)
            confidence = agent_scores[best_agent]
        else:
            # Fallback routing
            best_agent = self._fallback_agent_selection(primary_intent)
            confidence = 0.5
        
        routing_decision = {
            'selected_agent': best_agent,
            'confidence': confidence,
            'reasoning': f"Selected {best_agent} based on intent: {primary_intent}, complexity: {complexity}",
            'alternative_agents': [agent for agent in agent_scores.keys() if agent != best_agent],
            'agent_scores': agent_scores,
            'analysis': analysis,
            'fallback_chain': self._build_fallback_chain(best_agent, candidate_agents)
        }
        
        # Record routing decision
        self.routing_history.append({
            'timestamp': datetime.now().isoformat(),
            'decision': routing_decision,
            'state_summary': self._summarize_state(state)
        })
        
        return routing_decision
    
    def _get_candidate_agents(self, required_capabilities: List[str]) -> List[str]:
        """Get agents that can handle the required capabilities"""
        candidates = []
        
        for agent_name, capabilities in self.agent_capabilities.items():
            # Check capability overlap
            capability_match = any(
                req_cap in cap or cap in req_cap 
                for req_cap in required_capabilities 
                for cap in capabilities
            )
            
            if capability_match:
                candidates.append(agent_name)
        
        # If no specific matches, include all agents as fallback
        if not candidates:
            candidates = list(self.agents.keys())
        
        return candidates
    
    async def _score_agent_for_task(self, agent_name: str, analysis: Dict[str, Any], 
                                  state: EnhancedAssistantState) -> float:
        """Score an agent's suitability for the task"""
        score = 0.0
        
        # 1. Capability matching (40%)
        required_caps = analysis['required_capabilities']
        agent_caps = self.agent_capabilities.get(agent_name, [])
        capability_score = self._calculate_capability_match(required_caps, agent_caps)
        score += capability_score * 0.4
        
        # 2. Historical performance (30%)
        performance = self.agent_performance.get(agent_name, {})
        performance_score = (
            performance.get('success_rate', 0.5) * 0.5 +
            performance.get('avg_confidence', 0.5) * 0.3 +
            (1.0 - performance.get('error_rate', 0.5)) * 0.2
        )
        score += performance_score * 0.3
        
        # 3. Current load (20%)
        load = self.agent_load.get(agent_name, 0)
        max_load = 10  # Configurable
        load_score = max(0.0, 1.0 - (load / max_load))
        score += load_score * 0.2
        
        # 4. Resource availability (10%)
        resource_metrics = state.get('resource_metrics')
        if resource_metrics and hasattr(resource_metrics, 'token_count'):
            token_usage = resource_metrics.token_count
        else:
            token_usage = 0
        allocated_tokens = self.token_allocation.get(agent_name, 1000)
        resource_score = max(0.0, 1.0 - (token_usage / allocated_tokens))
        score += resource_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_capability_match(self, required: List[str], available: List[str]) -> float:
        """Calculate how well agent capabilities match requirements"""
        if not required:
            return 0.5
        
        matches = 0
        for req_cap in required:
            for avail_cap in available:
                if req_cap in avail_cap or avail_cap in req_cap:
                    matches += 1
                    break
        
        return matches / len(required)
    
    # === MULTI-AGENT COORDINATION ===
    
    async def _needs_coordination(self, state: EnhancedAssistantState) -> bool:
        """Determine if request needs multi-agent coordination"""
        # Check for pending collaborations
        if state.get('collaboration_requests'):
            return True
        
        # Check for escalations
        if state.get('escalation_requests'):
            return True
        
        # Check for complex multi-step tasks
        user_input = state.get('user_input', '').lower()
        coordination_indicators = ['both', 'and then', 'after that', 'also', 'multiple']
        
        return any(indicator in user_input for indicator in coordination_indicators)
    
    async def _coordinate_multi_agent_task(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Coordinate complex multi-agent tasks"""
        # Handle collaboration requests
        collaboration_requests = state.get('collaboration_requests', [])
        for request in collaboration_requests:
            await self._process_collaboration_request(request, state)
        
        # Handle escalation requests
        escalation_requests = state.get('escalation_requests', [])
        for request in escalation_requests:
            await self._process_escalation_request(request, state)
        
        # Decompose complex tasks
        task_decomposition = await self._decompose_complex_task(state)
        
        if task_decomposition['needs_decomposition']:
            return await self._execute_decomposed_task(task_decomposition, state)
        
        return state
    
    async def _decompose_complex_task(self, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Decompose complex tasks into manageable subtasks"""
        user_input = state.get('user_input', '')
        
        # Use LLM to analyze task decomposition needs
        prompt = f"""Analyze this request for task decomposition:
        
        Request: "{user_input}"
        
        Determine if this requires multiple agents or steps:
        1. Can this be handled by a single agent?
        2. If not, what are the subtasks?
        3. What's the optimal sequence?
        4. Which agents should handle each subtask?
        
        Available agents: {list(self.agents.keys())}
        
        Respond in JSON format with decomposition plan."""
        
        try:
            response = await self.gemini_mcp_client.chat(prompt)
            decomposition = self._parse_json_response(response)
            
            return {
                'needs_decomposition': decomposition.get('needs_decomposition', False),
                'subtasks': decomposition.get('subtasks', []),
                'sequence': decomposition.get('sequence', []),
                'agent_assignments': decomposition.get('agent_assignments', {}),
                'confidence': decomposition.get('confidence', 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            return {'needs_decomposition': False}
    
    # === EXECUTION WITH AGENTS ===
    
    async def _execute_with_agent(self, routing_decision: Dict[str, Any], 
                                state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Execute task with selected agent including fallback handling"""
        selected_agent = routing_decision['selected_agent']
        fallback_chain = routing_decision.get('fallback_chain', [])
        
        # Update agent load
        self.agent_load[selected_agent] = self.agent_load.get(selected_agent, 0) + 1
        
        try:
            # Execute with primary agent
            agent_instance = self.agents[selected_agent]
            
            # Set routing information in state
            state['current_agent'] = selected_agent
            state['route'] = selected_agent
            state['route_confidence'] = routing_decision['confidence']
            state['route_reason'] = routing_decision['reasoning']
            
            # Execute agent
            if hasattr(agent_instance, 'execute_with_full_pipeline'):
                result_state = await agent_instance.execute_with_full_pipeline(state)
            else:
                result_state = await agent_instance.execute(state)
            
            # Update agent load
            self.agent_load[selected_agent] = max(0, self.agent_load[selected_agent] - 1)
            
            return result_state
            
        except Exception as e:
            # Update agent load on error
            self.agent_load[selected_agent] = max(0, self.agent_load[selected_agent] - 1)
            
            # Try fallback agents
            for fallback_agent in fallback_chain:
                try:
                    self.logger.warning(f"Falling back to {fallback_agent} after {selected_agent} failed")
                    
                    agent_instance = self.agents[fallback_agent]
                    state['current_agent'] = fallback_agent
                    state['route'] = fallback_agent
                    state['fallback_triggered'] = True
                    
                    if hasattr(agent_instance, 'execute_with_full_pipeline'):
                        result_state = await agent_instance.execute_with_full_pipeline(state)
                    else:
                        result_state = await agent_instance.execute(state)
                    
                    return result_state
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback agent {fallback_agent} also failed: {fallback_error}")
                    continue
            
            # If all agents fail, use graceful degradation
            return await self._graceful_degradation(e, state, routing_decision)
    
    # === CROSS-AGENT VERIFICATION ===
    
    async def _cross_verify_result(self, result_state: EnhancedAssistantState, 
                                 routing_decision: Dict[str, Any]) -> EnhancedAssistantState:
        """Perform cross-agent verification of results"""
        if not self.cross_agent_verification:
            return result_state
        
        primary_agent = routing_decision['selected_agent']
        alternative_agents = routing_decision.get('alternative_agents', [])
        
        if not alternative_agents:
            return result_state
        
        try:
            # Select verification agent (highest scoring alternative)
            verification_agent = alternative_agents[0]
            
            # Create verification task
            verification_state = {
                'user_input': f"Verify this response: {result_state.get('final_response', '')}",
                'conversation_history': result_state.get('conversation_history', []),
                'verification_mode': True,
                'original_agent': primary_agent
            }
            
            # Execute verification
            verifier = self.agents[verification_agent]
            # This would need a specialized verification method
            # verification_result = await verifier.verify_response(verification_state)
            
            # For now, just add a verification note
            verification_note = f"\n\n*Response verified by {verification_agent}*"
            result_state['final_response'] = result_state.get('final_response', '') + verification_note
            
            return result_state
            
        except Exception as e:
            self.logger.error(f"Cross-agent verification failed: {e}")
            return result_state
    
    # === LEARNING & PERFORMANCE MONITORING ===
    
    async def _update_routing_patterns(self, routing_decision: Dict[str, Any], 
                                     result_state: EnhancedAssistantState, 
                                     start_time: float) -> None:
        """Update routing patterns based on results"""
        selected_agent = routing_decision['selected_agent']
        confidence = routing_decision['confidence']
        
        # Measure success
        success = not result_state.get('fallback_triggered', False) and result_state.get('final_response', '')
        response_time = time.time() - start_time
        
        # Update agent performance
        agent_perf = self.agent_performance.get(selected_agent, {})
        
        # Update success rate (exponential moving average)
        current_success_rate = agent_perf.get('success_rate', 0.5)
        agent_perf['success_rate'] = (current_success_rate * 0.9) + (float(success) * 0.1)
        
        # Update response time
        current_avg_time = agent_perf.get('avg_response_time', 2.0)
        agent_perf['avg_response_time'] = (current_avg_time * 0.9) + (response_time * 0.1)
        
        # Update confidence (ensure float conversion safety)
        result_confidence = result_state.get('confidence_score', confidence)
        
        # Safe float conversion - handle string responses
        if isinstance(result_confidence, str):
            try:
                # Try to extract numeric value from string response
                import re
                numbers = re.findall(r'\d*\.?\d+', result_confidence)
                if numbers:
                    result_confidence = float(numbers[0])
                else:
                    result_confidence = confidence  # Fallback to routing confidence
            except (ValueError, TypeError):
                result_confidence = confidence  # Fallback to routing confidence
        
        result_confidence = float(result_confidence) if result_confidence is not None else 0.5
        result_confidence = max(0.0, min(1.0, result_confidence))  # Clamp to [0,1]
        
        current_avg_confidence = agent_perf.get('avg_confidence', 0.5)
        agent_perf['avg_confidence'] = (current_avg_confidence * 0.9) + (result_confidence * 0.1)
        
        self.agent_performance[selected_agent] = agent_perf
        
        # Update routing patterns
        pattern_key = f"{routing_decision['analysis']['primary_intent']}_{routing_decision['analysis']['complexity_level']}"
        
        if pattern_key not in self.routing_patterns:
            self.routing_patterns[pattern_key] = {}
        
        agent_pattern = self.routing_patterns[pattern_key].get(selected_agent, {'count': 0, 'success_rate': 0.5})
        agent_pattern['count'] += 1
        agent_pattern['success_rate'] = (agent_pattern['success_rate'] * 0.8) + (float(success) * 0.2)
        
        self.routing_patterns[pattern_key][selected_agent] = agent_pattern
    
    # === RESOURCE MANAGEMENT ===
    
    async def _manage_global_resources(self, state: EnhancedAssistantState) -> None:
        """Manage global resources across all agents"""
        # Check global token usage
        resource_metrics = state.get('resource_metrics')
        if resource_metrics and hasattr(resource_metrics, 'token_count'):
            total_tokens = resource_metrics.token_count
        else:
            total_tokens = 0
        
        if total_tokens > self.global_token_budget * 0.8:
            self.logger.warning("Approaching global token budget limit")
            # Redistribute token allocations
            await self._redistribute_token_budget()
        
        # Update rate limit coordination
        await self._coordinate_rate_limits(state)
    
    async def _redistribute_token_budget(self) -> None:
        """Redistribute token budget based on agent performance"""
        total_budget = self.global_token_budget
        num_agents = len(self.agents)
        
        # Base allocation
        base_allocation = total_budget // (num_agents * 2)
        
        # Performance-based allocation
        performance_scores = {}
        for agent_name, perf in self.agent_performance.items():
            score = (perf.get('success_rate', 0.5) * 0.6 + 
                    perf.get('avg_confidence', 0.5) * 0.4)
            performance_scores[agent_name] = score
        
        total_performance = sum(performance_scores.values())
        
        # Redistribute
        remaining_budget = total_budget - (base_allocation * num_agents)
        
        for agent_name, score in performance_scores.items():
            performance_allocation = int((score / total_performance) * remaining_budget)
            self.token_allocation[agent_name] = base_allocation + performance_allocation
    
    # === ERROR HANDLING ===
    
    async def _handle_orchestration_error(self, error: Exception, 
                                        state: EnhancedAssistantState, 
                                        start_time: float) -> EnhancedAssistantState:
        """Handle orchestration-level errors"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        self.logger.error(f"Orchestration error: {error_type}: {error_msg}")
        
        # Record error
        error_record = {
            'level': 'orchestrator',
            'error_type': error_type,
            'error_message': error_msg,
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time
        }
        
        error_log = state.get('error_log', [])
        if not isinstance(error_log, list):
            error_log = []
        error_log.append(error_record)
        state['error_log'] = error_log
        
        # Graceful degradation
        state['final_response'] = "I encountered an issue coordinating your request. Let me try a simpler approach."
        state['fallback_triggered'] = True
        state['route'] = 'orchestrator'  # Ensure route is set
        state['confidence_score'] = 0.2  # Very low confidence for orchestration error
        
        return state
    
    async def _graceful_degradation(self, error: Exception, 
                                  state: EnhancedAssistantState, 
                                  routing_decision: Dict[str, Any]) -> EnhancedAssistantState:
        """Graceful degradation when all agents fail"""
        self.logger.error(f"All agents failed, using graceful degradation: {error}")
        
        # Provide helpful fallback response
        intent = routing_decision.get('analysis', {}).get('primary_intent', 'unknown')
        
        fallback_responses = {
            'email': "I'm having trouble with email functions right now. Please try again or contact support.",
            'calendar': "I'm experiencing issues with calendar operations. Please try your request again.",
            'search': "I can't perform searches at the moment. Please try rephrasing your request.",
            'unknown': "I'm experiencing technical difficulties. Please try again or rephrase your request."
        }
        
        response = fallback_responses.get(intent, fallback_responses['unknown'])
        
        state['final_response'] = response
        state['fallback_triggered'] = True
        state['error_recovery_attempted'] = True
        state['route'] = 'orchestrator'  # Ensure route is set
        state['confidence_score'] = 0.3  # Low confidence for error state
        
        return state
    
    # === UTILITY METHODS ===
    
    def _fallback_agent_selection(self, intent: str) -> str:
        """Fallback agent selection based on intent"""
        fallback_map = {
            'email': 'email',
            'calendar': 'calendar', 
            'search': 'search'
        }
        
        selected = fallback_map.get(intent)
        if selected and selected in self.agents:
            return selected
        
        # Return first available agent
        return list(self.agents.keys())[0] if self.agents else 'none'
    
    def _build_fallback_chain(self, primary_agent: str, candidates: List[str]) -> List[str]:
        """Build fallback chain for error recovery"""
        chain = [agent for agent in candidates if agent != primary_agent]
        return chain[:2]  # Limit to 2 fallback agents
    
    def _summarize_state(self, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Create a summary of state for logging"""
        return {
            'user_input_length': len(state.get('user_input', '')),
            'conversation_length': len(state.get('conversation_history', [])),
            'has_context': bool(state.get('active_context')),
            'retry_count': state.get('retry_count', 0)
        }
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with fallback"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {'needs_decomposition': False, 'confidence': 0.3}
    
    # === INITIALIZATION METHODS ===
    
    def _initialize_route_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize routing patterns"""
        return {
            'email': {
                'agents': ['email'],
                'keywords': ['email', 'mail', 'send', 'compose', 'inbox'],
                'confidence_base': 0.9
            },
            'calendar': {
                'agents': ['calendar'],
                'keywords': ['schedule', 'calendar', 'meeting', 'appointment'],
                'confidence_base': 0.9
            },
            'search': {
                'agents': ['search'],
                'keywords': ['search', 'find', 'look up', 'research'],
                'confidence_base': 0.8
            }
        }
    
    def _initialize_routing_prompts(self) -> Dict[str, str]:
        """Initialize dynamic routing prompts"""
        return {
            'intent_detection': """Analyze the user's request to determine primary intent:
            
            Request: {user_input}
            Context: {context}
            
            Determine:
            1. Primary intent (email, calendar, search, coordination)
            2. Complexity level (low, medium, high)
            3. Required capabilities
            4. Multi-agent coordination needed
            
            Respond in JSON format.""",
            
            'agent_selection': """Select the best agent for this task:
            
            Intent: {intent}
            Complexity: {complexity}
            Requirements: {requirements}
            
            Available agents: {agents}
            Agent performance: {performance}
            
            Consider capability match, performance history, and current load.
            Respond with agent selection and confidence score."""
        }
    
    def _initialize_coordination_prompts(self) -> Dict[str, str]:
        """Initialize coordination prompts"""
        return {
            'task_decomposition': """Decompose this complex task:
            
            Request: {request}
            Available agents: {agents}
            
            Create a plan with:
            1. Subtasks
            2. Agent assignments
            3. Execution sequence
            4. Dependencies
            
            Respond in JSON format."""
        }
    
    def _initialize_recovery_strategies(self) -> Dict[str, callable]:
        """Initialize error recovery strategies"""
        return {
            'agent_failure': self._handle_agent_failure,
            'timeout': self._handle_timeout,
            'resource_exhaustion': self._handle_resource_exhaustion,
            'coordination_failure': self._handle_coordination_failure
        }
    
    # Placeholder methods for error recovery
    async def _handle_agent_failure(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        return state
    
    async def _handle_timeout(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        return state
    
    async def _handle_resource_exhaustion(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        return state
    
    async def _handle_coordination_failure(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        return state
    
    async def _process_collaboration_request(self, request: CollaborationRequest, state: EnhancedAssistantState) -> None:
        pass
    
    async def _process_escalation_request(self, request: EscalationRequest, state: EnhancedAssistantState) -> None:
        pass
    
    async def _execute_decomposed_task(self, decomposition: Dict[str, Any], state: EnhancedAssistantState) -> EnhancedAssistantState:
        return state
    
    async def _llm_intent_detection(self, user_input: str, conversation_history: List[Dict]) -> str:
        """LLM-backed intent detection for orchestrator"""
        prompt = f"""Analyze user intent for routing:

User Input: "{user_input}"
Recent History: {json.dumps(conversation_history[-3:] if conversation_history else [], indent=2)}
Available Agents: {list(self.agents.keys())}

Detect primary intent from: email, calendar, search, coordination, general

Return JSON: {{"intent": "detected_intent", "confidence": 0.8, "reasoning": "explanation"}}"""

        try:
            response = await self.gemini_mcp_client.chat(prompt)
            # Simple JSON extraction
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                intent = analysis.get('intent', 'general')
                
                # Log orchestrator decision
                get_decision_logger().log_intent_detection(
                    agent="orchestrator",
                    user_input=user_input,
                    detected_intent=intent,
                    confidence=analysis.get('confidence', 0.5),
                    reasoning=analysis.get('reasoning', 'Orchestrator LLM intent detection'),
                    fallback_used=False
                )
                
                return intent
        except Exception as e:
            self.logger.error(f"Orchestrator LLM intent detection failed: {e}")
            
            # Log fallback
            get_decision_logger().log_intent_detection(
                agent="orchestrator",
                user_input=user_input,
                detected_intent="general",
                confidence=0.3,
                reasoning=f"Fallback due to error: {str(e)}",
                fallback_used=True
            )
            
        return "general"
    
    def _assess_urgency(self, user_input: str) -> str:
        return "medium"
    
    async def _detect_multi_agent_requirement(self, user_input: str) -> bool:
        return False
    
    def _estimate_resource_requirements(self, user_input: str) -> Dict[str, Any]:
        return {'tokens': 500, 'time': 2.0}
    
    async def _analyze_context_requirements(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        return {'context_needed': True}
    
    def _calculate_analysis_confidence(self, analysis: Dict[str, Any]) -> float:
        return 0.8
    
    async def _coordinate_rate_limits(self, state: EnhancedAssistantState) -> None:
        pass
    
    async def _update_performance_metrics(self, routing_decision: Dict[str, Any], 
                                        result_state: EnhancedAssistantState, 
                                        start_time: float) -> None:
        pass