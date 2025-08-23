import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import asdict

from src.core.enhanced_state_schema import (
    EnhancedAssistantState, TaskResult, AgentDecision, CollaborationRequest,
    EscalationRequest, VerificationResult, ContextCache, TaskType, 
    ConfidenceLevel, TaskComplexity,
    update_resource_metrics, add_collaboration_request, add_escalation_request,
    record_agent_decision, update_verification_result, get_context_cache,
    set_context_cache, is_rate_limited, increment_rate_limit_counter
)
from src.core.enhanced_message_types import (
    AgentMessage, MessageType, MessagePriority, MessageFactory, MessageRouter
)
from src.intelligence.decision_logger import get_decision_logger
from src.intelligence.pipeline_methods import PipelineMethods
from src.intelligence.agent_helpers import can_handle_task, suggest_routing

class EnhancedBaseAgent(PipelineMethods, ABC):
    """
    Advanced base agent with comprehensive AI capabilities:
    
    🧠 Intelligent Routing: Dynamic task analysis and agent selection
    🤝 Bidirectional Communication: Inter-agent collaboration and escalation
    🎯 Autonomous Decision Making: Independent choices with confidence scoring
    🔄 Error Handling & Recovery: Graceful failure management and auto-retry
    ⚡ Performance Optimization: Context caching and resource management
    📝 Prompt Engineering: Dynamic prompt optimization
    🛡️ Hallucination Mitigation: Verification workflows and fact-checking
    💾 Resource Management: Token budgeting and rate limiting
    """
    
    def __init__(self, mcp_client, agent_name: str, capabilities: List[str]):
        self.mcp_client = mcp_client
        self.agent_name = agent_name
        self.capabilities = capabilities
        
        # Core Systems
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.message_router = MessageRouter()
        
        # === RESOURCE MANAGEMENT ===
        self.token_budget = 8000
        self.request_count = 0
        self.last_reset = datetime.now()
        self.rate_limits = {
            'gemini': {'limit': 15, 'window': 60},  # 15 requests per minute
            'mcp_tools': {'limit': 100, 'window': 60}
        }
        
        # === CONTEXT & CACHING ===
        self.context_cache: Dict[str, ContextCache] = {}
        self.cache_ttl = 300  # 5 minutes
        self.performance_cache: Dict[str, Any] = {}
        
        # === DECISION MAKING & LEARNING ===
        self.decision_patterns: Dict[str, float] = {}
        self.success_metrics: Dict[str, List[float]] = {}
        self.confidence_threshold = 0.7
        
        # === VERIFICATION & QUALITY CONTROL ===
        self.verification_enabled = True
        self.hallucination_detectors = self._initialize_hallucination_detectors()
        self.fact_checkers = self._initialize_fact_checkers()
        
        # === PROMPT ENGINEERING ===
        self.prompt_templates = self._initialize_prompt_templates()
        self.dynamic_prompts = {}
        self.prompt_performance = {}
        
        # === ERROR HANDLING ===
        self.retry_config = {
            'max_retries': 3,
            'backoff_factor': 2,
            'rate_limit_wait': 8,
            'timeout': 30
        }
        self.fallback_strategies = self._initialize_fallback_strategies()
        
        # === COLLABORATION ===
        self.peer_agents: Dict[str, 'EnhancedBaseAgent'] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    async def execute(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Main execution method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_task_types(self) -> List[TaskType]:
        """Return task types this agent can handle"""
        pass
    
    # === CORE EXECUTION PIPELINE ===
    
    async def execute_with_full_pipeline(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Execute with full AI pipeline including all advanced features"""
        start_time = time.time()
        
        try:
            # 1. Resource Management
            await self._manage_resources(state)
            
            # 2. Context Analysis & Caching
            context = await self._analyze_context(state)
            
            # 3. Task Analysis & Routing
            task_analysis = await self._analyze_task(state, context)
            
            # 4. Autonomous Decision Making
            decision = await self._make_autonomous_decision(task_analysis, context, state)
            record_agent_decision(state, decision)
            
            # 5. Execute with Verification
            result = await self._execute_with_verification(decision, state, context)
            
            # 6. Quality Assurance & Hallucination Check
            if self.verification_enabled:
                result = await self._verify_result(result, state, context)
            
            # 7. Collaboration & Escalation Handling
            if result.needs_escalation or result.needs_help_from:
                return await self._handle_collaboration(result, state, context)
            
            # 8. Learning & Pattern Updates
            await self._update_learning_patterns(decision, result, state)
            
            # 9. Performance Metrics
            processing_time = time.time() - start_time
            update_resource_metrics(state, 0, 0, 0, processing_time)
            
            return self._format_final_response(result, state)
            
        except Exception as e:
            return await self._handle_error_with_recovery(e, state, start_time)
    
    # === RESOURCE MANAGEMENT ===
    
    async def _manage_resources(self, state: EnhancedAssistantState) -> None:
        """Comprehensive resource management"""
        current_time = datetime.now()
        
        # Reset rate limit counters if needed
        if (current_time - self.last_reset).seconds > 60:
            self.request_count = 0
            self.last_reset = current_time
        
        # Check rate limits
        for service, limits in self.rate_limits.items():
            if is_rate_limited(state, service):
                self.logger.warning(f"Rate limit hit for {service}, waiting...")
                await asyncio.sleep(self.retry_config['rate_limit_wait'])
        
        # Clean expired cache
        await self._clean_expired_cache()
        
        # Token budget management
        from src.core.enhanced_state_schema import safe_get_resource_metric
        current_usage = safe_get_resource_metric(state, 'token_count', 0)
        if current_usage > self.token_budget * 0.8:
            self.logger.warning("Approaching token budget limit")
            # Enable aggressive caching
            self.cache_ttl = 600  # 10 minutes
    
    async def _clean_expired_cache(self) -> None:
        """Clean expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, cached_item in self.context_cache.items():
            cache_time = datetime.fromisoformat(cached_item.timestamp)
            if (current_time - cache_time).seconds > cached_item.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.context_cache[key]
    
    # === CONTEXT ANALYSIS & CACHING ===
    
    async def _analyze_context(self, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Advanced context analysis with intelligent caching"""
        conversation_history = state.get('conversation_history', [])
        user_input = state.get('user_input', '')
        
        # Generate cache key
        cache_key = f"context_{hash(str(conversation_history[-5:]))}"
        
        # Check cache first
        cached_context = get_context_cache(state, cache_key)
        if cached_context:
            self.logger.info("Using cached context analysis")
            update_resource_metrics(state, cache_hits=1)
            return {
                'summary': cached_context.conversation_summary,
                'entities': cached_context.extracted_entities,
                'preferences': cached_context.user_preferences,
                'confidence': cached_context.confidence,
                'cached': True
            }
        
        # Perform new analysis
        try:
            context = await self._perform_context_analysis(conversation_history, user_input, state)
            
            # Cache the results
            cache_item = ContextCache(
                conversation_summary=context.get('summary', ''),
                extracted_entities=context.get('entities', {}),
                user_preferences=context.get('preferences', {}),
                confidence=context.get('confidence', 0.5),
                timestamp=datetime.now().isoformat(),
                ttl=self.cache_ttl
            )
            set_context_cache(state, cache_key, cache_item)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context analysis failed: {e}")
            return self._get_fallback_context(conversation_history, user_input)
    
    async def _perform_context_analysis(self, conversation_history: List[Dict], 
                                      user_input: str, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Perform detailed context analysis using LLM"""
        if is_rate_limited(state, 'gemini'):
            return self._get_fallback_context(conversation_history, user_input)
        
        prompt = self._build_context_analysis_prompt(conversation_history, user_input)
        
        try:
            increment_rate_limit_counter(state, 'gemini')
            response = await self.generate_response(prompt)
            update_resource_metrics(state, tokens_used=len(prompt + response) // 4, api_calls=1)
            
            analysis = self._parse_json_response(response)
            
            return {
                'summary': analysis.get('conversation_summary', ''),
                'entities': analysis.get('extracted_entities', {}),
                'preferences': analysis.get('user_preferences', {}),
                'confidence': analysis.get('confidence', 0.5),
                'cached': False
            }
            
        except Exception as e:
            self.logger.error(f"LLM context analysis failed: {e}")
            return self._get_fallback_context(conversation_history, user_input)
    
    # === TASK ANALYSIS & ROUTING ===
    
    async def _analyze_task(self, state: EnhancedAssistantState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent task analysis and complexity assessment"""
        user_input = state.get('user_input', '')
        
        # Detect task type
        task_type = await self._detect_task_type(user_input, context, state)
        
        # Assess complexity
        complexity = await self._assess_task_complexity(task_type, context, state)
        
        # Analyze information completeness
        comp_analysis = await self._assess_information_completeness(state, context)
        completeness_num = comp_analysis.get('confidence', 0.5) * (1.0 if comp_analysis.get('complete') else 0.5)
        
        # Check agent capability
        can_handle_result = await can_handle_task(self, task_type, complexity, completeness_num)
        
        return {
            'task_type': task_type,
            'complexity': complexity,
            'completeness': completeness_num,
            'completeness_details': comp_analysis,
            'can_handle': can_handle_result,
            'context_confidence': context.get('confidence', 0.5),
            'routing_suggestion': await suggest_routing(self.agent_name, task_type, complexity, can_handle_result)
        }
    
    async def _detect_task_type(self, user_input: str, context: Dict[str, Any], 
                              state: EnhancedAssistantState) -> str:
        """Detect task type using pattern matching and LLM analysis"""
        # First try pattern matching for speed
        patterns = self._get_task_patterns()
        for task_type, keywords in patterns.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                return task_type
        
        # Fall back to LLM analysis if needed
        if not is_rate_limited(state, 'gemini'):
            return await self._llm_task_detection(user_input, context, state)
        
        return "general"  # Default
    
    # === AUTONOMOUS DECISION MAKING ===
    
    async def _make_autonomous_decision(self, task_analysis: Dict[str, Any], 
                                      context: Dict[str, Any], 
                                      state: EnhancedAssistantState) -> AgentDecision:
        """Advanced autonomous decision making with learning"""
        task_type = task_analysis['task_type']
        complexity = task_analysis['complexity']
        completeness_num = task_analysis['completeness']
        completeness_details = task_analysis.get('completeness_details', {})
        
        # Check decision patterns from past successes
        pattern_key = f"{task_type}_{complexity}_{int(completeness_num * 10)}"
        pattern_confidence = self.decision_patterns.get(pattern_key, 0.5)
        
        # Decision logic with multiple factors
        confidence_factors = {
            'completeness': completeness_num * 0.3,
            'context_quality': context.get('confidence', 0.5) * 0.2,
            'pattern_history': pattern_confidence * 0.3,
            'agent_capability': (1.0 if task_analysis['can_handle'] else 0.3) * 0.2
        }
        
        total_confidence = sum(confidence_factors.values())
        
        # Make decision based on confidence
        if total_confidence > 0.8:
            action = "execute_immediately"
            approach = "autonomous"
        elif total_confidence > 0.6:
            action = "execute_with_verification"
            approach = "verified"
        elif total_confidence > 0.4:
            action = "request_clarification"
            approach = "collaborative"
        else:
            action = "escalate"
            approach = "orchestrated"
        
        # Assess risks
        risks = self._assess_decision_risks(action, task_analysis, context)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(action, task_analysis)
        
        decision = AgentDecision(
            action=action,
            reasoning=f"Total confidence: {total_confidence:.2f}. Factors: {confidence_factors}",
            confidence=total_confidence,
            approach=approach,
            parameters={
                'task_type': task_type,
                'complexity': complexity,
                'completeness': completeness_num,
                'completeness_details': completeness_details
            },
            alternatives=alternatives,
            risks=risks
        )
        
        self.logger.info(f"Decision made: {action} (confidence: {total_confidence:.2f})")
        return decision
    
    # === EXECUTION WITH VERIFICATION ===
    
    async def _execute_with_verification(self, decision: AgentDecision, 
                                       state: EnhancedAssistantState, 
                                       context: Dict[str, Any]) -> TaskResult:
        """Execute task with comprehensive verification"""
        action = decision.action
        
        try:
            if action == "execute_immediately":
                result = await self._execute_task(decision, state, context)
                
            elif action == "execute_with_verification":
                result = await self._execute_task(decision, state, context)
                if result.success and self.verification_enabled:
                    verification = await self._verify_task_result(result, state, context)
                    if not verification.passed:
                        result.confidence *= 0.5  # Reduce confidence if verification fails
                        result.verification_passed = False
                
            elif action == "request_clarification":
                result = await self._request_clarification(decision, state, context)
                
            elif action == "escalate":
                result = await self._escalate_task(decision, state, context)
                
            else:
                result = TaskResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    task_type=decision.parameters.get('task_type', 'unknown'),
                    agent=self.agent_name,
                    error=f"Unknown action: {action}"
                )
            
            return result
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=decision.parameters.get('task_type', 'unknown'),
                agent=self.agent_name,
                error=str(e),
                needs_escalation=True
            )
    
    @abstractmethod
    async def _execute_task(self, decision: AgentDecision, 
                          state: EnhancedAssistantState, 
                          context: Dict[str, Any]) -> TaskResult:
        """Execute the actual task - implemented by subclasses"""
        pass
    
    # === VERIFICATION & HALLUCINATION MITIGATION ===
    
    async def _verify_result(self, result: TaskResult, 
                           state: EnhancedAssistantState, 
                           context: Dict[str, Any]) -> TaskResult:
        """Comprehensive result verification"""
        if not self.verification_enabled or not result.success:
            return result
        
        try:
            # Multiple verification layers
            verifications = []
            
            # 1. Content verification
            content_verification = await self._verify_content(result, context, state)
            verifications.append(content_verification)
            
            # 2. Factual verification
            factual_verification = await self._verify_facts(result, context, state)
            verifications.append(factual_verification)
            
            # 3. Hallucination detection
            hallucination_check = await self._detect_hallucinations(result, context, state)
            verifications.append(hallucination_check)
            
            # Aggregate verification results
            overall_confidence = sum(v.confidence for v in verifications) / len(verifications)
            all_passed = all(v.passed for v in verifications)
            all_issues = [issue for v in verifications for issue in v.issues]
            
            # Update result based on verification
            if not all_passed or overall_confidence < self.confidence_threshold:
                result.confidence *= 0.6  # Reduce confidence
                result.verification_passed = False
                if all_issues:
                    result.error = f"Verification failed: {'; '.join(all_issues)}"
            
            # Record verification results
            for verification in verifications:
                update_verification_result(state, verification)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return result  # Return original result if verification fails
    
    async def _verify_content(self, result: TaskResult, context: Dict[str, Any], 
                            state: EnhancedAssistantState) -> VerificationResult:
        """Verify content alignment with conversation context"""
        # Implementation varies by agent type
        return VerificationResult(
            passed=True,
            confidence=0.8,
            issues=[],
            factual_accuracy=0.8,
            context_alignment=0.8,
            completeness=0.8
        )
    
    async def _verify_facts(self, result: TaskResult, context: Dict[str, Any], 
                          state: EnhancedAssistantState) -> VerificationResult:
        """Verify factual accuracy of result"""
        # Basic fact checking implementation
        issues = []
        confidence = 0.8
        
        # Check for common fact-checking patterns
        result_text = str(result.data).lower()
        
        # Look for uncertain language
        uncertain_phrases = ['i think', 'maybe', 'probably', 'might be']
        if any(phrase in result_text for phrase in uncertain_phrases):
            issues.append("Contains uncertain language")
            confidence *= 0.8
        
        return VerificationResult(
            passed=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            factual_accuracy=confidence,
            context_alignment=0.8,
            completeness=0.8
        )
    
    async def _detect_hallucinations(self, result: TaskResult, context: Dict[str, Any], 
                                   state: EnhancedAssistantState) -> VerificationResult:
        """Detect potential hallucinations in result"""
        issues = []
        confidence = 0.9
        
        # Check for common hallucination patterns
        result_text = str(result.data)
        
        # Look for specific claims not supported by context
        entities = context.get('entities', {})
        if 'specific_claims' in entities:
            for claim in entities['specific_claims']:
                if claim in result_text and not self._verify_claim_against_context(claim, context):
                    issues.append(f"Unsupported claim: {claim}")
                    confidence *= 0.7
        
        return VerificationResult(
            passed=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            factual_accuracy=confidence,
            context_alignment=confidence,
            completeness=0.8
        )
    
    # === COLLABORATION & ESCALATION ===
    
    async def _handle_collaboration(self, result: TaskResult, 
                                  state: EnhancedAssistantState, 
                                  context: Dict[str, Any]) -> EnhancedAssistantState:
        """Handle inter-agent collaboration and escalation"""
        if result.needs_escalation:
            return await self._escalate_to_orchestrator(result, state, context)
        elif result.needs_help_from:
            return await self._request_agent_help(result, state, context)
        
        return state
    
    async def _escalate_to_orchestrator(self, result: TaskResult, 
                                     state: EnhancedAssistantState, 
                                     context: Dict[str, Any]) -> EnhancedAssistantState:
        """Escalate complex task to orchestrator"""
        escalation = EscalationRequest(
            escalated_from=self.agent_name,
            reason=result.error or "Task complexity requires orchestration",
            task_type=result.task_type,
            context=context,
            attempted_actions=[result.task_type],
            confidence=result.confidence,
            complexity="high"
        )
        
        add_escalation_request(state, escalation)
        state['route'] = 'orchestrator'
        
        message = MessageFactory.create_escalation_request(
            agent=self.agent_name,
            reason=escalation.reason,
            content=f"Escalating {result.task_type} to orchestrator",
            attempted_actions=escalation.attempted_actions,
            complexity=escalation.complexity
        )
        
        return self._add_agent_message(state, message.content, message.message_type.value)
    
    async def _request_agent_help(self, result: TaskResult, 
                                state: EnhancedAssistantState, 
                                context: Dict[str, Any]) -> EnhancedAssistantState:
        """Request help from another agent"""
        collaboration = CollaborationRequest(
            requesting_agent=self.agent_name,
            target_agent=result.needs_help_from,
            task_type=result.task_type,
            context=context,
            priority="medium",
            expected_result=f"Assistance with {result.task_type}"
        )
        
        add_collaboration_request(state, collaboration)
        state['route'] = 'orchestrator'  # Route through orchestrator for coordination
        
        message = MessageFactory.create_collaboration_request(
            requesting_agent=self.agent_name,
            target_agent=result.needs_help_from,
            content=f"Requesting help with {result.task_type}",
            expected_result=collaboration.expected_result,
            context=context
        )
        
        return self._add_agent_message(state, message.content, message.message_type.value)
    
    # === ERROR HANDLING & RECOVERY ===
    
    async def _handle_error_with_recovery(self, error: Exception, 
                                        state: EnhancedAssistantState, 
                                        start_time: float) -> EnhancedAssistantState:
        """Comprehensive error handling with recovery strategies"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Log error
        self.logger.error(f"Agent {self.agent_name} error: {error_type}: {error_msg}")
        
        # Record error in state
        error_record = {
            'agent': self.agent_name,
            'error_type': error_type,
            'error_message': error_msg,
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time
        }
        
        error_log = state.get('error_log', [])
        error_log.append(error_record)
        state['error_log'] = error_log
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error_type, error_msg, state)
        
        try:
            if recovery_strategy == 'retry':
                return await self._retry_with_backoff(state)
            elif recovery_strategy == 'fallback':
                return await self._execute_fallback_strategy(state, error_msg)
            elif recovery_strategy == 'escalate':
                return await self._escalate_error(state, error_record)
            else:
                return await self._graceful_degradation(state, error_msg)
                
        except Exception as recovery_error:
            # If recovery fails, return minimal response
            self.logger.error(f"Recovery failed: {recovery_error}")
            return self._add_agent_message(
                state, 
                f"I encountered an issue and my recovery strategies failed. Please try rephrasing your request.",
                MessageType.ERROR_DETECTED.value
            )
    
    def _determine_recovery_strategy(self, error_type: str, error_msg: str, 
                                   state: EnhancedAssistantState) -> str:
        """Determine the best recovery strategy for the error"""
        retry_count = state.get('retry_count', 0)
        
        # Rate limit errors
        if "429" in error_msg or "quota" in error_msg.lower():
            return 'fallback' if retry_count > 0 else 'retry'
        
        # Network/timeout errors
        if any(term in error_msg.lower() for term in ['timeout', 'connection', 'network']):
            return 'retry' if retry_count < self.retry_config['max_retries'] else 'fallback'
        
        # JSON parsing errors
        if "json" in error_msg.lower():
            return 'fallback'
        
        # Complex task errors
        if retry_count >= self.retry_config['max_retries']:
            return 'escalate'
        
        return 'graceful_degradation'
    
    # === LEARNING & PATTERN UPDATES ===
    
    async def _update_learning_patterns(self, decision: AgentDecision, result: TaskResult, 
                                      state: EnhancedAssistantState) -> None:
        """Update learning patterns based on results"""
        if result.success:
            # Update success patterns
            pattern_key = f"{result.task_type}_{decision.approach}_{int(result.confidence * 10)}"
            current_success = self.decision_patterns.get(pattern_key, 0.5)
            # Exponential moving average
            self.decision_patterns[pattern_key] = (current_success * 0.7) + (result.confidence * 0.3)
            
            # Update success metrics
            if result.task_type not in self.success_metrics:
                self.success_metrics[result.task_type] = []
            self.success_metrics[result.task_type].append(result.confidence)
            
            # Keep only recent metrics
            if len(self.success_metrics[result.task_type]) > 100:
                self.success_metrics[result.task_type] = self.success_metrics[result.task_type][-50:]
    
    # === UTILITY METHODS ===
    
    async def generate_response(self, prompt: str, context: str = None) -> str:
        """Generate response using MCP client with error handling"""
        try:
            if context:
                full_prompt = f"{context}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            response = await self.mcp_client.chat(full_prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            raise
    
    async def use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Use MCP tool with resource tracking"""
        try:
            # This would be implemented based on your MCP client interface
            result = await getattr(self.mcp_client, f"_{tool_name}")(**parameters)
            return result
            
        except Exception as e:
            self.logger.error(f"Tool {tool_name} failed: {e}")
            raise
    
    def _add_agent_message(self, state: EnhancedAssistantState, content: str, 
                          message_type: str) -> EnhancedAssistantState:
        """Add agent message to state"""
        message = MessageFactory.create_agent_response(
            agent=self.agent_name,
            content=content,
            confidence=1.0
        )
        
        agent_messages = state.get('agent_messages', [])
        agent_messages.append(asdict(message))
        state['agent_messages'] = agent_messages
        state['final_response'] = content
        
        return state
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with fallback"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback response
        return {
            'confidence': 0.3,
            'reasoning': 'JSON parsing failed',
            'fallback': True
        }
    
    # === ABSTRACT METHODS FOR INITIALIZATION ===
    
    @abstractmethod
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize agent-specific prompt templates"""
        pass
    
    def _initialize_hallucination_detectors(self) -> List[callable]:
        """Initialize hallucination detection methods"""
        return [
            self._detect_uncertain_language,
            self._detect_unsupported_claims,
            self._detect_inconsistencies
        ]
    
    def _initialize_fact_checkers(self) -> List[callable]:
        """Initialize fact checking methods"""
        return [
            self._check_against_context,
            self._check_common_knowledge,
            self._check_logical_consistency
        ]
    
    def _initialize_fallback_strategies(self) -> Dict[str, callable]:
        """Initialize fallback strategies"""
        return {
            'rate_limit': self._fallback_cached_response,
            'network_error': self._fallback_offline_response,
            'parsing_error': self._fallback_template_response,
            'general': self._fallback_graceful_degradation
        }
    
    # === LLM-BACKED HELPER METHODS (Safe Defaults) ===
    
    def _build_context_analysis_prompt(self, conversation_history: List[Dict], user_input: str) -> str:
        """Build LLM prompt for context analysis"""
        recent_history = conversation_history[-5:] if conversation_history else []
        return f"""Analyze this conversation context:

Recent conversation: {json.dumps(recent_history, indent=2) if recent_history else "None"}
Current user input: "{user_input}"

Provide JSON analysis:
{{
    "conversation_summary": "brief summary of conversation flow",
    "extracted_entities": {{
        "people": [],
        "topics": [],
        "specific_claims": []
    }},
    "user_preferences": {{
        "communication_style": "formal|casual|technical",
        "response_length": "brief|detailed|comprehensive"
    }},
    "confidence": 0.8
}}"""

    async def _assess_task_complexity(self, task_type: str, context: Dict[str, Any], 
                                    state: EnhancedAssistantState) -> str:
        """LLM-backed task complexity assessment - returns complexity string"""
        if is_rate_limited(state, 'gemini'):
            return self._fallback_complexity_assessment(task_type, context)
        
        prompt = f"""Assess task complexity:

Task Type: {task_type}
Context: {json.dumps(context, indent=2) if context else "{}"}
Agent Capabilities: {self.capabilities}

Classify complexity as "low", "medium", or "high" based on:
- Number of steps required
- Dependencies on external systems  
- Information gathering needs
- Processing complexity

Return JSON: {{"complexity": "low|medium|high", "reasoning": "explanation", "confidence": 0.8}}"""

        try:
            response = await self.generate_response(prompt)
            analysis = self._parse_json_response(response)
            complexity = analysis.get('complexity', 'medium')
            
            # Log the LLM decision
            get_decision_logger().log_complexity_assessment(
                agent=self.agent_name,
                task_type=task_type,
                context=str(context),
                complexity=complexity,
                confidence=analysis.get('confidence', 0.5),
                reasoning=analysis.get('reasoning', 'LLM complexity assessment'),
                fallback_used=False
            )
            
            return complexity
        except Exception as e:
            self.logger.error(f"LLM complexity assessment failed: {e}")
            complexity = self._fallback_complexity_assessment(task_type, context)
            
            # Log fallback decision
            get_decision_logger().log_complexity_assessment(
                agent=self.agent_name,
                task_type=task_type,
                context=str(context),
                complexity=complexity,
                confidence=0.3,
                reasoning=f"Fallback used due to error: {str(e)}",
                fallback_used=True
            )
            
            return complexity

    async def _llm_task_detection(self, user_input: str, context: Dict[str, Any], 
                                state: EnhancedAssistantState) -> str:
        """LLM-backed task type detection - returns task type string"""
        if is_rate_limited(state, 'gemini'):
            return "general"
        
        available_types = [t.value if hasattr(t, 'value') else str(t) for t in self.get_task_types()]
        prompt = f"""Detect the primary task type for this request:

User Input: "{user_input}"
Context: {json.dumps(context, indent=2) if context else "{}"}
Available Task Types: {available_types}

Return JSON: {{"task_type": "detected_type", "confidence": 0.8, "reasoning": "explanation"}}"""

        try:
            response = await self.generate_response(prompt)
            analysis = self._parse_json_response(response)
            task_type = analysis.get('task_type', 'general')
            
            # Log the LLM decision
            get_decision_logger().log_intent_detection(
                agent=self.agent_name,
                user_input=user_input,
                detected_intent=task_type,
                confidence=analysis.get('confidence', 0.5),
                reasoning=analysis.get('reasoning', 'LLM task detection'),
                fallback_used=False
            )
            
            return task_type
        except Exception as e:
            self.logger.error(f"LLM task detection failed: {e}")
            
            # Log fallback decision
            get_decision_logger().log_intent_detection(
                agent=self.agent_name,
                user_input=user_input,
                detected_intent="general",
                confidence=0.3,
                reasoning=f"Fallback used due to error: {str(e)}",
                fallback_used=True
            )
            
            return "general"

    async def _graceful_degradation(self, state: EnhancedAssistantState, error_msg: str) -> EnhancedAssistantState:
        """LLM-backed graceful degradation returning proper state format"""
        user_input = state.get('user_input', '')
        
        # Use LLM to generate contextual fallback response
        fallback_prompt = f"""Generate a helpful fallback response for this situation:

User Request: "{user_input}"
Error Context: "{error_msg}"
Agent Type: "{self.agent_name}"

Provide a response that:
1. Acknowledges the issue professionally
2. Suggests alternative approaches
3. Maintains user confidence
4. Offers specific next steps

Keep response concise and actionable."""

        try:
            fallback_response = await self.generate_response(fallback_prompt)
        except Exception:
            # Ultimate fallback if even the LLM fails
            fallback_response = f"I encountered an issue processing your request about {user_input[:50]}... Let me try a different approach or please provide more specific details."
            
        # Return state with required fields for orchestrator
        state['final_response'] = fallback_response
        state['confidence_score'] = 0.3  # Low confidence for degraded response
        state['route'] = self.agent_name
        state['fallback_triggered'] = True
        
        return state

    def _fallback_complexity_assessment(self, task_type: str, context: Dict[str, Any]) -> str:
        """Fallback complexity assessment when LLM is unavailable"""
        summary = context.get('summary', '') if context else ''
        
        # Simple heuristics
        if len(summary.split()) > 30:
            return "high"
        elif any(word in summary.lower() for word in ['analyze', 'compare', 'research', 'comprehensive']):
            return "high"
        elif any(word in summary.lower() for word in ['simple', 'quick', 'just', 'only']):
            return "low"
        else:
            return "medium"
    
    async def _assess_information_completeness(self, state: EnhancedAssistantState, 
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if we have sufficient information to complete the task"""
        user_input = state.get('user_input', '')
        
        if is_rate_limited(state, 'gemini'):
            return self._fallback_information_completeness(user_input, context)
        
        prompt = f"""Assess information completeness for this request:

User Request: "{user_input}"
Available Context: {json.dumps(context, indent=2) if context else "None"}
Agent Capabilities: {self.capabilities}

Determine if we have sufficient information to complete this request:
1. Are all required parameters present?
2. Is the request clear and unambiguous?
3. Do we need additional clarification?

Return JSON: {{"complete": true/false, "missing_info": ["list", "of", "missing"], "confidence": 0.8, "reasoning": "explanation"}}"""

        try:
            response = await self.generate_response(prompt)
            analysis = self._parse_json_response(response)
            
            # Log the assessment decision
            get_decision_logger().log_decision(
                decision_type="information_completeness",
                agent=self.agent_name,
                input_context=user_input,
                llm_reasoning=analysis.get('reasoning', 'Information completeness assessment'),
                decision_outcome=analysis.get('complete', False),
                confidence=analysis.get('confidence', 0.5)
            )
            
            return analysis
        except Exception as e:
            self.logger.error(f"Information completeness assessment failed: {e}")
            return self._fallback_information_completeness(user_input, context)
    
    def _fallback_information_completeness(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback information completeness assessment"""
        # Simple heuristics
        has_question_words = any(word in user_input.lower() for word in ['what', 'how', 'when', 'where', 'why', 'who'])
        is_short = len(user_input.split()) < 5
        
        if has_question_words and not is_short:
            complete = True
        elif any(word in user_input.lower() for word in ['send', 'create', 'schedule']) and is_short:
            complete = False  # Needs more details
        else:
            complete = True  # Assume complete for other cases
            
        return {
            "complete": complete,
            "missing_info": ["specific details"] if not complete else [],
            "confidence": 0.6,
            "reasoning": "Fallback heuristic assessment"
        }
    
    
    def _format_final_response(self, result: TaskResult, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Format final response for orchestrator consumption"""
        # Ensure required fields exist with proper types
        state['final_response'] = str(result.data) if result.data else "Task completed with no output"
        state['confidence_score'] = float(result.confidence) if result.confidence is not None else 0.5
        state['route'] = self.agent_name
        state['success'] = result.success
        
        # Add agent message for consistency
        message = MessageFactory.create_agent_response(
            agent=self.agent_name,
            content=state['final_response'],
            confidence=state['confidence_score']
        )
        
        agent_messages = state.get('agent_messages', [])
        agent_messages.append(asdict(message))
        state['agent_messages'] = agent_messages
        
        return state

    # === ABSTRACT PLACEHOLDER METHODS ===
    
    def _get_task_patterns(self) -> Dict[str, List[str]]:
        """Get task detection patterns - override in subclasses"""
        return {}
    
    def _get_fallback_context(self, conversation_history: List[Dict], user_input: str) -> Dict[str, Any]:
        """Get fallback context when analysis fails"""
        return {
            'summary': 'Context analysis unavailable',
            'entities': {},
            'preferences': {},
            'confidence': 0.3,
            'cached': False
        }
    
    async def _fallback_cached_response(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Fallback using cached responses"""
        return self._add_agent_message(state, "I'm managing system resources. How can I help you?", MessageType.RESOURCE_WARNING.value)
    
    async def _fallback_offline_response(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Fallback for network issues"""
        return self._add_agent_message(state, "I'm experiencing connectivity issues. Please try again in a moment.", MessageType.ERROR_DETECTED.value)
    
    async def _fallback_template_response(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Fallback using templates"""
        return self._add_agent_message(state, "I encountered a processing issue. Could you please rephrase your request?", MessageType.ERROR_DETECTED.value)
    
    async def _fallback_graceful_degradation(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """General graceful degradation"""
        return self._add_agent_message(state, "I'm having trouble processing your request. Please provide more specific details.", MessageType.ERROR_DETECTED.value)
    
    # Placeholder implementations for abstract detection methods
    def _detect_uncertain_language(self, text: str) -> List[str]: return []
    def _detect_unsupported_claims(self, text: str, context: Dict) -> List[str]: return []
    def _detect_inconsistencies(self, text: str, context: Dict) -> List[str]: return []
    def _check_against_context(self, text: str, context: Dict) -> bool: return True
    def _check_common_knowledge(self, text: str) -> bool: return True
    def _check_logical_consistency(self, text: str) -> bool: return True
    def _verify_claim_against_context(self, claim: str, context: Dict) -> bool: return True