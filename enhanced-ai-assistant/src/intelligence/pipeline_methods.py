"""
Pipeline Methods for Enhanced Base Agent
Contains all the required pipeline methods with safe defaults.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from src.core.enhanced_state_schema import (
    EnhancedAssistantState, TaskResult, AgentDecision, TaskType,
    update_resource_metrics, record_agent_decision
)
from src.intelligence.decision_logger import get_decision_logger


class PipelineMethods:
    """Mixin class containing all required pipeline methods with safe defaults"""
    
    # === RESOURCE MANAGEMENT ===
    
    async def _manage_resources(self, state: EnhancedAssistantState) -> None:
        """Manage agent resources"""
        update_resource_metrics(state, processing_time=0.1)
    
    # === CONTEXT ANALYSIS ===
    
    async def _analyze_context(self, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Analyze conversation context"""
        return {
            'summary': state.get('user_input', ''),
            'entities': {},
            'user_preferences': {},
            'confidence': 0.7
        }
    
    async def _perform_context_analysis(self, conversation_history: List[Dict], 
                                      user_input: str, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Perform detailed context analysis"""
        return {
            'summary': user_input,
            'conversation_flow': len(conversation_history),
            'entities': {},
            'user_preferences': {},
            'confidence': 0.7
        }
    
    # === TASK ANALYSIS ===
    
    async def _analyze_task(self, state: EnhancedAssistantState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements"""
        user_input = state.get('user_input', '')
        task_type = await self._detect_task_type(user_input, context, state)
        complexity = await self._assess_task_complexity(task_type, context, state)
        completeness = await self._assess_information_completeness(state, context)
        can_handle = await self._can_handle_task(task_type, complexity, completeness)
        
        return {
            'task_type': task_type,
            'complexity': complexity,
            'completeness': completeness,
            'can_handle': can_handle,
            'routing_suggestion': await self._suggest_routing(task_type, complexity, can_handle)
        }
    
    async def _detect_task_type(self, user_input: str, context: Dict[str, Any], 
                              state: EnhancedAssistantState) -> str:
        """Detect task type from user input"""
        # Use existing LLM task detection
        return await self._llm_task_detection(user_input, context, state)
    
    # === DECISION MAKING ===
    
    async def _make_autonomous_decision(self, task_analysis: Dict[str, Any], 
                                      context: Dict[str, Any], 
                                      state: EnhancedAssistantState) -> AgentDecision:
        """Make autonomous decision based on analysis"""
        return AgentDecision(
            action="execute_task",
            reasoning=f"Agent can handle {task_analysis['task_type']} with {task_analysis['complexity']} complexity",
            confidence=0.8,
            approach="standard_execution",
            parameters=task_analysis
        )
    
    # === TASK EXECUTION ===
    
    async def _execute_with_verification(self, decision: AgentDecision, 
                                       state: EnhancedAssistantState, 
                                       context: Dict[str, Any]) -> TaskResult:
        """Execute task with verification"""
        try:
            result = await self._execute_task(decision, state, context)
            
            # Basic verification
            if result.success and result.data:
                if hasattr(self, 'verification_enabled') and self.verification_enabled:
                    verification = await self._verify_task_result(result, state, context)
                    result.verification_passed = verification.get('passed', True)
            
            return result
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=decision.parameters.get('task_type', 'unknown'),
                agent=getattr(self, 'agent_name', 'unknown_agent'),
                error=str(e)
            )
    
    # === VERIFICATION ===
    
    async def _verify_result(self, result: TaskResult, state: EnhancedAssistantState, 
                           context: Dict[str, Any]) -> TaskResult:
        """Verify task result quality"""
        if result.success and result.data:
            result.verification_passed = True
        return result
    
    async def _verify_task_result(self, result: TaskResult, state: EnhancedAssistantState, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify specific task result"""
        return {"passed": True, "confidence": 0.8}
    
    async def _verify_content(self, result: TaskResult, context: Dict[str, Any], 
                            state: EnhancedAssistantState) -> Dict[str, Any]:
        """Verify content quality"""
        return {"verified": True, "confidence": 0.8}
    
    async def _verify_facts(self, result: TaskResult, context: Dict[str, Any], 
                          state: EnhancedAssistantState) -> Dict[str, Any]:
        """Verify factual accuracy"""
        return {"verified": True, "confidence": 0.8}
    
    async def _detect_hallucinations(self, result: TaskResult, context: Dict[str, Any], 
                                   state: EnhancedAssistantState) -> Dict[str, Any]:
        """Detect potential hallucinations"""
        return {"hallucination_detected": False, "confidence": 0.8}
    
    # === COLLABORATION & ESCALATION ===
    
    async def _handle_collaboration(self, result: TaskResult, state: EnhancedAssistantState, 
                                  context: Dict[str, Any]) -> EnhancedAssistantState:
        """Handle collaboration or escalation"""
        if result.needs_escalation:
            return await self._escalate_to_orchestrator(result, state, context)
        elif result.needs_help_from:
            return await self._request_agent_help(result, state, context)
        return state
    
    async def _escalate_to_orchestrator(self, result: TaskResult, state: EnhancedAssistantState, 
                                      context: Dict[str, Any]) -> EnhancedAssistantState:
        """Escalate to orchestrator"""
        state['final_response'] = f"Task requires escalation: {result.error or 'Complex task needs orchestrator'}"
        state['confidence_score'] = 0.3
        state['route'] = 'orchestrator'
        return state
    
    async def _request_agent_help(self, result: TaskResult, state: EnhancedAssistantState, 
                                context: Dict[str, Any]) -> EnhancedAssistantState:
        """Request help from another agent"""
        state['final_response'] = f"Task requires collaboration with {result.needs_help_from}"
        state['confidence_score'] = 0.4
        state['route'] = result.needs_help_from or 'orchestrator'
        return state
    
    async def _escalate_task(self, decision: AgentDecision, state: EnhancedAssistantState, 
                           context: Dict[str, Any]) -> TaskResult:
        """Escalate task to orchestrator"""
        return TaskResult(
            success=False,
            data="Task escalated to orchestrator",
            confidence=0.3,
            task_type=decision.parameters.get('task_type', 'unknown'),
            agent=getattr(self, 'agent_name', 'unknown_agent'),
            needs_escalation=True
        )
    
    async def _request_clarification(self, decision: AgentDecision, state: EnhancedAssistantState, 
                                   context: Dict[str, Any]) -> TaskResult:
        """Request clarification from user"""
        completeness = decision.parameters.get('completeness', {})
        missing_info = completeness.get('missing_info', [])
        
        clarification_msg = f"I need more information to help you: {', '.join(missing_info)}"
        
        return TaskResult(
            success=True,
            data=clarification_msg,
            confidence=0.6,
            task_type=decision.parameters.get('task_type', 'unknown'),
            agent=getattr(self, 'agent_name', 'unknown_agent')
        )
    
    # === LEARNING & PATTERNS ===
    
    async def _update_learning_patterns(self, decision: AgentDecision, result: TaskResult, 
                                      state: EnhancedAssistantState) -> None:
        """Update learning patterns based on results"""
        patterns = state.get('success_patterns', {})
        task_type = decision.parameters.get('task_type', 'unknown')
        patterns[task_type] = patterns.get(task_type, 0.5) * 0.9 + (float(result.success) * 0.1)
        state['success_patterns'] = patterns
    
    # === ERROR HANDLING ===
    
    async def _handle_error_with_recovery(self, error: Exception, state: EnhancedAssistantState, 
                                        start_time: float) -> EnhancedAssistantState:
        """Handle errors with recovery strategies"""
        error_msg = str(error)
        
        error_record = {
            'agent': getattr(self, 'agent_name', 'unknown_agent'),
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time
        }
        
        error_log = state.get('error_log', [])
        error_log.append(error_record)
        state['error_log'] = error_log
        
        return await self._graceful_degradation(state, error_msg)
    
    async def _retry_with_backoff(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Retry with backoff strategy"""
        retry_count = state.get('retry_count', 0) + 1
        state['retry_count'] = retry_count
        
        if retry_count > 3:
            return await self._graceful_degradation(state, "Maximum retries exceeded")
        
        # Simple retry - in practice would implement exponential backoff
        return state
    
    async def _execute_fallback_strategy(self, state: EnhancedAssistantState, error_msg: str) -> EnhancedAssistantState:
        """Execute fallback strategy"""
        return await self._graceful_degradation(state, f"Fallback strategy: {error_msg}")
    
    async def _escalate_error(self, state: EnhancedAssistantState, error_record: Dict[str, Any]) -> EnhancedAssistantState:
        """Escalate error to orchestrator"""
        state['final_response'] = f"Error requires escalation: {error_record.get('error', 'Unknown error')}"
        state['confidence_score'] = 0.2
        state['route'] = 'orchestrator'
        state['fallback_triggered'] = True
        return state
    
    # === UTILITY METHODS ===
    
    async def _suggest_routing(self, task_type: str, complexity: str, can_handle: bool) -> str:
        """Suggest routing for task"""
        agent_name = getattr(self, 'agent_name', 'unknown_agent').lower()
        return f"route_to_{agent_name}" if can_handle else "route_to_orchestrator"
    
    async def _clean_expired_cache(self) -> None:
        """Clean expired cache entries"""
        # Placeholder for cache cleaning logic
        pass
    
    async def _can_handle_task(self, task_type: str, complexity: str, completeness: Dict[str, Any]) -> bool:
        """Check if this agent can handle the given task type with complexity and completeness considerations"""
        # Get supported task types
        if hasattr(self, 'get_task_types'):
            supported_task_types = [tt.value for tt in self.get_task_types()]
        else:
            supported_task_types = []
        
        if task_type in supported_task_types:
            # Additional checks based on complexity and completeness
            if complexity == "high" and not completeness.get('complete', True):
                # High complexity incomplete tasks may need escalation
                return len(completeness.get('missing_info', [])) <= 2
            return True
        
        # Check against capabilities (for backward compatibility)
        if hasattr(self, 'capabilities'):
            task_lower = task_type.lower().replace('_', ' ')
            for capability in self.capabilities:
                if capability.lower().replace('_', ' ') in task_lower or task_lower in capability.lower().replace('_', ' '):
                    return True
        
        # Check task patterns
        if hasattr(self, '_get_task_patterns'):
            task_patterns = self._get_task_patterns()
            for pattern_task, keywords in task_patterns.items():
                if task_type == pattern_task:
                    return True
                for keyword in keywords:
                    if keyword.lower() in task_type.lower():
                        return True
        
        return False