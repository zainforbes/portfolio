from typing import TypedDict, List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict

class TaskType(Enum):
    EMAIL_COMPOSITION = "email_composition"
    EMAIL_SEARCH = "email_search"
    EMAIL_SUMMARIZATION = "email_summarization"
    EMAIL_CLASSIFICATION = "email_classification"
    INBOX_MANAGEMENT = "inbox_management"

    CALENDAR_SCHEDULING = "calendar_scheduling"
    CALENDAR_SEARCH = "calendar_search"
    CALENDAR_ANALYSIS = "calendar_analysis"
    AVAILABILITY_CHECKING = "availability_checking"
    CONFLICT_RESOLUTION = "conflict_resolution"
    EVENT_MANAGEMENT = "event_management"

    WEB_SEARCH = "web_search"
    RESEARCH_ANALYSIS = "research_analysis"
    FACT_VERIFICATION = "fact_verification"
    INFORMATION_SYNTHESIS = "information_synthesis"
    SOURCE_VALIDATION = "source_validation"

    MULTI_AGENT_COORDINATION = "multi_agent_coordination"

class AgentType(Enum):
    EMAIL = "email"
    CALENDAR = "calendar"
    SEARCH = "search"
    ORCHESTRATOR = "orchestrator"

class ConfidenceLevel(Enum):
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.0-0.5

class TaskComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class ContextCache:
    """Cached conversation context with TTL"""
    conversation_summary: str
    extracted_entities: Dict[str, Any]
    user_preferences: Dict[str, Any]
    confidence: float
    timestamp: str
    ttl: int = 180  # 3 minutes

@dataclass
class TaskResult:
    """Standardized task result across all agents"""
    success: bool
    data: Any
    confidence: float
    task_type: str
    agent: str
    error: Optional[str] = None
    needs_escalation: bool = False
    needs_help_from: Optional[str] = None
    verification_passed: bool = True
    resource_usage: Dict[str, Any] = None

@dataclass
class AgentDecision:
    """Agent autonomous decision structure"""
    action: str
    reasoning: str
    confidence: float
    approach: str
    parameters: Dict[str, Any]
    alternatives: List[str] = None
    risks: List[str] = None

@dataclass
class CollaborationRequest:
    """Inter-agent collaboration request"""
    requesting_agent: str
    target_agent: str
    task_type: str
    context: Dict[str, Any]
    priority: str
    expected_result: str
    deadline: Optional[str] = None

@dataclass
class EscalationRequest:
    """Task escalation to orchestrator"""
    escalated_from: str
    reason: str
    task_type: str
    context: Dict[str, Any]
    attempted_actions: List[str]
    confidence: float
    complexity: str

@dataclass
class VerificationResult:
    """Result verification and hallucination detection"""
    passed: bool
    confidence: float
    issues: List[str]
    factual_accuracy: float
    context_alignment: float
    completeness: float

@dataclass
class ResourceMetrics:
    """Resource usage and performance metrics"""
    token_count: int
    api_calls: int
    cache_hits: int
    processing_time: float
    memory_usage: float
    rate_limit_status: Dict[str, Any]

class EnhancedAssistantState(TypedDict, total=False):
    """Enhanced state schema supporting advanced AI capabilities"""
    
    # === CORE INPUT/OUTPUT ===
    user_input: str
    final_response: str
    
    # === INTELLIGENT ROUTING ===
    current_agent: str
    task_type: str
    task_complexity: str
    confidence_score: float
    route: str
    route_confidence: float
    route_reason: str
    routing_history: List[Dict[str, Any]]
    
    # === AGENT COMMUNICATION & COORDINATION ===
    agent_messages: List[Dict[str, Any]]
    pending_requests: List[Dict[str, Any]]
    completed_actions: List[Dict[str, Any]]
    collaboration_requests: List[CollaborationRequest]
    escalation_requests: List[EscalationRequest]
    inter_agent_context: Dict[str, Any]
    
    # === AUTONOMOUS DECISION MAKING ===
    decision_history: List[AgentDecision]
    success_patterns: Dict[str, float]
    learning_metrics: Dict[str, Any]
    autonomous_actions: List[Dict[str, Any]]
    decision_confidence: float
    
    # === CONTEXT & MEMORY MANAGEMENT ===
    conversation_history: List[Dict[str, Any]]
    context_cache: Dict[str, ContextCache]
    extracted_entities: Dict[str, Any]
    user_preferences: Dict[str, Any]
    active_context: Dict[str, Any]
    session_context: Dict[str, Any]
    
    # === VERIFICATION & HALLUCINATION MITIGATION ===
    verification_results: List[VerificationResult]
    confidence_threshold: float
    verification_enabled: bool
    hallucination_flags: List[str]
    fact_check_results: Dict[str, Any]
    
    # === ERROR HANDLING & RECOVERY ===
    retry_count: int
    error_log: List[Dict[str, Any]]
    fallback_triggered: bool
    recovery_actions: List[str]
    error_patterns: Dict[str, int]
    
    # === RESOURCE MANAGEMENT & OPTIMIZATION ===
    resource_metrics: ResourceMetrics
    token_budget: int
    token_usage: Dict[str, Any]
    api_rate_limits: Dict[str, Any]
    cache_performance: Dict[str, Any]
    optimization_flags: List[str]
    
    # === PERFORMANCE MONITORING ===
    response_time: float
    cache_hits: int
    mcp_tool_calls: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    quality_scores: Dict[str, float]
    
    # === PROMPT ENGINEERING ===
    prompt_templates: Dict[str, str]
    dynamic_prompts: Dict[str, Any]
    context_optimization: Dict[str, Any]
    prompt_performance: Dict[str, float]

def make_enhanced_initial_state(user_input: str, user: Optional[str] = None) -> EnhancedAssistantState:
    """Factory to create enhanced initial state with all advanced features"""
    return EnhancedAssistantState(
        # Core
        user_input=user_input,
        final_response="",
        
        # Routing
        current_agent="",
        task_type="",
        task_complexity="medium",
        confidence_score=0.0,
        route="",
        route_confidence=0.0,
        route_reason="",
        routing_history=[],
        
        # Communication
        agent_messages=[],
        pending_requests=[],
        completed_actions=[],
        collaboration_requests=[],
        escalation_requests=[],
        inter_agent_context={},
        
        # Decision Making
        decision_history=[],
        success_patterns={},
        learning_metrics={},
        autonomous_actions=[],
        decision_confidence=0.0,
        
        # Context
        conversation_history=[{"user": user or "me", "text": user_input}],
        context_cache={},
        extracted_entities={},
        user_preferences={},
        active_context={},
        session_context={"start_time": datetime.now().isoformat()},
        
        # Verification
        verification_results=[],
        confidence_threshold=0.7,
        verification_enabled=True,
        hallucination_flags=[],
        fact_check_results={},
        
        # Error Handling
        retry_count=0,
        error_log=[],
        fallback_triggered=False,
        recovery_actions=[],
        error_patterns={},
        
        # Resource Management
        resource_metrics=ResourceMetrics(
            token_count=0,
            api_calls=0,
            cache_hits=0,
            processing_time=0.0,
            memory_usage=0.0,
            rate_limit_status={}
        ),
        token_budget=8000,
        token_usage={},
        api_rate_limits={},
        cache_performance={},
        optimization_flags=[],
        
        # Performance
        response_time=0.0,
        cache_hits=0,
        mcp_tool_calls=[],
        performance_metrics={},
        quality_scores={},
        
        # Prompt Engineering
        prompt_templates={},
        dynamic_prompts={},
        context_optimization={},
        prompt_performance={}
    )

# Utility functions for state management
def update_resource_metrics(state: EnhancedAssistantState, 
                           tokens_used: int = 0, 
                           api_calls: int = 0, 
                           cache_hits: int = 0,
                           processing_time: float = 0.0) -> None:
    """Update resource metrics in state"""
    metrics = state.get('resource_metrics', ResourceMetrics(0, 0, 0, 0.0, 0.0, {}))
    metrics.token_count += tokens_used
    metrics.api_calls += api_calls
    metrics.cache_hits += cache_hits
    metrics.processing_time += processing_time
    state['resource_metrics'] = metrics

def add_collaboration_request(state: EnhancedAssistantState, 
                            request: CollaborationRequest) -> None:
    """Add collaboration request to state"""
    requests = state.get('collaboration_requests', [])
    if not isinstance(requests, list):
        requests = []
    requests.append(request)
    state['collaboration_requests'] = requests

def add_escalation_request(state: EnhancedAssistantState, 
                         request: EscalationRequest) -> None:
    """Add escalation request to state"""
    requests = state.get('escalation_requests', [])
    if not isinstance(requests, list):
        requests = []
    requests.append(request)
    state['escalation_requests'] = requests

def record_agent_decision(state: EnhancedAssistantState, 
                        decision: AgentDecision) -> None:
    """Record agent decision for learning"""
    history = state.get('decision_history', [])
    if not isinstance(history, list):
        history = []
    history.append(decision)
    state['decision_history'] = history

def update_verification_result(state: EnhancedAssistantState, 
                             result: VerificationResult) -> None:
    """Update verification results"""
    results = state.get('verification_results', [])
    if not isinstance(results, list):
        results = []
    results.append(result)
    state['verification_results'] = results

def get_context_cache(state: EnhancedAssistantState, key: str) -> Optional[ContextCache]:
    """Get cached context by key"""
    cache = state.get('context_cache', {})
    cached_item = cache.get(key)
    
    if cached_item and isinstance(cached_item, dict):
        # Check if TTL has expired
        cache_time = datetime.fromisoformat(cached_item['timestamp'])
        if (datetime.now() - cache_time).seconds > cached_item.get('ttl', 180):
            # Remove expired cache
            del cache[key]
            state['context_cache'] = cache
            return None
        return ContextCache(**cached_item)
    
    return None

def set_context_cache(state: EnhancedAssistantState, key: str, context: ContextCache) -> None:
    """Set context cache with key"""
    cache = state.get('context_cache', {})
    cache[key] = asdict(context)
    state['context_cache'] = cache

def is_rate_limited(state: EnhancedAssistantState, service: str) -> bool:
    """Check if service is rate limited"""
    metrics = state.get('resource_metrics', ResourceMetrics(0, 0, 0, 0.0, 0.0, {}))
    rate_limits = metrics.rate_limit_status.get(service, {})
    
    if 'last_reset' in rate_limits and 'request_count' in rate_limits:
        last_reset = datetime.fromisoformat(rate_limits['last_reset'])
        if (datetime.now() - last_reset).seconds > 60:
            # Reset counters
            rate_limits['request_count'] = 0
            rate_limits['last_reset'] = datetime.now().isoformat()
            return False
        
        return rate_limits['request_count'] >= rate_limits.get('limit', 15)
    
    return False

def increment_rate_limit_counter(state: EnhancedAssistantState, service: str, limit: int = 15) -> None:
    """Increment rate limit counter for service"""
    metrics = state.get('resource_metrics', ResourceMetrics(0, 0, 0, 0.0, 0.0, {}))
    if service not in metrics.rate_limit_status:
        metrics.rate_limit_status[service] = {
            'request_count': 0,
            'limit': limit,
            'last_reset': datetime.now().isoformat()
        }
    
    metrics.rate_limit_status[service]['request_count'] += 1
    state['resource_metrics'] = metrics

def safe_get_resource_metric(state: EnhancedAssistantState, metric_name: str, default_value: Any = 0) -> Any:
    """Safely get resource metric from state, handling both dict and dataclass formats"""
    resource_metrics = state.get('resource_metrics')
    
    if resource_metrics is None:
        return default_value
    
    # If it's a dataclass instance
    if hasattr(resource_metrics, metric_name):
        return getattr(resource_metrics, metric_name)
    
    # If it's a dictionary (fallback)
    if isinstance(resource_metrics, dict):
        return resource_metrics.get(metric_name, default_value)
    
    return default_value