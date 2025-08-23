from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

class MessageType(Enum):
    """Enhanced message types for all agent communications"""
    
    # === CORE COMMUNICATION ===
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_MESSAGE = "system_message"
    
    # === TASK EXECUTION ===
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    
    # === AUTONOMOUS DECISION MAKING ===
    DECISION_ANALYSIS = "decision_analysis"
    DECISION_MADE = "decision_made"
    DECISION_REVIEW = "decision_review"
    AUTONOMOUS_ACTION = "autonomous_action"
    
    # === INTER-AGENT COLLABORATION ===
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    AGENT_HANDOFF = "agent_handoff"
    COORDINATION_UPDATE = "coordination_update"
    
    # === ESCALATION & ROUTING ===
    ESCALATION_REQUEST = "escalation_request"
    ESCALATION_RESPONSE = "escalation_response"
    ROUTE_DECISION = "route_decision"
    ROUTE_CHANGE = "route_change"
    
    # === VERIFICATION & QUALITY CONTROL ===
    VERIFICATION_START = "verification_start"
    VERIFICATION_RESULT = "verification_result"
    HALLUCINATION_DETECTED = "hallucination_detected"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    
    # === ERROR HANDLING & RECOVERY ===
    ERROR_DETECTED = "error_detected"
    RECOVERY_ATTEMPT = "recovery_attempt"
    FALLBACK_TRIGGERED = "fallback_triggered"
    RETRY_INITIATED = "retry_initiated"
    
    # === RESOURCE MANAGEMENT ===
    RESOURCE_WARNING = "resource_warning"
    RATE_LIMIT_HIT = "rate_limit_hit"
    CACHE_HIT = "cache_hit"
    OPTIMIZATION_APPLIED = "optimization_applied"
    
    # === CONTEXT & MEMORY ===
    CONTEXT_EXTRACTED = "context_extracted"
    MEMORY_UPDATED = "memory_updated"
    CACHE_UPDATED = "cache_updated"
    PATTERN_LEARNED = "pattern_learned"
    
    # === SPECIALIZED AGENT MESSAGES ===
    # Email Agent
    EMAIL_DRAFT_CREATED = "email_draft_created"
    EMAIL_SENT = "email_sent"
    EMAIL_SEARCH_RESULT = "email_search_result"
    INBOX_ANALYZED = "inbox_analyzed"
    
    # Calendar Agent
    CALENDAR_EVENT_CREATED = "calendar_event_created"
    SCHEDULE_ANALYZED = "schedule_analyzed"
    AVAILABILITY_CHECKED = "availability_checked"
    CONFLICT_DETECTED = "conflict_detected"
    
    # Search Agent
    SEARCH_EXECUTED = "search_executed"
    RESEARCH_COMPILED = "research_compiled"
    FACT_VERIFIED = "fact_verified"
    SOURCE_VALIDATED = "source_validated"
    
    # === USER INTERACTION ===
    CLARIFICATION_REQUEST = "clarification_request"
    USER_CONFIRMATION = "user_confirmation"
    FEEDBACK_REQUEST = "feedback_request"
    INPUT_VALIDATION = "input_validation"

class MessagePriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MessageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentMessage:
    """Enhanced agent message with full metadata"""
    agent: str
    message_type: MessageType
    content: str
    timestamp: str
    priority: MessagePriority = MessagePriority.MEDIUM
    status: MessageStatus = MessageStatus.COMPLETED
    
    # Context & Tracking
    task_id: Optional[str] = None
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    
    # Quality & Verification
    confidence: float = 1.0
    verified: bool = True
    hallucination_risk: float = 0.0
    
    # Resource Usage
    tokens_used: int = 0
    processing_time: float = 0.0
    api_calls: int = 0
    
    # Collaboration
    target_agent: Optional[str] = None
    requires_response: bool = False
    deadline: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CollaborationMessage(AgentMessage):
    """Specialized message for inter-agent collaboration"""
    collaboration_type: str = "request"  # request, response, handoff, coordination
    expected_result: str = ""
    context_shared: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.context_shared is None:
            self.context_shared = {}

@dataclass
class EscalationMessage(AgentMessage):
    """Specialized message for task escalation"""
    escalation_reason: str = ""
    attempted_actions: List[str] = None
    complexity_level: str = "medium"
    suggested_approach: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if self.attempted_actions is None:
            self.attempted_actions = []

@dataclass
class VerificationMessage(AgentMessage):
    """Specialized message for verification and quality control"""
    verification_type: str = "content"  # content, factual, context, hallucination
    original_content: str = ""
    verification_result: Dict[str, Any] = None
    issues_found: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.verification_result is None:
            self.verification_result = {}
        if self.issues_found is None:
            self.issues_found = []

@dataclass
class ResourceMessage(AgentMessage):
    """Specialized message for resource management"""
    resource_type: str = "api"  # api, memory, cache, tokens
    usage_metrics: Dict[str, Any] = None
    warning_level: str = "info"  # info, warning, critical
    optimization_suggestion: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if self.usage_metrics is None:
            self.usage_metrics = {}

class MessageFactory:
    """Factory for creating different types of messages"""
    
    @staticmethod
    def create_agent_response(agent: str, content: str, confidence: float = 1.0, 
                            tokens_used: int = 0, task_id: str = None) -> AgentMessage:
        """Create a standard agent response message"""
        return AgentMessage(
            agent=agent,
            message_type=MessageType.AGENT_RESPONSE,
            content=content,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            tokens_used=tokens_used,
            task_id=task_id
        )
    
    @staticmethod
    def create_collaboration_request(requesting_agent: str, target_agent: str, 
                                   content: str, expected_result: str,
                                   context: Dict[str, Any] = None) -> CollaborationMessage:
        """Create a collaboration request message"""
        return CollaborationMessage(
            agent=requesting_agent,
            message_type=MessageType.COLLABORATION_REQUEST,
            content=content,
            timestamp=datetime.now().isoformat(),
            target_agent=target_agent,
            requires_response=True,
            collaboration_type="request",
            expected_result=expected_result,
            context_shared=context or {}
        )
    
    @staticmethod
    def create_escalation_request(agent: str, reason: str, content: str,
                                attempted_actions: List[str] = None,
                                complexity: str = "high") -> EscalationMessage:
        """Create an escalation request message"""
        return EscalationMessage(
            agent=agent,
            message_type=MessageType.ESCALATION_REQUEST,
            content=content,
            timestamp=datetime.now().isoformat(),
            target_agent="orchestrator",
            requires_response=True,
            priority=MessagePriority.HIGH,
            escalation_reason=reason,
            attempted_actions=attempted_actions or [],
            complexity_level=complexity
        )
    
    @staticmethod
    def create_verification_message(agent: str, original_content: str,
                                  verification_type: str = "content",
                                  result: Dict[str, Any] = None) -> VerificationMessage:
        """Create a verification message"""
        return VerificationMessage(
            agent=agent,
            message_type=MessageType.VERIFICATION_RESULT,
            content=f"Verification completed for {verification_type}",
            timestamp=datetime.now().isoformat(),
            verification_type=verification_type,
            original_content=original_content,
            verification_result=result or {}
        )
    
    @staticmethod
    def create_resource_warning(agent: str, resource_type: str, metrics: Dict[str, Any],
                              warning_level: str = "warning") -> ResourceMessage:
        """Create a resource management message"""
        return ResourceMessage(
            agent=agent,
            message_type=MessageType.RESOURCE_WARNING,
            content=f"Resource warning: {resource_type} usage high",
            timestamp=datetime.now().isoformat(),
            priority=MessagePriority.HIGH if warning_level == "critical" else MessagePriority.MEDIUM,
            resource_type=resource_type,
            usage_metrics=metrics,
            warning_level=warning_level
        )
    
    @staticmethod
    def create_decision_message(agent: str, decision: str, reasoning: str,
                              confidence: float, alternatives: List[str] = None) -> AgentMessage:
        """Create an autonomous decision message"""
        return AgentMessage(
            agent=agent,
            message_type=MessageType.DECISION_MADE,
            content=f"Decision: {decision}. Reasoning: {reasoning}",
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            metadata={
                "decision": decision,
                "reasoning": reasoning,
                "alternatives": alternatives or [],
                "decision_time": datetime.now().isoformat()
            }
        )
    
    @staticmethod
    def create_task_message(agent: str, task_type: str, status: str, 
                          content: str, confidence: float = 1.0) -> AgentMessage:
        """Create a task-related message"""
        message_type_map = {
            "start": MessageType.TASK_START,
            "progress": MessageType.TASK_PROGRESS,
            "complete": MessageType.TASK_COMPLETE,
            "failed": MessageType.TASK_FAILED
        }
        
        return AgentMessage(
            agent=agent,
            message_type=message_type_map.get(status, MessageType.TASK_PROGRESS),
            content=content,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            metadata={
                "task_type": task_type,
                "status": status
            }
        )

class MessageValidator:
    """Validate and sanitize messages"""
    
    @staticmethod
    def validate_message(message: AgentMessage) -> bool:
        """Validate message structure and content"""
        required_fields = ['agent', 'message_type', 'content', 'timestamp']
        
        for field in required_fields:
            if not hasattr(message, field) or not getattr(message, field):
                return False
        
        # Validate confidence range
        if not (0.0 <= message.confidence <= 1.0):
            return False
        
        # Validate hallucination risk range
        if not (0.0 <= message.hallucination_risk <= 1.0):
            return False
        
        return True
    
    @staticmethod
    def sanitize_content(content: str) -> str:
        """Sanitize message content"""
        # Remove potentially sensitive information
        import re
        
        # Remove potential API keys
        content = re.sub(r'[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[:=]\s*\S+', '[API_KEY_REDACTED]', content)
        
        # Remove potential passwords
        content = re.sub(r'[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]\s*[:=]\s*\S+', '[PASSWORD_REDACTED]', content)
        
        return content
    
    @staticmethod
    def assess_hallucination_risk(content: str, context: Dict[str, Any] = None) -> float:
        """Assess hallucination risk in content"""
        risk_score = 0.0
        
        # Check for uncertain language
        uncertain_phrases = ['i think', 'maybe', 'probably', 'might be', 'could be']
        for phrase in uncertain_phrases:
            if phrase in content.lower():
                risk_score += 0.1
        
        # Check for specific claims without context
        if context and 'factual_claims' in context:
            claims = context['factual_claims']
            for claim in claims:
                if claim in content and not context.get('verified_facts', {}).get(claim):
                    risk_score += 0.2
        
        return min(risk_score, 1.0)

class MessageRouter:
    """Route messages between agents"""
    
    def __init__(self):
        self.message_queue: List[AgentMessage] = []
        self.routing_rules: Dict[str, Any] = {}
    
    def add_message(self, message: AgentMessage) -> bool:
        """Add message to routing queue"""
        if MessageValidator.validate_message(message):
            # Sanitize content
            message.content = MessageValidator.sanitize_content(message.content)
            
            # Assess hallucination risk
            message.hallucination_risk = MessageValidator.assess_hallucination_risk(message.content)
            
            self.message_queue.append(message)
            return True
        return False
    
    def get_messages_for_agent(self, agent: str) -> List[AgentMessage]:
        """Get pending messages for specific agent"""
        agent_messages = [msg for msg in self.message_queue if msg.target_agent == agent]
        # Remove processed messages
        self.message_queue = [msg for msg in self.message_queue if msg.target_agent != agent]
        return agent_messages
    
    def get_high_priority_messages(self) -> List[AgentMessage]:
        """Get high priority messages that need immediate attention"""
        return [msg for msg in self.message_queue 
                if msg.priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]]
    
    def clear_old_messages(self, max_age_minutes: int = 30) -> None:
        """Clear old messages from queue"""
        cutoff_time = datetime.now().timestamp() - (max_age_minutes * 60)
        self.message_queue = [
            msg for msg in self.message_queue 
            if datetime.fromisoformat(msg.timestamp).timestamp() > cutoff_time
        ]