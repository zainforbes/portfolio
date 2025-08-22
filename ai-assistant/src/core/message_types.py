# src/core/message_types.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.
    """
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'message_type': self.message_type,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(
            sender=data['sender'],
            recipient=data['recipient'],
            message_type=data['message_type'],
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=data.get('correlation_id')
        )


# Message types for agent communication
class MessageTypes:
    # Basic communication
    PING = "ping"
    PONG = "pong"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    
    # Request delegation
    REQUEST = "request"
    RESPONSE = "response"
    COMPLETE = "complete"
    FAILED = "failed"
    
    # Coordination
    ESCALATION = "escalation"
    COLLABORATION_REQUEST = "collaboration_request"
    CONTEXT_SHARE = "context_share"
    
    # Data exchange
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    UPDATE_NOTIFICATION = "update_notification"
    
    # Email-specific messages
    EMAIL_CLASSIFY = "email_classify"
    EMAIL_COMPOSE = "email_compose"
    EMAIL_SEND = "email_send"
    EMAIL_ORGANIZE = "email_organize"
    
    # Calendar-specific messages
    CALENDAR_SCHEDULE = "calendar_schedule"
    CALENDAR_RESCHEDULE = "calendar_reschedule"
    CALENDAR_CANCEL = "calendar_cancel"
    CALENDAR_CHECK_AVAILABILITY = "calendar_check_availability"
    
    # Search-specific messages
    SEARCH_QUERY = "search_query"
    SEARCH_RESULTS = "search_results"
    SEARCH_REFINE = "search_refine"