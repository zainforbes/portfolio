"""
Decision Logger for tracking LLM decisions and agent choices.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Decision:
    """Represents a single LLM decision"""
    timestamp: str
    decision_type: str  # 'intent_detection', 'complexity_assessment', 'tool_choice', etc.
    agent: str
    input_context: str
    llm_reasoning: str
    decision_outcome: Any
    confidence: float
    fallback_used: bool = False
    processing_time_ms: Optional[float] = None


class DecisionLogger:
    """Lightweight decision logging for tracking LLM choices"""
    
    def __init__(self, log_to_file: bool = True, max_entries: int = 1000):
        self.logger = logging.getLogger("decision_logger")
        self.decisions: List[Decision] = []
        self.log_to_file = log_to_file
        self.max_entries = max_entries
        
    def log_decision(self, 
                    decision_type: str,
                    agent: str,
                    input_context: str,
                    llm_reasoning: str,
                    decision_outcome: Any,
                    confidence: float,
                    fallback_used: bool = False,
                    processing_time_ms: Optional[float] = None) -> None:
        """Log a single LLM decision"""
        
        decision = Decision(
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type,
            agent=agent,
            input_context=input_context[:200] + "..." if len(input_context) > 200 else input_context,
            llm_reasoning=llm_reasoning,
            decision_outcome=decision_outcome,
            confidence=confidence,
            fallback_used=fallback_used,
            processing_time_ms=processing_time_ms
        )
        
        self.decisions.append(decision)
        
        # Keep only recent decisions
        if len(self.decisions) > self.max_entries:
            self.decisions = self.decisions[-self.max_entries//2:]
            
        # Log to file if enabled
        if self.log_to_file:
            self.logger.info(f"DECISION: {decision.decision_type} | {decision.agent} | "
                           f"confidence={decision.confidence:.2f} | "
                           f"fallback={decision.fallback_used} | "
                           f"outcome={str(decision.decision_outcome)[:50]}")
    
    def log_intent_detection(self, agent: str, user_input: str, detected_intent: str, 
                            confidence: float, reasoning: str, fallback_used: bool = False) -> None:
        """Log intent detection decision"""
        self.log_decision(
            decision_type="intent_detection",
            agent=agent,
            input_context=f"User input: {user_input}",
            llm_reasoning=reasoning,
            decision_outcome=detected_intent,
            confidence=confidence,
            fallback_used=fallback_used
        )
    
    def log_complexity_assessment(self, agent: str, task_type: str, context: str,
                                 complexity: str, confidence: float, reasoning: str, 
                                 fallback_used: bool = False) -> None:
        """Log task complexity assessment"""
        self.log_decision(
            decision_type="complexity_assessment",
            agent=agent,
            input_context=f"Task: {task_type}, Context: {context}",
            llm_reasoning=reasoning,
            decision_outcome=complexity,
            confidence=confidence,
            fallback_used=fallback_used
        )
    
    def log_tool_choice(self, agent: str, available_tools: List[str], chosen_tool: str,
                       context: str, confidence: float, reasoning: str) -> None:
        """Log tool selection decision"""
        self.log_decision(
            decision_type="tool_choice",
            agent=agent,
            input_context=f"Available: {available_tools}, Context: {context}",
            llm_reasoning=reasoning,
            decision_outcome=chosen_tool,
            confidence=confidence
        )
    
    def log_response_generation(self, agent: str, prompt: str, response: str,
                               confidence: float, reasoning: str = "Response generated") -> None:
        """Log response generation"""
        self.log_decision(
            decision_type="response_generation",
            agent=agent,
            input_context=prompt,
            llm_reasoning=reasoning,
            decision_outcome=response[:100] + "..." if len(response) > 100 else response,
            confidence=confidence
        )
    
    def get_recent_decisions(self, count: int = 10, decision_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent decisions, optionally filtered by type"""
        decisions = self.decisions
        
        if decision_type:
            decisions = [d for d in decisions if d.decision_type == decision_type]
            
        recent = decisions[-count:] if len(decisions) > count else decisions
        return [asdict(decision) for decision in reversed(recent)]
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get statistics about recent decisions"""
        if not self.decisions:
            return {"total_decisions": 0}
            
        stats = {
            "total_decisions": len(self.decisions),
            "by_type": {},
            "by_agent": {},
            "average_confidence": 0.0,
            "fallback_rate": 0.0
        }
        
        confidences = []
        fallbacks = 0
        
        for decision in self.decisions:
            # By type
            if decision.decision_type not in stats["by_type"]:
                stats["by_type"][decision.decision_type] = 0
            stats["by_type"][decision.decision_type] += 1
            
            # By agent
            if decision.agent not in stats["by_agent"]:
                stats["by_agent"][decision.agent] = 0
            stats["by_agent"][decision.agent] += 1
            
            # Confidence and fallbacks
            confidences.append(decision.confidence)
            if decision.fallback_used:
                fallbacks += 1
        
        if confidences:
            stats["average_confidence"] = sum(confidences) / len(confidences)
            stats["fallback_rate"] = fallbacks / len(self.decisions)
        
        return stats


# Global instance
_global_decision_logger = None


def get_decision_logger() -> DecisionLogger:
    """Get the global decision logger instance"""
    global _global_decision_logger
    if _global_decision_logger is None:
        _global_decision_logger = DecisionLogger()
    return _global_decision_logger


def log_llm_decision(decision_type: str, agent: str, input_context: str, 
                    llm_reasoning: str, decision_outcome: Any, confidence: float,
                    fallback_used: bool = False) -> None:
    """Convenience function to log decisions"""
    get_decision_logger().log_decision(
        decision_type=decision_type,
        agent=agent,
        input_context=input_context,
        llm_reasoning=llm_reasoning,
        decision_outcome=decision_outcome,
        confidence=confidence,
        fallback_used=fallback_used
    )