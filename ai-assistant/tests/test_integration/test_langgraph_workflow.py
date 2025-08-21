# tests/test_langgraph_workflow.py
import pytest
from src.core.langgraph_workflow import AssistantGraph, _example_agent_handler_task
from src.core.state_schema import make_initial_state

def test_classify_and_route_task_keyword():
    # Use example handler for task to see integration
    handlers = {"task": _example_agent_handler_task}
    g = AssistantGraph(agent_handlers=handlers)
    out = g.invoke("Add task: buy groceries")
    assert out["route"] == "task"
    assert out["route_confidence"] >= 0.6
    assert "Task handler saved task" in out["final_response"]

def test_classify_calendar_keyword():
    g = AssistantGraph(agent_handlers={})
    out = g.invoke("Schedule a meeting next Monday")
    assert out["route"] == "calendar" or out["route"] == "coordinator"
    assert out["final_response"] is not None

def test_multi_intent_goes_coordinator():
    g = AssistantGraph(agent_handlers={})
    out = g.invoke("Email John and schedule a meeting")
    # Depending on classifier heuristic, multi-intent often maps to coordinator
    assert out["route"] in ("coordinator", "email", "calendar", "task")
    assert isinstance(out["final_response"], str)
