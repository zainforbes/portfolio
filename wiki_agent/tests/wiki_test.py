import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

import pytest
import os
from wiki_agent import create_agent

def test_env_variable_loaded():
    """Ensure GEMINI_API_KEY is loaded properly from .env"""
    assert os.getenv("GEMINI_API_KEY") is not None, "GEMINI_API_KEY is missing"

def test_create_agent_instance():
    """Verify that the agent initializes correctly"""
    agent = create_agent()
    assert hasattr(agent, "model"), "Agent missing model attribute"
    assert hasattr(agent, "tools"), "Agent missing tools attribute"
    assert any("WikipediaTools" in str(t.__class__) for t in agent.tools), "WikipediaTools not found in agent"
