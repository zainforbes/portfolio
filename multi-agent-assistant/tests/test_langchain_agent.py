import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.langchain_agent import LangChainAgent

@pytest.fixture
def mock_gemini_key():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake_key"}):
        yield

@pytest.mark.asyncio
async def test_langchain_agent_init(mock_gemini_key):
    # Mock ChatGoogleGenerativeAI
    with patch("src.core.langchain_agent.ChatGoogleGenerativeAI"), \
         patch("src.core.langchain_agent.create_tool_calling_agent"), \
         patch("src.core.langchain_agent.AgentExecutor"):
        agent = LangChainAgent()
        assert agent is not None

@pytest.mark.asyncio
async def test_langchain_agent_ainvoke_basic(mock_gemini_key):
    with patch("src.core.langchain_agent.ChatGoogleGenerativeAI"), \
         patch("src.core.langchain_agent.create_tool_calling_agent"), \
         patch("src.core.langchain_agent.AgentExecutor") as MockExecutor:

        mock_executor_instance = AsyncMock()
        MockExecutor.return_value = mock_executor_instance
        mock_executor_instance.ainvoke.return_value = {"output": "Hello there!"}

        agent = LangChainAgent()

        state = {
            "user_input": "Hi",
            "history": [],
            "agent_messages": []
        }

        updated_state = await agent.ainvoke(state)

        assert updated_state["history"][-1]["content"] == "Hello there!"
        assert any(msg["payload"].get("result") == "Hello there!" for msg in updated_state["agent_messages"])

@pytest.mark.asyncio
async def test_langchain_agent_confirmation(mock_gemini_key):
    with patch("src.core.langchain_agent.ChatGoogleGenerativeAI"), \
         patch("src.core.langchain_agent.create_tool_calling_agent"), \
         patch("src.core.langchain_agent.AgentExecutor"):

        agent = LangChainAgent()
        # Mock a tool
        mock_tool = AsyncMock()
        mock_tool.name = "gmail_send"
        agent.tools = [mock_tool]

        state = {
            "confirm": True,
            "confirm_context": {
                "tool": "gmail_send",
                "args": {"to": "test@example.com", "subject": "test", "body": "test"}
            },
            "history": [],
            "agent_messages": []
        }

        updated_state = await agent.ainvoke(state)

        mock_tool.ainvoke.assert_called_once()
        assert updated_state["confirm"] is False
        assert updated_state["confirm_context"] is None

@pytest.mark.asyncio
async def test_langchain_agent_error_handling(mock_gemini_key):
    with patch("src.core.langchain_agent.ChatGoogleGenerativeAI"), \
         patch("src.core.langchain_agent.create_tool_calling_agent"), \
         patch("src.core.langchain_agent.AgentExecutor") as MockExecutor:

        mock_executor_instance = AsyncMock()
        MockExecutor.return_value = mock_executor_instance
        mock_executor_instance.ainvoke.side_effect = Exception("Agent failed")

        agent = LangChainAgent()

        state = {
            "user_input": "Fail me",
            "history": [],
            "agent_messages": []
        }

        updated_state = await agent.ainvoke(state)

        assert any(msg["message_type"] == "error" for msg in updated_state["agent_messages"])
        assert any("Agent failed" in msg["payload"]["result"] for msg in updated_state["agent_messages"] if msg["message_type"] == "error")

@pytest.mark.asyncio
async def test_langchain_agent_empty_history(mock_gemini_key):
    with patch("src.core.langchain_agent.ChatGoogleGenerativeAI"), \
         patch("src.core.langchain_agent.create_tool_calling_agent"), \
         patch("src.core.langchain_agent.AgentExecutor") as MockExecutor:

        mock_executor_instance = AsyncMock()
        MockExecutor.return_value = mock_executor_instance
        mock_executor_instance.ainvoke.return_value = {"output": "Response"}

        agent = LangChainAgent()

        state = {
            "user_input": "Hello",
            # No 'history' key at all
            "agent_messages": []
        }

        updated_state = await agent.ainvoke(state)
        assert len(updated_state["history"]) == 2
