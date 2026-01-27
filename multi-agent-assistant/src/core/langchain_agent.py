import os
import asyncio
from typing import Dict, Any, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime

from .langchain_tools import (
    get_all_tools,
    gmail_send_actual,
    gcal_create_actual,
    gcal_update_actual,
    gcal_delete_actual
)
from .state_schema import AssistantState

class StreamlitCallbackHandler(AsyncCallbackHandler):
    """
    Callback handler to push tool execution events into the AssistantState's
    agent_messages for real-time UI updates in Streamlit.
    """
    def __init__(self, state: AssistantState):
        self.state = state
        self.run_to_tool = {}
        if "agent_messages" not in self.state:
            self.state["agent_messages"] = []

    def _add_msg(self, sender: str, message_type: str, payload: Dict[str, Any]):
        self.state["agent_messages"].append({
            "sender": sender,
            "message_type": message_type,
            "payload": payload
        })

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        name = serialized.get("name", "")
        self.run_to_tool[run_id] = name

        sender = "default"
        if "gmail" in name or "email" in name: sender = "email"
        elif "calendar" in name: sender = "calendar"
        elif "search" in name: sender = "search"

        # Optionally add a 'thinking' or 'plan' message to the UI
        self._add_msg("planner", "trace", {
            "explain": f"Executing tool: {name}",
            "thinking": [f"Calling {name} with input: {input_str}"]
        })

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        name = self.run_to_tool.get(run_id, "unknown_tool")

        # Map tool name to UI sender
        sender = "default"
        if "gmail" in name or "email" in name: sender = "email"
        elif "calendar" in name: sender = "calendar"
        elif "search" in name: sender = "search"

        # Format payload for UI rendering
        payload = output
        if isinstance(output, list):
            payload = {"items": output}
        elif isinstance(output, dict):
            payload = output
        else:
            payload = {"result": str(output)}

        self._add_msg(sender, "response", payload)

class LangChainAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )

        self.tools = get_all_tools()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sophisticated personal AI assistant with access to Gmail, Google Calendar, and Web Search.
Your goal is to help the user manage their life efficiently and safely.

Guidelines:
1. **Planning**: Before taking actions, think about the steps needed.
2. **Safety**: For mutating actions like sending emails or creating/deleting calendar events, be precise.
3. **Context**: Use information from previous steps (e.g., search results) to inform subsequent actions (e.g., drafting an email).
4. **Conciseness**: Provide helpful but brief responses.

Current time: {current_time}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def ainvoke(self, state: AssistantState) -> AssistantState:
        # Handle confirmation if present
        if state.get("confirm") and state.get("confirm_context"):
            ctx = state["confirm_context"]
            tool_name = ctx.get("tool")
            tool_args = ctx.get("args", {})

            # Use the actual implementations for confirmed actions
            actual_impls = {
                "gmail_send": gmail_send_actual,
                "gcal_create_event": gcal_create_actual,
                "gcal_update_event": gcal_update_actual,
                "gcal_delete_event": gcal_delete_actual,
            }

            impl = actual_impls.get(tool_name)
            if impl:
                try:
                    result = await impl(**tool_args)

                    # Update history and clear confirm
                    state["history"].append({"role": "user", "content": state.get("user_input", "Confirmed.")})
                    msg = f"Action confirmed and executed: {tool_name}"
                    state["history"].append({"role": "assistant", "content": msg})

                    if "agent_messages" not in state:
                        state["agent_messages"] = []

                    # Push a final result message for the UI
                    state["agent_messages"].append({
                        "sender": "default",
                        "message_type": "response",
                        "payload": {"result": msg, "server_result": result}
                    })

                    state["confirm"] = False
                    state["confirm_context"] = None
                    return state
                except Exception as e:
                    error_msg = f"Error during confirmed execution: {str(e)}"
                    state["agent_messages"].append({
                        "sender": "default",
                        "message_type": "error",
                        "payload": {"result": error_msg}
                    })
                    return state

        # Convert history to LangChain format
        chat_history = []
        for h in (state.get("history") or []):
            if h["role"] == "user":
                chat_history.append(HumanMessage(content=h["content"]))
            else:
                chat_history.append(AIMessage(content=h["content"]))

        # Current time for the prompt
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare inputs
        inputs = {
            "input": state.get("user_input", ""),
            "chat_history": chat_history,
            "current_time": current_time
        }

        # Execute with callback for UI
        callback = StreamlitCallbackHandler(state)

        try:
            result = await self.executor.ainvoke(inputs, config={"callbacks": [callback]})
            output = result.get("output", "I'm sorry, I couldn't complete the request.")

            # Update history
            new_history = state.get("history", [])
            new_history.append({"role": "user", "content": state.get("user_input", "")})
            new_history.append({"role": "assistant", "content": output})
            state["history"] = new_history[-20:]

            # Final message for UI
            if "agent_messages" not in state:
                state["agent_messages"] = []
            state["agent_messages"].append({
                "sender": "default",
                "message_type": "response",
                "payload": {"result": output}
            })

        except Exception as e:
            error_msg = f"Error during execution: {str(e)}"
            state["agent_messages"].append({
                "sender": "default",
                "message_type": "error",
                "payload": {"result": error_msg}
            })

        return state
