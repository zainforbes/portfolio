from typing import Dict, Any, List
from .base_agent import BaseAgent

_SYS = (
    "You are a helpful personal AI assistant. Be concise and clear. "
    "Your job is to act as a AI assistant that can understand if the user needs things done relating" \
    "to their emails, calendar or even a web search."
)

class DefaultAgent(BaseAgent):
    name = "default"

    def __init__(self, gemini=None, mcp=None):
        super().__init__(gemini=gemini, mcp=mcp)

    def _format_history(self, hist: List[Dict[str, str]], last_user: str) -> str:
        # Simple in-context memory (last 10 turns)
        parts = [_SYS, ""]
        for h in hist[-10:]:
            parts.append(f"{h['role'].capitalize()}: {h['content']}")
        parts.append(f"User: {last_user}")
        return "\n".join(parts)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input", "")
        history = state.get("history", []) or []

        if not self.gemini:
            # Signal to UI to handle LLM fallback, but don't append any extra text
            self.add_msg(state, "response", {"result": "(No LLM configured)."})
            return state

        prompt = self._format_history(history, text)
        reply = (self.gemini.chat(prompt) or "").strip()

        # Update rolling state history used by the app
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        state["history"] = history

        self.add_msg(state, "response", {"result": reply})
        return state
