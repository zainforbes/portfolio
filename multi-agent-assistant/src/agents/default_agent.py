# src/agents/default_agent.py
from __future__ import annotations

import re
from typing import Dict, Any, List

from .base_agent import BaseAgent

# Simple greeting / small-talk detector
_GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|howdy|yo|sup|morning|afternoon|evening|how are you)\b",
    re.IGNORECASE,
)

_STYLE = """
You are a warm, succinct personal assistant.

Tone & style:
- Be friendly, specific, and concise.
- Never echo the user's prompt or describe your internal process.
- Prefer 1–3 short sentences per reply.

Special cases:
- If the message is a greeting/small-talk (e.g., “hey”, “how are you”):
  * Reply with one short sentence acknowledging the greeting.
  * Then add ONE compact CTA: “What can I help with—email, calendar, or search?”
  * Do not add anything else.

General chat:
- Answer directly and helpfully in 1–4 short sentences.
- If the user likely wants to take an action with email/calendar/search, end with ONE concrete, optional next step.
- Do not add generic closers like “anything else I can help with”.
"""

def _format_history(hist: List[Dict[str, str]], last_user: str, max_turns: int = 10) -> str:
    lines: List[str] = []
    for h in hist[-max_turns:]:
        role = (h.get("role") or "").strip().capitalize() or "User"
        content = h.get("content") or ""
        lines.append(f"{role}: {content}")
    lines.append(f"User: {last_user}")
    return "\n".join(lines)

class DefaultAgent(BaseAgent):
    name = "default"

    def __init__(self, gemini=None, mcp=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = (state.get("user_input") or "").strip()
        history: List[Dict[str, str]] = state.get("history", []) or []

        # If no LLM configured, provide a clean fallback
        if not self.gemini:
            reply = "Hi! What can I help with—email, calendar, or search?"
            self.add_msg(state, "response", {"result": reply})
            # Maintain short history even in fallback
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply})
            state["history"] = history
            return state

        is_greeting = bool(_GREETING_RE.match(text))

        conv = _format_history(history, text, max_turns=12)
        prompt = (
            _STYLE
            + "\n\nConversation:\n"
            + conv
            + "\n\nWrite the assistant's reply now following the rules above."
        )

        reply = (self.gemini.chat(prompt) or "").strip()

        # Update rolling history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        state["history"] = history

        # If this was small-talk, tell downstream synthesis to skip extra narration
        if is_greeting:
            state["suppress_synthesis"] = True

        self.add_msg(state, "response", {"result": reply})
        return state
