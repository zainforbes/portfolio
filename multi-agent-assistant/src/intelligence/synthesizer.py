# src/intelligence/synthesizer.py
from typing import Dict, Any, List
from src.utils.gemini_client import GeminiClient

_MICRO_SYS = """
You are the Conversational Brain. Briefly summarize what just happened,
mention the most useful findings (up to 3 bullets), and optionally ask ONE short follow-up question if it helps progress.
If no question is needed, omit it.
Return plain text for the summary, and put any question on a new line starting with 'Follow-up:'.
"""

_FINAL_SYS = """
You are the Conversational Brain. Write a concise, friendly reply that:
- says what you did,
- summarizes key results,
- proposes 1–2 concrete next actions.
No raw JSON; be natural and helpful.
"""

def _summarize_results(results: Any) -> str:
    if isinstance(results, list):
        lines = [f"{len(results)} items"]
        for it in results[:5]:
            title = it.get("subject") or it.get("summary") or it.get("title") or "(item)"
            lines.append(f"- {title}")
        return "\n".join(lines)
    if isinstance(results, dict):
        return "object with keys: " + ", ".join(list(results.keys())[:8])
    return str(results)[:200]

def micro_summarize(user_text: str, step: Dict[str,Any], step_result: Any, gemini: GeminiClient) -> Dict[str,str]:
    step_desc = f"{step.get('agent','?')}.{step.get('tool','?')}"
    result_sum = _summarize_results(step_result)
    prompt = (
        f"{_MICRO_SYS}\n\n"
        f"User asked: {user_text}\n"
        f"Just executed: {step_desc}\n"
        f"Result summary:\n{result_sum}\n\n"
        "Write the brief summary now."
    )
    text = gemini.chat(prompt)
    follow = ""
    if "Follow-up:" in text:
        parts = text.split("Follow-up:", 1)
        text, follow = parts[0].strip(), parts[1].strip()
    return {"summary": text, "followup": follow}

def final_summarize(
    user_text: str,
    history: List[Dict[str,str]],
    plan: Dict[str, Any],
    results: Dict[str, Any],
    gemini: GeminiClient
) -> str:
    ctx = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history[-20:])
    lines = []
    for k,v in results.items():
        lines.append(f"{k}: {_summarize_results(v)}")
    res_block = "\n".join(lines) or "(no results)"
    thinking = "\n".join(f"- {t}" for t in plan.get("thinking", []))
    explain = plan.get("explain","")

    prompt = (
        f"{_FINAL_SYS}\n\nContext:\n{ctx or '(none)'}\n\n"
        f"User: {user_text}\n\n"
        f"Planner explain:\n{explain}\n\n"
        f"Planner thinking:\n{thinking}\n\n"
        f"Results:\n{res_block}\n\n"
        "Write the final assistant reply."
    )
    return gemini.chat(prompt)
