from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient
from src.intelligence.verifier import verify_response

ZA = ZoneInfo("Africa/Johannesburg")
UTC = ZoneInfo("UTC")

def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)

def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")

def _parse_window(text: str) -> Tuple[Optional[str], Optional[str], str]:
    t = (text or "").lower()
    now = datetime.now(ZA)
    label = "upcoming"

    morning = (now.replace(hour=8,  minute=0), now.replace(hour=12, minute=0))
    afternoon= (now.replace(hour=12, minute=0), now.replace(hour=17, minute=0))
    evening = (now.replace(hour=17, minute=0), now.replace(hour=23, minute=59, second=59))

    if "later today" in t or ("later" in t and "today" in t):
        label = "later today"
        return _to_utc_z(now), _to_utc_z(_end_of_day(now)), label

    if "this morning" in t:
        label = "this morning";   return _to_utc_z(morning[0]),   _to_utc_z(morning[1]),   label
    if "this afternoon" in t:
        label = "this afternoon"; return _to_utc_z(afternoon[0]), _to_utc_z(afternoon[1]), label
    if "this evening" in t or "tonight" in t:
        label = "this evening";   return _to_utc_z(evening[0]),   _to_utc_z(evening[1]),   label

    if "today" in t:
        label = "today"
        return _to_utc_z(_start_of_day(now)), _to_utc_z(_end_of_day(now)), label

    if "tomorrow" in t:
        label = "tomorrow"
        tomorrow = now + timedelta(days=1)
        return _to_utc_z(_start_of_day(tomorrow)), _to_utc_z(_end_of_day(tomorrow)), label

    if "this week" in t:
        label = "this week"
        # Monday→Sunday of the current week
        start = _start_of_day(now - timedelta(days=now.weekday()))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end), label

    if "next week" in t:
        label = "next week"
        days_until_mon = (7 - now.weekday()) % 7 or 7
        start = _start_of_day(now + timedelta(days=days_until_mon))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end), label

    if "next 7 days" in t or "next seven days" in t:
        label = "next 7 days"
        return _to_utc_z(now), _to_utc_z(now + timedelta(days=7)), label

    # default: from now forward
    return _to_utc_z(now), None, label

def _parse_iso_maybe_local(s: str) -> datetime:
    if not s:
        return datetime.now(ZA)
    if len(s) == 10 and s.count("-") == 2 and "T" not in s:  # date-only
        return datetime.fromisoformat(s).replace(tzinfo=ZA)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ZA)

def _is_date_only(s: str) -> bool:
    return bool(s) and len(s) == 10 and "T" not in s

def _find_conflicts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    for i in range(1, len(events)):
        a, b = events[i-1], events[i]
        a_end_s = a.get("end", ""); b_start_s = b.get("start", "")
        # Google all-day events: end.date is exclusive (next day's 00:00).
        if _is_date_only(a_end_s):
            # treat as end-of-day of the previous (inclusive) day
            prev_day = datetime.fromisoformat(a_end_s).replace(tzinfo=ZA) - timedelta(days=1)
            a_end = _end_of_day(prev_day)
        else:
            a_end = _parse_iso_maybe_local(a_end_s)

        if _is_date_only(b_start_s):
            b_start = _start_of_day(datetime.fromisoformat(b_start_s).replace(tzinfo=ZA))
        else:
            b_start = _parse_iso_maybe_local(b_start_s)

        if b_start < a_end:
            conflicts.append({
                "a": a.get("summary",""),
                "b": b.get("summary",""),
                "range": f'{a.get("start","")} → {b.get("end","")}',
            })
    return conflicts

class CalendarAgent(BaseAgent):
    name = "calendar"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input", "")
        tmin, tmax, label = _parse_window(text)

        events = await self.mcp.call_tool("gcal_list_events", max_results=20, time_min=tmin, time_max=tmax)
        events_sorted = sorted(events, key=lambda e: _parse_iso_maybe_local(e.get("start","")))
        conflicts = _find_conflicts(events_sorted)

        payload: Dict[str, Any] = {
            "window": {"label": label, "time_min": tmin, "time_max": tmax},
            "summary": f"{len(events_sorted)} events",
            "items": events_sorted,
            "conflicts": conflicts,
        }

        # Optional Gemini summary BEFORE adding the message
        if self.gemini and events_sorted:
            bullets = "\n".join(
                f"- {e.get('summary','(no title)')} | {e.get('start','')} → {e.get('end','')}"
                for e in events_sorted[:5]
            )
            prompt = ("Summarize these events for the specified window. "
                      "Highlight any scheduling conflicts if present (short bullets).\n") + bullets
            payload["summary_llm"] = self.gemini.chat(prompt)

        # Attach verification and emit a single response message
        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state
