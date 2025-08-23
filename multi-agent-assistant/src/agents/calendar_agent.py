from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import re

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
        start = _start_of_day(now - timedelta(days=now.weekday()))  # Monday
        end   = _end_of_day(start + timedelta(days=6))               # Sunday
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

def _cal_intent(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("create event", "add event", "schedule", "book")): return "create"
    if any(k in t for k in ("update event", "move", "reschedule", "change")): return "update"
    if any(k in t for k in ("delete event", "remove", "cancel")): return "delete"
    return "read"

def _pick_day(text: str) -> datetime:
    now = datetime.now(ZA)
    if "tomorrow" in (text or "").lower():
        return now + timedelta(days=1)
    return now

class CalendarAgent(BaseAgent):
    name = "calendar"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def _handle_read(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input", "")
        tmin, tmax, label = _parse_window(text)

        events = await self.mcp.call_tool("gcal_list_events", max_results=20, time_min=tmin, time_max=tmax)
        events_sorted = sorted(events, key=lambda e: _parse_iso_maybe_local(e.get("start","")))
        conflicts = _find_conflicts(events_sorted)

        payload: Dict[str, Any] = {
            "intent": "read",
            "window": {"label": label, "time_min": tmin, "time_max": tmax},
            "summary": f"{len(events_sorted)} events",
            "items": events_sorted,
            "conflicts": conflicts,
        }

        if self.gemini and events_sorted:
            bullets = "\n".join(
                f"- {e.get('summary','(no title)')} | {e.get('start','')} → {e.get('end','')}"
                for e in events_sorted[:5]
            )
            prompt = ("Summarize these events for the specified window. "
                      "Highlight any scheduling conflicts if present (short bullets).\n") + bullets
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _handle_create(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        # title: grab quoted string if available
        m_title = re.search(r"'([^']+)'|\"([^\"]+)\"", text)
        title = (m_title.group(1) or m_title.group(2)) if m_title else "New Event"

        # time range HH:MM-HH:MM (local)
        m_time = re.search(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})", text)
        day = _pick_day(text).strftime("%Y-%m-%d")
        start_local = f"{day}T10:00:00+02:00"
        end_local   = f"{day}T11:00:00+02:00"
        if m_time:
            start_local = f"{day}T{m_time.group(1)}:00+02:00" if len(m_time.group(1))==5 else f"{day}T{m_time.group(1)}+02:00"
            end_local   = f"{day}T{m_time.group(2)}:00+02:00" if len(m_time.group(2))==5 else f"{day}T{m_time.group(2)}+02:00"

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "intent":"create",
                "requires_confirmation": True,
                "message": "Create event detected. Re-run with confirm=True to proceed.",
                "proposal": {"summary": title, "start": start_local, "end": end_local}
            })
            return state

        created = await self.mcp.call_tool("gcal_create_event", summary=title, start_iso=start_local, end_iso=end_local, tz="Africa/Johannesburg")
        payload: Dict[str, Any] = {"action":"created","event":created}
        if self.gemini:
            payload["summary_llm"] = self.gemini.chat(f"Created event '{title}' from {start_local} to {end_local}. Summarize briefly.")
        payload.update(verify_response("calendar", {"items":[created]}))
        self.add_msg(state, "response", payload)
        return state

    async def _handle_delete(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        tmin, tmax, _ = _parse_window(text)
        events = await self.mcp.call_tool("gcal_list_events", max_results=5, time_min=tmin, time_max=tmax)
        if not events:
            self.add_msg(state, "response", {"intent":"delete","message":"No events to delete in window."})
            return state
        target = events[0]

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "intent":"delete",
                "requires_confirmation": True,
                "message": f"Delete '{target.get('summary','(no title)')}'? Re-run with confirm=True to proceed.",
                "target": target
            })
            return state

        await self.mcp.call_tool("gcal_delete_event", event_id=target["id"])
        payload: Dict[str, Any] = {"action":"deleted","target_id": target["id"]}
        if self.gemini:
            payload["summary_llm"] = self.gemini.chat(f"Deleted event '{target.get('summary','')}'. Summarize briefly.")
        payload.update(verify_response("calendar", {"items":[target]}))
        self.add_msg(state, "response", payload)
        return state

    async def _handle_update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        tmin, tmax, _ = _parse_window(text)
        events = await self.mcp.call_tool("gcal_list_events", max_results=1, time_min=tmin, time_max=tmax)
        if not events:
            self.add_msg(state, "response", {"intent":"update","message":"No events to update in window."})
            return state

        ev = events[0]
        # Simple demo: shift by +30m, unless a new HH:MM-HH:MM is provided
        m_time = re.search(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})", text)
        if m_time:
            # set explicit new times (local)
            day = _pick_day(text).strftime("%Y-%m-%d")
            new_start = f"{day}T{m_time.group(1)}:00+02:00" if len(m_time.group(1))==5 else f"{day}T{m_time.group(1)}+02:00"
            new_end   = f"{day}T{m_time.group(2)}:00+02:00" if len(m_time.group(2))==5 else f"{day}T{m_time.group(2)}+02:00"
        else:
            # shift by +30 minutes
            start_dt = _parse_iso_maybe_local(ev["start"]) + timedelta(minutes=30)
            end_dt   = _parse_iso_maybe_local(ev["end"])   + timedelta(minutes=30)
            new_start = start_dt.isoformat()
            new_end   = end_dt.isoformat()

        patch = {"start":{"dateTime":new_start}, "end":{"dateTime":new_end}}

        if not state.get("confirm"):
            self.add_msg(state, "response", {"intent":"update","requires_confirmation": True,
                                             "proposal":{"id": ev["id"], "patch": patch}})
            return state

        updated = await self.mcp.call_tool("gcal_update_event", event_id=ev["id"], patch=patch)
        payload: Dict[str, Any] = {"action":"updated","event":updated}
        if self.gemini:
            payload["summary_llm"] = self.gemini.chat("Event updated. Summarize briefly.")
        payload.update(verify_response("calendar", {"items":[updated]}))
        self.add_msg(state, "response", payload)
        return state

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent = _cal_intent(state.get("user_input",""))
        if intent == "create":
            return await self._handle_create(state)
        if intent == "delete":
            return await self._handle_delete(state)
        if intent == "update":
            return await self._handle_update(state)
        return await self._handle_read(state)
