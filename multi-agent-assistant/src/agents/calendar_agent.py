# src/agents/calendar_agent.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import re

from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient
from src.intelligence.verifier import verify_response

ZA = ZoneInfo("Africa/Johannesburg")
UTC = ZoneInfo("UTC")

def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")

def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)

# ---------- view window parsing ----------
def _parse_view_window(text: str) -> Tuple[Optional[str], Optional[str], str]:
    t = (text or "").lower()
    now = datetime.now(ZA)
    label = "upcoming"

    if "tomorrow" in t:
        label = "tomorrow"
        day = now + timedelta(days=1)
        return _to_utc_z(_start_of_day(day)), _to_utc_z(_end_of_day(day)), label
    if "today" in t:
        label = "today"
        return _to_utc_z(_start_of_day(now)), _to_utc_z(_end_of_day(now)), label
    if "next 7 days" in t or "next seven days" in t:
        label = "next 7 days"
        return _to_utc_z(now), _to_utc_z(now + timedelta(days=7)), label
    if "this week" in t:
        label = "this week"
        start = _start_of_day(now - timedelta(days=now.weekday()))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end), label
    if "next week" in t:
        label = "next week"
        days_until_mon = (7 - now.weekday()) % 7 or 7
        start = _start_of_day(now + timedelta(days=days_until_mon))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end), label

    # default: from now forward
    return _to_utc_z(now), None, label

def _parse_iso_local(s: str) -> datetime:
    if not s:
        return datetime.now(ZA)
    if len(s) == 10 and "T" not in s:
        return datetime.fromisoformat(s).replace(tzinfo=ZA)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ZA)

def _is_date_only(s: str) -> bool:
    return bool(s) and len(s) == 10 and "T" not in s

def _find_conflicts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    for i in range(1, len(events)):
        a, b = events[i-1], events[i]
        a_end_s = a.get("end","")
        b_start_s = b.get("start","")
        if _is_date_only(a_end_s):
            prev_day = datetime.fromisoformat(a_end_s).replace(tzinfo=ZA) - timedelta(days=1)
            a_end = _end_of_day(prev_day)
        else:
            a_end = _parse_iso_local(a_end_s)
        b_start = _start_of_day(datetime.fromisoformat(b_start_s).replace(tzinfo=ZA)) if _is_date_only(b_start_s) else _parse_iso_local(b_start_s)
        if b_start < a_end:
            conflicts.append({"a": a.get("summary",""), "b": b.get("summary",""), "range": f'{a.get("start","")} → {b.get("end","")}'})
    return conflicts

# ---------- intent parsing ----------
CREATE_RE = re.compile(r"\b(create|add|schedule)\b", re.I)
UPDATE_RE = re.compile(r"\b(update|move|reschedule|change)\b", re.I)
DELETE_RE = re.compile(r"\b(cancel|delete|remove)\b", re.I)

def _parse_create(text: str) -> Optional[Dict[str, Any]]:
    t = text or ""
    # title: quoted or after 'called/titled/about'
    m_title = re.search(r"'([^']+)'|\"([^\"]+)\"|(?:called|titled|about)\s+([^\n]+)", t, flags=re.I)
    title = next((g for g in (m_title.group(1) if m_title else None,
                              m_title.group(2) if m_title else None,
                              m_title.group(3) if m_title else None) if g), "New event")

    # date keywords
    now = datetime.now(ZA)
    day = now
    if re.search(r"\btomorrow\b", t, flags=re.I): day = now + timedelta(days=1)
    elif re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t):
        day = datetime.fromisoformat(re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t).group(1)).replace(tzinfo=ZA)

    # time range e.g. 10:00-10:30 or 3pm to 4pm
    def _parse_time(s: str) -> datetime:
        m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s.strip(), flags=re.I)
        if not m:
            return day.replace(hour=10, minute=0)
        h = int(m.group(1)); mins = int(m.group(2) or 0); ampm = (m.group(3) or "").lower()
        if ampm == "pm" and h < 12: h += 12
        if ampm == "am" and h == 12: h = 0
        return day.replace(hour=h, minute=mins)

    m_range = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:-|to|–|—)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", t, flags=re.I)
    if m_range:
        start = _parse_time(m_range.group(1))
        end   = _parse_time(m_range.group(2))
    else:
        # default 30 min slot at 10:00-10:30
        start = day.replace(hour=10, minute=0)
        end   = start + timedelta(minutes=30)

    loc = ""
    m_loc = re.search(r"\bat\s+([A-Za-z0-9 ,._-]+)", t, flags=re.I)
    if m_loc: loc = m_loc.group(1).strip()

    attendees = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)

    return {
        "summary": title.strip(),
        "start_iso": _to_utc_z(start),
        "end_iso":   _to_utc_z(end),
        "location": loc,
        "description": "",
        "attendees": attendees or None,
    }

class CalendarAgent(BaseAgent):
    name = "calendar"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def _view(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        tmin, tmax, label = _parse_view_window(text)
        events = await self.mcp.call_tool("gcal_list_events", max_results=20, time_min=tmin, time_max=tmax)
        events_sorted = sorted(events, key=lambda e: _parse_iso_local(e.get("start","")))
        payload: Dict[str, Any] = {
            "window": {"label": label, "time_min": tmin, "time_max": tmax},
            "summary": f"{len(events_sorted)} events",
            "items": events_sorted,
            "conflicts": _find_conflicts(events_sorted),
        }
        if self.gemini and events_sorted:
            bullets = "\n".join(f"- {e.get('summary','(no title)')} | {e.get('start','')} → {e.get('end','')}" for e in events_sorted[:6])
            prompt = "Summarize these events and note any conflicts (brief):\n" + bullets
            payload["summary_llm"] = self.gemini.chat(prompt)
        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _create(self, state: Dict[str, Any]) -> Dict[str, Any]:
        spec = _parse_create(state.get("user_input","")) or {}
        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Create this calendar event?",
                "proposal": spec,
                "intent": "calendar.create",
            })
            return state
        # confirmed: perform
        created = await self.mcp.call_tool("gcal_create_event", **spec)
        payload = {"result": "Event created.", "event": created}
        if self.gemini:
            prompt = (f"Confirm to the user that the event '{created.get('summary','')}' "
                      f"was created for {created.get('start','')} → {created.get('end','')}.")
            payload["summary_llm"] = self.gemini.chat(prompt)
        self.add_msg(state, "response", payload)
        return state

    async def _update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Heuristic: expect an event id and maybe new times/title
        text = state.get("user_input","")
        m_id = re.search(r"\b([a-zA-Z0-9_\-]{10,})\b", text)
        if not m_id:
            self.add_msg(state, "response", {"result": "Please provide the event ID to update."})
            return state

        spec = {"event_id": m_id.group(1)}
        # optional title
        m_title = re.search(r"(?:to|as)\s+'([^']+)'|\"([^\"]+)\"", text)
        if m_title:
            spec["summary"] = m_title.group(1) or m_title.group(2)

        # optional time range
        m_range = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:-|to|–|—)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", text, flags=re.I)
        if m_range:
            now = datetime.now(ZA)
            def _pt(s): 
                h = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s, flags=re.I); 
                hh=int(h.group(1)); mm=int(h.group(2) or 0); ap=(h.group(3) or "").lower()
                if ap=="pm" and hh<12: hh+=12
                if ap=="am" and hh==12: hh=0
                return now.replace(hour=hh, minute=mm)
            spec["start_iso"] = _to_utc_z(_pt(m_range.group(1)))
            spec["end_iso"]   = _to_utc_z(_pt(m_range.group(2)))

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Update this event?",
                "proposal": spec,
                "intent": "calendar.update",
            })
            return state

        updated = await self.mcp.call_tool("gcal_update_event", **spec)
        self.add_msg(state, "response", {"result": "Event updated.", "event": updated})
        return state

    async def _delete(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        m_id = re.search(r"\b([a-zA-Z0-9_\-]{10,})\b", text)
        if not m_id:
            self.add_msg(state, "response", {"result": "Please provide the event ID to delete."})
            return state
        spec = {"event_id": m_id.group(1)}
        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Delete this event?",
                "proposal": spec,
                "intent": "calendar.delete",
            })
            return state
        ok = await self.mcp.call_tool("gcal_delete_event", **spec)
        self.add_msg(state, "response", {"result": "Event deleted." if ok else "Delete failed."})
        return state

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        if DELETE_RE.search(text):
            return await self._delete(state)
        if UPDATE_RE.search(text):
            return await self._update(state)
        if CREATE_RE.search(text):
            return await self._create(state)
        return await self._view(state)
