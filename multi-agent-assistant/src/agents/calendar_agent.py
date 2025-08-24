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

    # ---------- planner-aware helpers ----------
    async def _plan_view(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        # View by explicit bounds or NL window from user_input
        tmin = args.get("time_min"); tmax = args.get("time_max"); label = None
        if not tmin and not tmax:
            # Use NL window from the user's utterance
            user_text = state.get("user_input","")
            tmin, tmax, label = _parse_view_window(user_text)

        items = await self.mcp.call_tool("gcal_list_events", max_results=int(args.get("max_results") or 20),
                                         time_min=tmin, time_max=tmax)
        items_sorted = sorted(items, key=lambda e: _parse_iso_local(e.get("start","")))
        payload: Dict[str, Any] = {
            "mode": "list",
            "window": {"label": label or "", "time_min": tmin, "time_max": tmax},
            "summary": f"{len(items_sorted)} events",
            "items": items_sorted,
            "conflicts": _find_conflicts(items_sorted),
            "memory_patch": {"last_calendar_view": {"events": items_sorted, "window": {"label": label or "", "time_min": tmin, "time_max": tmax}}},
        }
        if self.gemini and items_sorted:
            bullets = "\n".join(f"- {e.get('summary','(no title)')} | {e.get('start','')} → {e.get('end','')}" for e in items_sorted[:6])
            prompt = "Summarize these events and note any conflicts (brief):\n" + bullets
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _plan_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        spec = {
            "summary": args.get("summary"),
            "start_iso": args.get("start_iso"),
            "end_iso": args.get("end_iso"),
            "location": args.get("location") or "",
            "description": args.get("description") or "",
            "attendees": args.get("attendees") or None,
        }
        # If any required field missing, propose a draft via confirm_context instead of failing
        if not (spec["summary"] and spec["start_iso"] and spec["end_iso"]):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "I have a partial event. Reply with details (title/time) or say 'cancel'.",
                "mode": "propose",
                "proposal": spec,
                "agent": "calendar",
                "intent": "calendar.create",
                "confirm_context": {"agent":"calendar","tool":"gcal_create_event","args": spec},
            })
            return state

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Create this calendar event? Reply 'send' to confirm.",
                "mode": "propose",
                "proposal": spec,
                "agent": "calendar",
                "intent": "calendar.create",
                "confirm_context": {"agent":"calendar","tool":"gcal_create_event","args": spec},
            })
            return state

        created = await self.mcp.call_tool("gcal_create_event", **spec)
        payload = {"mode":"created","result":"Event created.","event": created,
                   "memory_patch": {"last_calendar_mutation": {"event": created}}}
        if self.gemini:
            prompt = (f"Confirm to the user that the event '{created.get('summary','')}' "
                      f"was created for {created.get('start','')} → {created.get('end','')}.")
            payload["summary_llm"] = self.gemini.chat(prompt)

        self.add_msg(state, "response", payload)
        return state

    async def _plan_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        spec = {
            "event_id": args.get("event_id"),
            "summary": args.get("summary"),
            "start_iso": args.get("start_iso"),
            "end_iso": args.get("end_iso"),
            "location": args.get("location"),
            "description": args.get("description"),
        }
        if not spec["event_id"]:
            self.add_msg(state, "response", {"result": "Please provide the event ID to update."})
            return state

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Update this event? Reply 'send' to confirm.",
                "mode": "propose",
                "proposal": spec,
                "agent": "calendar",
                "intent": "calendar.update",
                "confirm_context": {"agent":"calendar","tool":"gcal_update_event","args": spec},
            })
            return state

        updated = await self.mcp.call_tool("gcal_update_event", **spec)
        payload = {"mode":"updated","result":"Event updated.","event": updated,
                   "memory_patch": {"last_calendar_mutation": {"event": updated}}}
        self.add_msg(state, "response", payload)
        return state

    async def _plan_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        spec = {"event_id": args.get("event_id")}
        if not spec["event_id"]:
            self.add_msg(state, "response", {"result": "Please provide the event ID to delete."})
            return state

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Delete this event? Reply 'send' to confirm.",
                "mode": "propose",
                "proposal": spec,
                "agent": "calendar",
                "intent": "calendar.delete",
                "confirm_context": {"agent":"calendar","tool":"gcal_delete_event","args": spec},
            })
            return state

        ok = await self.mcp.call_tool("gcal_delete_event", **spec)
        payload = {"mode":"deleted","result":"Event deleted." if ok else "Delete failed.", "target": spec}
        self.add_msg(state, "response", payload)
        return state

    # ---------- legacy NL paths (view default) ----------
    async def _view(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        tmin, tmax, label = _parse_view_window(text)
        events = await self.mcp.call_tool("gcal_list_events", max_results=20, time_min=tmin, time_max=tmax)
        events_sorted = sorted(events, key=lambda e: _parse_iso_local(e.get("start","")))
        payload: Dict[str, Any] = {
            "mode":"list",
            "window": {"label": label, "time_min": tmin, "time_max": tmax},
            "summary": f"{len(events_sorted)} events",
            "items": events_sorted,
            "conflicts": _find_conflicts(events_sorted),
            "memory_patch": {"last_calendar_view": {"events": events_sorted, "window": {"label": label, "time_min": tmin, "time_max": tmax}}},
        }
        if self.gemini and events_sorted:
            bullets = "\n".join(f"- {e.get('summary','(no title)')} | {e.get('start','')} → {e.get('end','')}" for e in events_sorted[:6])
            prompt = "Summarize these events and note any conflicts (brief):\n" + bullets
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---------- entrypoint ----------
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        step = state.get("current_step") or {}
        tool = (step.get("tool") or "").strip()
        args = (step.get("args") or {}).copy()
        args["_state"] = state

        if tool == "gcal_list_events":
            return await self._plan_view(args)
        if tool == "gcal_create_event":
            return await self._plan_create(args)
        if tool == "gcal_update_event":
            return await self._plan_update(args)
        if tool == "gcal_delete_event":
            return await self._plan_delete(args)

        # NL fallback: default to view
        return await self._view(state)
