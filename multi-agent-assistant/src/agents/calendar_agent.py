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
def _window_to_bounds(window: str) -> Tuple[Optional[str], Optional[str], str]:
    t = (window or "").lower().strip()
    now = datetime.now(ZA)
    if t in ("tomorrow",):
        day = now + timedelta(days=1)
        return _to_utc_z(_start_of_day(day)), _to_utc_z(_end_of_day(day)), "tomorrow"
    if t in ("today",):
        return _to_utc_z(_start_of_day(now)), _to_utc_z(_end_of_day(now)), "today"
    if t in ("next 7 days", "next seven days"):
        return _to_utc_z(now), _to_utc_z(now + timedelta(days=7)), "next 7 days"
    if t in ("this week",):
        start = _start_of_day(now - timedelta(days=now.weekday()))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end), "this week"
    if t in ("next week",):
        days_until_mon = (7 - now.weekday()) % 7 or 7
        start = _start_of_day(now + timedelta(days=days_until_mon))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end), "next week"
    # default: upcoming
    return _to_utc_z(now), None, "upcoming"

def _parse_view_window(text: str) -> Tuple[Optional[str], Optional[str], str]:
    # fallback NL parsing
    t = (text or "").lower()
    if "tomorrow" in t: return _window_to_bounds("tomorrow")
    if "today" in t:    return _window_to_bounds("today")
    if "next 7 days" in t or "next seven days" in t: return _window_to_bounds("next 7 days")
    if "this week" in t: return _window_to_bounds("this week")
    if "next week" in t: return _window_to_bounds("next week")
    return _window_to_bounds("")

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

# ---------- small parsing helpers ----------
CREATE_RE = re.compile(r"\b(create|add|schedule)\b", re.I)
UPDATE_RE = re.compile(r"\b(update|move|reschedule|change)\b", re.I)
DELETE_RE = re.compile(r"\b(cancel|delete|remove)\b", re.I)

def _parse_local_time(s: str, base: Optional[datetime] = None) -> datetime:
    """Parse 'tomorrow 20:00', 'today 8:30', '2025-08-25 13:00', '8pm' relative to ZA."""
    base = base or datetime.now(ZA)
    s0 = (s or "").strip().lower()

    # yyyy-mm-dd hh:mm?
    m_full = re.match(r"(\d{4}-\d{2}-\d{2})(?:[ T](\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?$", s0)
    if m_full:
        day = datetime.fromisoformat(m_full.group(1)).replace(tzinfo=ZA)
        if m_full.group(2):
            hh = int(m_full.group(2))
            mm = int(m_full.group(3) or 0)
            ap = (m_full.group(4) or "").lower()
            if ap == "pm" and hh < 12: hh += 12
            if ap == "am" and hh == 12: hh = 0
            return day.replace(hour=hh, minute=mm)
        return _start_of_day(day)

    # "tomorrow" or "today" with optional time
    if s0.startswith("tomorrow"):
        day = (base + timedelta(days=1))
        rest = s0.replace("tomorrow", "").strip()
    elif s0.startswith("today"):
        day = base
        rest = s0.replace("today", "").strip()
    else:
        day = base
        rest = s0

    if rest:
        m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", rest)
        if m:
            hh = int(m.group(1)); mm = int(m.group(2) or 0); ap = (m.group(3) or "").lower()
            if ap == "pm" and hh < 12: hh += 12
            if ap == "am" and hh == 12: hh = 0
            return day.replace(hour=hh, minute=mm)

    # default
    return day.replace(hour=10, minute=0)

def _parse_create_from_text(text: str) -> Optional[Dict[str, Any]]:
    t = text or ""
    m_title = re.search(r"'([^']+)'|\"([^\"]+)\"|(?:called|titled|about)\s+([^\n]+)", t, flags=re.I)
    title = next((g for g in (m_title.group(1) if m_title else None,
                              m_title.group(2) if m_title else None,
                              m_title.group(3) if m_title else None) if g), "New event")
    now = datetime.now(ZA)
    day = now
    if re.search(r"\btomorrow\b", t, flags=re.I): day = now + timedelta(days=1)
    elif re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t):
        day = datetime.fromisoformat(re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t).group(1)).replace(tzinfo=ZA)

    def _parse_time(s: str) -> datetime:
        m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s.strip(), flags=re.I)
        if not m:
            return day.replace(hour=10, minute=0)
        h = int(m.group(1)); mins = int(m.group(2) or 0); ap=(m.group(3) or "").lower()
        if ap == "pm" and h < 12: h += 12
        if ap == "am" and h == 12: h = 0
        return day.replace(hour=h, minute=mins)

    m_range = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:-|to|–|—)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", t, flags=re.I)
    if m_range:
        start = _parse_time(m_range.group(1))
        end   = _parse_time(m_range.group(2))
    else:
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

    # ---- planner-aware paths ----
    async def _plan_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        window = (args.get("window") or "").strip().lower()
        if window:
            tmin, tmax, label = _window_to_bounds(window)
        else:
            tmin, tmax, label = _parse_view_window(state.get("user_input",""))

        events = await self.mcp.call_tool("gcal_list_events", max_results=int(args.get("max_results") or 20), time_min=tmin, time_max=tmax)
        events_sorted = sorted(events, key=lambda e: _parse_iso_local(e.get("start","")))
        payload: Dict[str, Any] = {
            "window": {"label": label, "time_min": tmin, "time_max": tmax},
            "summary": f"{len(events_sorted)} events",
            "items": events_sorted,
            "conflicts": _find_conflicts(events_sorted),
            "suggested_prompts": ["create 'Standup' tomorrow 09:00-09:15", "move event <id> to 11:00-11:30", "delete event <id>"],
        }
        if self.gemini and events_sorted:
            bullets = "\n".join(f"- {e.get('summary','(no title)')} | {e.get('start','')} → {e.get('end','')}" for e in events_sorted[:6])
            prompt = "Summarize these events and note any conflicts (brief):\n" + bullets
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _plan_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        summary = (args.get("summary") or "").strip()
        start_local = (args.get("start_local") or "").strip()
        end_local   = (args.get("end_local") or "").strip()
        location    = (args.get("location") or "").strip()
        description = (args.get("description") or "").strip()
        attendees   = args.get("attendees") or None
        if isinstance(attendees, str): attendees = [attendees]

        if start_local and end_local:
            sdt = _parse_local_time(start_local)
            edt = _parse_local_time(end_local, base=sdt)
            spec = {
                "summary": summary or "New event",
                "start_iso": _to_utc_z(sdt),
                "end_iso": _to_utc_z(edt),
                "location": location,
                "description": description,
                "attendees": attendees,
            }
        else:
            # fallback to NL parsing from the user's text
            spec = _parse_create_from_text(state.get("user_input","")) or {}

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Create this calendar event?",
                "proposal": spec,
                "agent": "calendar",
                "intent": "calendar.create",
                "confirm_context": {"agent":"calendar","tool":"gcal_create_event","args": spec},
            })
            return state

        created = await self.mcp.call_tool("gcal_create_event", **spec)
        payload: Dict[str, Any] = {"result": "Event created.", "event": created}
        if self.gemini:
            prompt = (f"Confirm to the user that the event '{created.get('summary','')}' "
                      f"was created for {created.get('start','')} → {created.get('end','')}.")
            payload["summary_llm"] = self.gemini.chat(prompt)
        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _plan_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        event_id = (args.get("event_id") or "").strip()
        if not event_id:
            self.add_msg(state, "response", {"result": "Please provide the event ID to update."})
            return state

        patch: Dict[str, Any] = {"event_id": event_id}
        if "summary" in args and args["summary"] is not None:
            patch["summary"] = (args["summary"] or "").strip()
        if "location" in args and args["location"] is not None:
            patch["location"] = (args["location"] or "").strip()
        if "description" in args and args["description"] is not None:
            patch["description"] = (args["description"] or "").strip()
        if args.get("start_local"):
            sdt = _parse_local_time(str(args["start_local"]))
            patch["start_iso"] = _to_utc_z(sdt)
        if args.get("end_local"):
            edt = _parse_local_time(str(args["end_local"]))
            patch["end_iso"] = _to_utc_z(edt)

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Update this event?",
                "proposal": patch,
                "agent": "calendar",
                "intent": "calendar.update",
                "confirm_context": {"agent":"calendar","tool":"gcal_update_event","args": patch},
            })
            return state

        updated = await self.mcp.call_tool("gcal_update_event", **patch)
        payload = {"result": "Event updated.", "event": updated}
        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _plan_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        event_id = (args.get("event_id") or "").strip()
        if not event_id:
            self.add_msg(state, "response", {"result": "Please provide the event ID to delete."})
            return state

        if not state.get("confirm"):
            self.add_msg(state, "response", {
                "requires_confirmation": True,
                "message": "Delete this event?",
                "proposal": {"event_id": event_id},
                "agent": "calendar",
                "intent": "calendar.delete",
                "confirm_context": {"agent":"calendar","tool":"gcal_delete_event","args": {"event_id": event_id}},
            })
            return state

        ok = await self.mcp.call_tool("gcal_delete_event", event_id=event_id)
        payload = {"result": "Event deleted." if ok else "Delete failed."}
        payload.update(verify_response("calendar", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---- NL fallbacks ----
    async def _view(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tmin, tmax, label = _parse_view_window(state.get("user_input",""))
        return await self._plan_list({"window": label, "max_results": 20, "_state": state})

    async def _create(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._plan_create({"_state": state})

    async def _update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # fallback: expect ID and maybe new range in the text
        text = state.get("user_input","")
        m_id = re.search(r"\b([a-zA-Z0-9_\-]{10,})\b", text)
        args: Dict[str, Any] = {"_state": state}
        if m_id:
            args["event_id"] = m_id.group(1)
        return await self._plan_update(args)

    async def _delete(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("user_input","")
        m_id = re.search(r"\b([a-zA-Z0-9_\-]{10,})\b", text)
        args: Dict[str, Any] = {"_state": state}
        if m_id:
            args["event_id"] = m_id.group(1)
        return await self._plan_delete(args)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        step = state.get("current_step") or {}
        tool = (step.get("tool") or "").strip()
        args = (step.get("args") or {}).copy()
        args["_state"] = state

        if tool == "gcal_list_events":
            return await self._plan_list(args)
        if tool == "gcal_create_event":
            return await self._plan_create(args)
        if tool == "gcal_update_event":
            return await self._plan_update(args)
        if tool == "gcal_delete_event":
            return await self._plan_delete(args)

        # NL fallbacks
        text = state.get("user_input","")
        if DELETE_RE.search(text):  return await self._delete(state)
        if UPDATE_RE.search(text):  return await self._update(state)
        if CREATE_RE.search(text):  return await self._create(state)
        return await self._view(state)
