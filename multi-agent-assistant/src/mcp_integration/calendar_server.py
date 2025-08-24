# src/mcp_integration/calendar_server.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.utils.google_auth import get_credentials, CAL_SCOPES

LOCAL_TZ = ZoneInfo("Africa/Johannesburg")
UTC = timezone.utc

# ----------------------------
# Generic time helpers
# ----------------------------
def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")

def _parse_iso_loose(s: Optional[str]) -> Optional[datetime]:
    """Try to parse RFC3339/ISO; return aware UTC dt or None."""
    if not s:
        return None
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None

# ----------------------------
# NL → window parsing
# ----------------------------
def _nl_window(phrase: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Convert a natural-language phrase into (timeMin_utcZ, timeMax_utcZ_or_None).
    Supported:
      "now"
      "today", "tomorrow", "later today"
      "this morning" / "this afternoon" / "this evening" / "tonight"
      "this week", "next week"
      "next 7 days", "next seven days"
    """
    if not phrase:
        return None
    t = phrase.strip().lower()
    now_local = datetime.now(LOCAL_TZ)

    morning   = (now_local.replace(hour=8,  minute=0), now_local.replace(hour=12, minute=0))
    afternoon = (now_local.replace(hour=12, minute=0), now_local.replace(hour=17, minute=0))
    evening   = (now_local.replace(hour=17, minute=0), now_local.replace(hour=23, minute=59, microsecond=999999))

    if t == "now":
        return _to_utc_z(now_local), None
    if t == "later today":
        return _to_utc_z(now_local), _to_utc_z(_end_of_day(now_local))
    if t in ("this morning", "morning"):
        return _to_utc_z(morning[0]), _to_utc_z(morning[1])
    if t in ("this afternoon", "afternoon"):
        return _to_utc_z(afternoon[0]), _to_utc_z(afternoon[1])
    if t in ("this evening", "tonight", "evening", "night"):
        return _to_utc_z(evening[0]), _to_utc_z(evening[1])
    if t == "today":
        return _to_utc_z(_start_of_day(now_local)), _to_utc_z(_end_of_day(now_local))
    if t == "tomorrow":
        tm = now_local + timedelta(days=1)
        return _to_utc_z(_start_of_day(tm)), _to_utc_z(_end_of_day(tm))
    if t == "this week":
        start = _start_of_day(now_local - timedelta(days=now_local.weekday()))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end)
    if t == "next week":
        days_until_mon = (7 - now_local.weekday()) % 7 or 7
        start = _start_of_day(now_local + timedelta(days=days_until_mon))
        end   = _end_of_day(start + timedelta(days=6))
        return _to_utc_z(start), _to_utc_z(end)
    if t in ("next 7 days", "next seven days"):
        return _to_utc_z(now_local), _to_utc_z(now_local + timedelta(days=7))

    return None

def _normalize_bounds(time_min: Optional[str], time_max: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Accept RFC3339 strings or NL phrases for min/max and return proper RFC3339 'Z' strings.
    If both are None -> (now, None).
    If a phrase defines a full window (e.g., 'tomorrow'), we use it unless the other bound is ISO.
    """
    min_phrase = _nl_window(time_min) if isinstance(time_min, str) else None
    max_phrase = _nl_window(time_max) if isinstance(time_max, str) else None

    min_iso = _parse_iso_loose(time_min) if isinstance(time_min, str) else None
    max_iso = _parse_iso_loose(time_max) if isinstance(time_max, str) else None

    if min_phrase and not min_iso:
        min_z, phrase_max_z = min_phrase
        if max_iso:
            return min_z, _to_utc_z(max_iso)
        if max_phrase:
            _, max_z_phrase = max_phrase
            return min_z, max_z_phrase or phrase_max_z
        return min_z, phrase_max_z

    if max_phrase and not max_iso:
        if min_iso:
            return _to_utc_z(min_iso), max_phrase[1] or None
        if max_phrase[0] and max_phrase[1]:
            return max_phrase[0], max_phrase[1]
        now_z, _ = _nl_window("now")
        return now_z, max_phrase[1]

    if min_iso and max_iso:
        return _to_utc_z(min_iso), _to_utc_z(max_iso)
    if min_iso and not max_iso:
        return _to_utc_z(min_iso), None
    if not min_iso and max_iso:
        now_local = datetime.now(LOCAL_TZ)
        return _to_utc_z(now_local), _to_utc_z(max_iso)

    now_local = datetime.now(LOCAL_TZ)
    return _to_utc_z(now_local), None

# ----------------------------
# Normalization for create/update
# ----------------------------
def _norm_iso_datetime(s: str) -> str:
    """
    Strict dateTime normalizer for create/update: always returns RFC3339 dateTime in UTC (Z).
    If a date-only string arrives, we treat it as start-of-day local.
    """
    if not s:
        raise ValueError("Missing datetime")
    # date-only: interpret local start-of-day
    if len(s) == 10 and s.count("-") == 2 and "T" not in s:
        dt = datetime.fromisoformat(s).replace(tzinfo=LOCAL_TZ)
    else:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
    return _to_utc_z(dt)

def _normalize_event(ev: Dict) -> Dict:
    start = ev.get("start", {})
    end   = ev.get("end", {})
    return {
        "id": ev.get("id",""),
        "summary": ev.get("summary",""),
        "start": start.get("dateTime") or start.get("date") or "",
        "end":   end.get("dateTime")   or end.get("date")   or "",
        "location": ev.get("location","") or "",
        "description": ev.get("description","") or "",
    }

# ----------------------------
# READ
# ----------------------------
async def list_events(
    max_results: int = 5,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> List[Dict]:
    creds = get_credentials(CAL_SCOPES)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)

    try:
        tmin_z, tmax_z = _normalize_bounds(time_min, time_max)

        req = service.events().list(
            calendarId="primary",
            timeMin=tmin_z,
            timeMax=tmax_z,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        res = await asyncio.to_thread(req.execute)
        return [_normalize_event(e) for e in (res.get("items") or [])]
    except HttpError as e:
        if e.resp.status == 403 and "insufficientPermissions" in str(e):
            raise RuntimeError(
                "Google Calendar token lacks required permissions. Delete token and re-auth with CAL_SCOPES."
            ) from e
        raise

# ----------------------------
# CREATE
# ----------------------------
async def create_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    location: str = "",
    description: str = "",
    attendees: Optional[List[str]] = None,
) -> Dict:
    creds = get_credentials(CAL_SCOPES)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    body = {
        "summary": summary,
        "location": location or None,
        "description": description or None,
        "start": {"dateTime": _norm_iso_datetime(start_iso)},
        "end":   {"dateTime": _norm_iso_datetime(end_iso)},
    }
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]

    try:
        res = await asyncio.to_thread(
            service.events().insert(calendarId="primary", body=body, sendUpdates="all").execute
        )
        return _normalize_event(res)
    except HttpError as e:
        if e.resp.status == 403:
            raise RuntimeError(
                "Google Calendar token lacks required permissions. Delete token and re-auth with CAL_SCOPES."
            ) from e
        raise

# ----------------------------
# UPDATE (PATCH)
# ----------------------------
async def update_event(
    event_id: str,
    summary: Optional[str] = None,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict:
    creds = get_credentials(CAL_SCOPES)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    patch: Dict = {}
    if summary is not None:     patch["summary"]     = summary
    if location is not None:    patch["location"]    = location or None
    if description is not None: patch["description"] = description or None
    if start_iso is not None:   patch.setdefault("start", {})["dateTime"] = _norm_iso_datetime(start_iso)
    if end_iso   is not None:   patch.setdefault("end",   {})["dateTime"] = _norm_iso_datetime(end_iso)

    try:
        res = await asyncio.to_thread(
            service.events().patch(
                calendarId="primary", eventId=event_id, body=patch, sendUpdates="all"
            ).execute
        )
        return _normalize_event(res)
    except HttpError as e:
        if e.resp.status == 404:
            raise RuntimeError(f"Event not found: {event_id}") from e
        raise

# ----------------------------
# DELETE
# ----------------------------
async def delete_event(event_id: str) -> bool:
    creds = get_credentials(CAL_SCOPES)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    try:
        await asyncio.to_thread(
            service.events().delete(calendarId="primary", eventId=event_id, sendUpdates="all").execute
        )
        return True
    except HttpError as e:
        if e.resp.status == 404:
            raise RuntimeError(f"Event not found: {event_id}") from e
        raise
