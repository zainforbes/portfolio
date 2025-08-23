# src/mcp_integration/calendar_server.py
from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.utils.google_auth import get_credentials, CAL_SCOPES_RO, CAL_SCOPES_RW

DEFAULT_TZ = "Africa/Johannesburg"

def _utc_iso(dt: datetime) -> str:
    """Return RFC3339 with 'Z' (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")

def _service_read():
    creds = get_credentials(CAL_SCOPES_RO)
    return build("calendar", "v3", credentials=creds)

def _service_write():
    creds = get_credentials(CAL_SCOPES_RW)
    return build("calendar", "v3", credentials=creds)

# -------------------------
# READ
# -------------------------
async def list_events(
    max_results: int = 5,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    query: Optional[str] = None,
    order_by: str = "startTime",
) -> List[Dict[str, Any]]:
    """
    List events from the primary calendar.
    time_min/time_max: RFC3339 strings. If time_min omitted, defaults to now (UTC).
    Returns: [{id, summary, start, end, location}]
    """
    service = await asyncio.to_thread(_service_read)
    try:
        if not time_min:
            time_min = _utc_iso(datetime.now(timezone.utc))

        req = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy=order_by,
            q=query or None,
        )
        res = await asyncio.to_thread(req.execute)
        items = res.get("items", []) or []

        out: List[Dict[str, Any]] = []
        for ev in items:
            start = ev.get("start", {})
            end = ev.get("end", {})
            out.append({
                "id": ev.get("id", ""),
                "summary": ev.get("summary", ""),
                "start": start.get("dateTime") or start.get("date") or "",
                "end": end.get("dateTime") or end.get("date") or "",
                "location": ev.get("location", "") or "",
            })
        return out
    except HttpError as e:
        if e.resp.status == 403 and "insufficientPermissions" in str(e):
            raise RuntimeError(
                "Google Calendar token lacks required scope. Delete config/token_calendar_rw.json "
                "and re-auth with Calendar full scope if you intend to use CRUD; or "
                "config/token_calendar.json for read-only."
            ) from e
        raise

# -------------------------
# CREATE
# -------------------------
async def create_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    tz: str = DEFAULT_TZ,
    location: str = "",
    description: str = "",
    attendees: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create an event using dateTime strings (e.g., '2025-08-24T13:00:00+02:00').
    """
    service = await asyncio.to_thread(_service_write)
    body: Dict[str, Any] = {
        "summary": summary,
        "location": location,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": tz},
        "end":   {"dateTime": end_iso,   "timeZone": tz},
    }
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]

    try:
        req = service.events().insert(calendarId="primary", body=body)
        return await asyncio.to_thread(req.execute)
    except HttpError as e:
        if e.resp.status == 403 and "insufficientPermissions" in str(e):
            raise RuntimeError(
                "Insufficient permission to create events. Re-auth with Calendar full scope "
                "(delete config/token_calendar_rw.json)."
            ) from e
        raise

# -------------------------
# UPDATE (PATCH)
# -------------------------
async def update_event(event_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Patch an event. Example patch:
      {"summary":"New title"}
      {"start":{"dateTime":"2025-08-24T10:30:00+02:00"}, "end":{"dateTime":"2025-08-24T11:00:00+02:00"}}
    """
    service = await asyncio.to_thread(_service_write)
    try:
        req = service.events().patch(calendarId="primary", eventId=event_id, body=patch)
        return await asyncio.to_thread(req.execute)
    except HttpError as e:
        if e.resp.status == 404:
            raise RuntimeError(f"Event not found: {event_id}") from e
        if e.resp.status == 403 and "insufficientPermissions" in str(e):
            raise RuntimeError(
                "Insufficient permission to update events. Re-auth with Calendar full scope "
                "(delete config/token_calendar_rw.json)."
            ) from e
        raise

# -------------------------
# DELETE
# -------------------------
async def delete_event(event_id: str) -> None:
    """Delete an event by ID."""
    service = await asyncio.to_thread(_service_write)
    try:
        req = service.events().delete(calendarId="primary", eventId=event_id)
        await asyncio.to_thread(req.execute)
    except HttpError as e:
        if e.resp.status == 404:
            raise RuntimeError(f"Event not found: {event_id}") from e
        if e.resp.status == 403 and "insufficientPermissions" in str(e):
            raise RuntimeError(
                "Insufficient permission to delete events. Re-auth with Calendar full scope "
                "(delete config/token_calendar_rw.json)."
            ) from e
        raise
