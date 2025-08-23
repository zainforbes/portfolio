# src/mcp_integration/calendar_server.py
from __future__ import annotations
from typing import List, Dict, Optional
import asyncio
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.utils.google_auth import (
    get_credentials,
    CAL_SCOPES_RW,   # NEW
)

def _norm_iso(s: str) -> str:
    """Ensure RFC3339 with timezone. Accepts 'YYYY-MM-DD' or any ISO."""
    if not s:
        return s
    # date only -> treat as all-day
    if len(s) == 10 and s.count("-") == 2 and "T" not in s:
        return s  # all-day
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")

def _normalize(ev: Dict) -> Dict:
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

# -------- READ --------
async def list_events(
    max_results: int = 5,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    token_file: str = "token_calendar_ro.json",         # NEW default read-only token
) -> List[Dict]:
    creds = get_credentials(CAL_SCOPES_RW, token_filename=token_file)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    if not time_min:
        time_min = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if time_max:
        time_max = _norm_iso(time_max)

    try:
        req = service.events().list(
            calendarId="primary",
            timeMin=_norm_iso(time_min),
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        res = await asyncio.to_thread(req.execute)
        return [_normalize(e) for e in (res.get("items") or [])]
    except HttpError as e:
        if e.resp.status == 403 and "insufficientPermissions" in str(e):
            raise RuntimeError(
                "Calendar token lacks required READ scope. "
                "Delete the token file you are using and re-auth with CAL_SCOPES_READ."
            ) from e
        raise

# -------- CREATE --------
async def create_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    location: str = "",
    description: str = "",
    attendees: Optional[List[str]] = None,
    token_file: str = "token_calendar_rw.json",         # NEW: write token
) -> Dict:
    creds = get_credentials(CAL_SCOPES_RW, token_filename=token_file)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    body = {
        "summary": summary,
        "location": location or None,
        "description": description or None,
        "start": {"dateTime": _norm_iso(start_iso)},
        "end":   {"dateTime": _norm_iso(end_iso)},
    }
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]

    try:
        res = await asyncio.to_thread(service.events().insert(
            calendarId="primary", body=body, sendUpdates="all"
        ).execute)
        return _normalize(res)
    except HttpError as e:
        if e.resp.status == 403:
            raise RuntimeError(
                "Calendar token lacks required WRITE scope. "
                "Delete the write token and re-auth with CAL_SCOPES_WRITE."
            ) from e
        raise

# -------- UPDATE (PATCH) --------
async def update_event(
    event_id: str,
    summary: Optional[str] = None,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
    token_file: str = "token_calendar_rw.json",
) -> Dict:
    creds = get_credentials(CAL_SCOPES_RW, token_filename=token_file)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    patch: Dict = {}
    if summary is not None: patch["summary"] = summary
    if location is not None: patch["location"] = location or None
    if description is not None: patch["description"] = description or None
    if start_iso is not None: patch.setdefault("start", {})["dateTime"] = _norm_iso(start_iso)
    if end_iso   is not None: patch.setdefault("end",   {})["dateTime"] = _norm_iso(end_iso)
    try:
        res = await asyncio.to_thread(service.events().patch(
            calendarId="primary", eventId=event_id, body=patch, sendUpdates="all"
        ).execute)
        return _normalize(res)
    except HttpError as e:
        if e.resp.status == 404:
            raise RuntimeError(f"Event not found: {event_id}") from e
        raise

# -------- DELETE --------
async def delete_event(
    event_id: str,
    token_file: str = "token_calendar_rw.json",
) -> bool:
    creds = get_credentials(CAL_SCOPES_RW, token_filename=token_file)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    try:
        await asyncio.to_thread(service.events().delete(
            calendarId="primary", eventId=event_id, sendUpdates="all"
        ).execute)
        return True
    except HttpError as e:
        if e.resp.status == 404:
            raise RuntimeError(f"Event not found: {event_id}") from e
        raise
