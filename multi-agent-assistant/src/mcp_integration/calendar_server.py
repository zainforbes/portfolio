from typing import List, Dict, Optional
import asyncio
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.utils.google_auth import get_credentials, CAL_SCOPES

def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    # Google accepts Z-suffixed RFC3339
    return dt.isoformat().replace("+00:00", "Z")

async def list_events(
    max_results: int = 5,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> List[Dict]:
    """
    List upcoming events in the primary calendar.
    time_min/time_max are optional RFC3339 strings. If omitted, defaults to 'now'.
    Returns: [{id, summary, start, end, location}]
    """
    creds = get_credentials(CAL_SCOPES)
    service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)
    try:
        if not time_min:
            time_min = _utc_iso(datetime.now(timezone.utc))

        req = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        res = await asyncio.to_thread(req.execute)
        items = res.get("items", []) or []

        out: List[Dict] = []
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
                "Google Calendar token lacks required scope. Delete config/token_calendar.json "
                "and re-auth with Calendar read-only scope."
            ) from e
        raise
