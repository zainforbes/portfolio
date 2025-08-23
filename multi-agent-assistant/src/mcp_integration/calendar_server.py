from typing import List, Dict
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from src.utils.google_auth import get_credentials, CAL_SCOPES

async def list_events(max_results: int = 5) -> List[Dict]:
    creds = get_credentials(CAL_SCOPES)
    service = build("calendar", "v3", credentials=creds)

    now = datetime.utcnow().isoformat() + "Z"
    events_result = service.events().list(
        calendarId="primary",
        timeMin=now,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])
    out = []
    for e in events:
        out.append({
            "id": e["id"],
            "summary": e.get("summary", ""),
            "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date")),
            "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date")),
            "location": e.get("location", ""),
        })
    return out
