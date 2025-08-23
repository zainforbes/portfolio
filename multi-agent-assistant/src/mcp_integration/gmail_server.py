from typing import List, Dict, Optional
import asyncio
from base64 import urlsafe_b64decode
from googleapiclient.discovery import build

from src.utils.google_auth import get_credentials, GMAIL_SCOPES

# --- helpers -------------------------------------------------

def _headers_to_dict(payload_headers: List[Dict]) -> Dict[str, str]:
    return {h.get("name", ""): h.get("value", "") for h in (payload_headers or [])}

def _extract_plain_text(payload: Dict) -> str:
    """
    Recursively walk the payload to find text/plain; fallback to first text/* part.
    """
    if not payload:
        return ""
    mime = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}
    data = body.get("data")

    # leaf node
    if data:
        try:
            decoded = urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        return decoded

    # multipart
    parts = payload.get("parts") or []
    # prefer text/plain
    for p in parts:
        if p.get("mimeType", "").startswith("text/plain"):
            txt = _extract_plain_text(p)
            if txt:
                return txt
    # fallback: any text/*
    for p in parts:
        if p.get("mimeType", "").startswith("text/"):
            txt = _extract_plain_text(p)
            if txt:
                return txt
    return ""

# --- API functions ------------------------------------------

async def list_recent_emails(query: Optional[str] = None, max_results: int = 5) -> List[Dict]:
    """
    Returns: [{id, from, subject, snippet, date}]
    """
    creds = get_credentials(GMAIL_SCOPES)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)

    results = await asyncio.to_thread(
        service.users().messages().list,
        userId="me", q=query or "", maxResults=max_results
    )
    results = await asyncio.to_thread(results.execute)
    messages = results.get("messages", [])

    out: List[Dict] = []
    for m in messages:
        msg_req = service.users().messages().get(userId="me", id=m["id"], format="metadata", metadataHeaders=["From","Subject","Date"])
        msg = await asyncio.to_thread(msg_req.execute)
        headers = _headers_to_dict(msg.get("payload", {}).get("headers", []))
        out.append({
            "id": m["id"],
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "snippet": msg.get("snippet", ""),
            "date": headers.get("Date", ""),
        })
    return out

async def read_email(message_id: str) -> Dict:
    """
    Returns a single email with decoded plain-text body (best effort).
    {id, threadId, subject, from, to, date, body, snippet}
    """
    creds = get_credentials(GMAIL_SCOPES)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)

    req = service.users().messages().get(userId="me", id=message_id, format="full")
    msg = await asyncio.to_thread(req.execute)

    headers = _headers_to_dict(msg.get("payload", {}).get("headers", []))
    body_text = _extract_plain_text(msg.get("payload", {})) or msg.get("snippet", "")

    return {
        "id": msg.get("id"),
        "threadId": msg.get("threadId"),
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "date": headers.get("Date", ""),
        "body": body_text,
        "snippet": msg.get("snippet", ""),
    }
