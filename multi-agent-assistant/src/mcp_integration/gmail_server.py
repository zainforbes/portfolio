from __future__ import annotations
import base64, asyncio
from email.message import EmailMessage
import base64
from typing import List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.utils import formataddr
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.utils.google_auth import get_credentials, GMAIL_SCOPES_RO, GMAIL_SCOPES_RW

# ---- READ ----
async def list_recent_emails(query: Optional[str] = None, max_results: int = 10) -> List[Dict]:
    creds = get_credentials(GMAIL_SCOPES_RO)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)
    try:
        mlist = await asyncio.to_thread(service.users().messages().list,
                                        userId="me", maxResults=max_results, q=query or "")
        res = await asyncio.to_thread(mlist.execute)
        ids = [m["id"] for m in res.get("messages", [])]
        out: List[Dict] = []
        for mid in ids:
            req = service.users().messages().get(userId="me", id=mid, format="metadata", metadataHeaders=["From","Subject","Date"])
            m = await asyncio.to_thread(req.execute)
            headers = {h["name"]: h["value"] for h in m.get("payload", {}).get("headers", [])}
            out.append({
                "id": m.get("id",""),
                "from": headers.get("From",""),
                "subject": headers.get("Subject",""),
                "date": headers.get("Date",""),
                "snippet": m.get("snippet",""),
            })
        return out
    except HttpError as e:
        raise RuntimeError(str(e)) from e

async def read_email(message_id: str) -> Dict:
    creds = get_credentials(GMAIL_SCOPES_RO)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)
    req = service.users().messages().get(userId="me", id=message_id, format="full")
    m = await asyncio.to_thread(req.execute)
    return m

# ---- WRITE ----
def _build_raw(from_name: Optional[str], to: List[str], subject: str, body: str, cc: List[str] | None = None, bcc: List[str] | None = None) -> str:
    msg = MIMEText(body, _subtype="plain", _charset="utf-8")
    msg["To"] = ", ".join(to)
    if cc:  msg["Cc"]  = ", ".join(cc)
    if bcc: msg["Bcc"] = ", ".join(bcc)
    msg["From"] = formataddr((from_name or "", "me"))  # "me" is fine; Gmail replaces with your account
    msg["Subject"] = subject
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()

async def create_draft(to: List[str], subject: str, body: str, cc: List[str] | None = None, bcc: List[str] | None = None) -> Dict:
    creds = get_credentials(GMAIL_SCOPES_RW)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)
    raw = _build_raw(None, to, subject, body, cc, bcc)
    draft = {"message": {"raw": raw}}
    req = service.users().drafts().create(userId="me", body=draft)
    return await asyncio.to_thread(req.execute)

async def send_email(to: List[str], subject: str, body: str, cc: List[str] | None = None, bcc: List[str] | None = None) -> Dict:
    creds = get_credentials(GMAIL_SCOPES_RW)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)
    raw = _build_raw(None, to, subject, body, cc, bcc)
    req = service.users().messages().send(userId="me", body={"raw": raw})
    return await asyncio.to_thread(req.execute)

async def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Send an email via Gmail API."""
    creds = get_credentials(GMAIL_SCOPES_RW)  # ensure scope includes gmail.send
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)

    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    req = service.users().messages().send(userId="me", body={"raw": raw})
    res = await asyncio.to_thread(req.execute)
    return {"id": res.get("id"), "status": "sent"}

async def mark_read(ids: List[str]) -> Dict[str, Any]:
    """Batch remove UNREAD label."""
    creds = get_credentials(GMAIL_SCOPES_RW)
    service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds)
    req = service.users().messages().batchModify(
        userId="me",
        body={"ids": ids, "removeLabelIds": ["UNREAD"]}
    )
    res = await asyncio.to_thread(req.execute)
    return {"modified": res.get("resultSizeEstimate", len(ids))}