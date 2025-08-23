from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

CRED_PATH = Path("config/credentials.json")
TOKEN_GMAIL = Path("config/token_gmail.json")          # read-only (if you keep it)
TOKEN_GMAIL_RW = Path("config/token_gmail_rw.json")    # read+write/send
TOKEN_CAL_RO = Path("config/token_calendar.json")      # read-only
TOKEN_CAL_RW = Path("config/token_calendar_rw.json")   # read+write

# Scopes
GMAIL_SCOPES_RO  = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_SCOPES_RW  = ["https://www.googleapis.com/auth/gmail.modify", "https://www.googleapis.com/auth/gmail.send"]
CAL_SCOPES_RO    = ["https://www.googleapis.com/auth/calendar.readonly"]
CAL_SCOPES_RW    = ["https://www.googleapis.com/auth/calendar"]  # full CRUD

def _pick_token_path(scopes: List[str], token_path: Optional[str] = None) -> Path:
    if token_path: return Path(token_path)
    s = " ".join(scopes)
    if "gmail.modify" in s or "gmail.send" in s: return TOKEN_GMAIL_RW
    if "gmail.readonly" in s:                    return TOKEN_GMAIL
    if "calendar.readonly" in s:                 return TOKEN_CAL_RO
    if "calendar" in s:                          return TOKEN_CAL_RW
    return Path("config/token_generic.json")

def get_credentials(scopes: List[str], token_path: Optional[str] = None) -> Credentials:
    if not CRED_PATH.exists():
        raise FileNotFoundError("Missing config/credentials.json (Google OAuth client secret).")
    token_p = _pick_token_path(scopes, token_path)
    creds: Optional[Credentials] = None
    if token_p.exists():
        creds = Credentials.from_authorized_user_file(str(token_p), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CRED_PATH), scopes)
            creds = flow.run_local_server(port=0)
        token_p.write_text(creds.to_json())
    return creds
