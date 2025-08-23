from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Paths
CRED_PATH = Path("config/credentials.json")
TOKEN_GMAIL = Path("config/token_gmail.json")
TOKEN_CAL   = Path("config/token_calendar.json")
TOKEN_GENERIC = Path("config/token_generic.json")  # fallback

# Scopes
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CAL_SCOPES   = ["https://www.googleapis.com/auth/calendar"]

def _pick_token_path(scopes: List[str], token_path: Optional[str] = None) -> Path:
    if token_path:
        return Path(token_path)
    s = " ".join(scopes)
    if "calendar" in s:
        return TOKEN_CAL
    if "gmail" in s:
        return TOKEN_GMAIL
    return TOKEN_GENERIC

def get_credentials(scopes: List[str], token_path: Optional[str] = None) -> Credentials:
    """
    Returns valid credentials for the requested scopes, storing/refreshing
    a token file per service so scopes don't clash.
    """
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
