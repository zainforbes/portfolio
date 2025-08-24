# src/utils/google_auth.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Paths
CRED_PATH = Path("config/credentials.json")   # Google OAuth client secret (downloaded from GCP)
TOKEN_PATH = Path("config/token.json")        # Single unified token file

# Scopes (FULL CRUD)
GMAIL_SCOPES: List[str] = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]
CAL_SCOPES: List[str] = [
    "https://www.googleapis.com/auth/calendar",
]

# Unified (we’ll always request the full set so everything works)
ALL_SCOPES: List[str] = sorted(set(GMAIL_SCOPES + CAL_SCOPES))


def get_credentials(scopes: Optional[List[str]] = None) -> Credentials:
    """
    Return Google OAuth Credentials with full CRUD scopes (Gmail modify/send + Calendar).
    The optional 'scopes' arg is accepted for compatibility but we always ensure ALL_SCOPES
    are requested so every feature works without juggling multiple tokens.
    """
    if not CRED_PATH.exists():
        raise FileNotFoundError(
            "Missing config/credentials.json (Google OAuth client secret). "
            "Download it from your Google Cloud project and place it there."
        )

    # We always include ALL_SCOPES
    requested_scopes = sorted(set((scopes or []) + ALL_SCOPES))

    creds: Optional[Credentials] = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), requested_scopes)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(CRED_PATH), requested_scopes)
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds
