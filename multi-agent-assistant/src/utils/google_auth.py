import os
import json
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CREDS_FILE = CONFIG_DIR / "credentials.json"
TOKEN_FILE = CONFIG_DIR / "token.json"

# Scopes define what access you need
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CAL_SCOPES   = ['https://www.googleapis.com/auth/calendar']
ALL_SCOPES   = sorted(set(GMAIL_SCOPES + CAL_SCOPES))

def get_credentials(scopes):
    """Load stored creds or prompt login if first time."""
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, scopes)
    # Refresh or create new creds if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), scopes)
            creds = flow.run_local_server(port=0)
        # Save creds for next time
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds
