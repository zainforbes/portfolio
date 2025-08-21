import os
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

class GmailClient:
    def __init__(self):
        self.creds = None
        self.token_path = "config/gmail_token.pickle"
        self.secret_file = os.getenv("GOOGLE_CLIENT_SECRET")

        # Load saved token
        if os.path.exists(self.token_path):
            with open(self.token_path, "rb") as token:
                self.creds = pickle.load(token)

        # If no creds, go through auth flow
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.secret_file, SCOPES
                )
                self.creds = flow.run_local_server(port=0)  # Use console-based auth

            # Save token for next time
            with open(self.token_path, "wb") as token:
                pickle.dump(self.creds, token)

        # Build Gmail service
        self.service = build("gmail", "v1", credentials=self.creds)

    def list_messages(self, max_results=5):
        """Fetch message metadata (IDs only for now)."""
        results = (
            self.service.users()
            .messages()
            .list(userId="me", maxResults=max_results)
            .execute()
        )
        return results.get("messages", [])
    
    def get_message_details(self, message_id: str):
        msg = self.service.users().messages().get(
            userId="me", id=message_id, format="metadata", metadataHeaders=["Subject"]
        ).execute()
        headers = msg.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No subject")
        snippet = msg.get("snippet", "")
        return f"📧 {subject} → {snippet[:80]}..."
