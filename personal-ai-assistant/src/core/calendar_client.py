import os
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

class GoogleCalendarClient:
    def __init__(self):
        self.creds = None
        self.token_path = "config/token.pickle"
        self.secret_file = os.getenv("GOOGLE_CLIENT_SECRET")

        # Load existing token if it exists
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
                self.creds = flow.run_local_server(port=0)

            # Save token for next run
            with open(self.token_path, "wb") as token:
                pickle.dump(self.creds, token)

        # Build Calendar service
        self.service = build("calendar", "v3", credentials=self.creds)

    def get_upcoming_events(self, max_results=5):
        events_result = (
            self.service.events()
            .list(
                calendarId="primary",
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return events_result.get("items", [])
