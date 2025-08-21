from src.agents.calendar_client import GoogleCalendarClient

calendar = GoogleCalendarClient()
events = calendar.get_upcoming_events(5)

if not events:
    print("No upcoming events found.")
else:
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        print(start, event["summary"])
