from src.agents.gmail_client import GmailClient

gmail = GmailClient()
msgs = gmail.list_messages(5)

if not msgs:
    print("No messages found.")
else:
    for m in msgs:
        print("Message ID:", m["id"])
