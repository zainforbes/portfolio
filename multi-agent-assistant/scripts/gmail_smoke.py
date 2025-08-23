import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import asyncio
from src.mcp_integration.gmail_server import list_recent_emails, read_email

async def main():
    emails = await list_recent_emails(max_results=3)
    print(f"Fetched {len(emails)} emails")
    for e in emails:
        print("-", e["subject"], "|", e["from"])
    if emails:
        detail = await read_email(emails[0]["id"])
        print("\nFirst email detail:", detail["subject"], "| body len:", len(detail.get("body","")))

if __name__ == "__main__":
    asyncio.run(main())
