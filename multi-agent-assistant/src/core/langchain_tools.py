from __future__ import annotations
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

# Import the existing server implementations
from src.mcp_integration.search_server import web_search
from src.mcp_integration.gmail_server import (
    list_recent_emails,
    read_email,
    send_email as gmail_send_impl,
)
from src.mcp_integration.calendar_server import (
    list_events,
    create_event,
    update_event,
    delete_event,
)

@tool("web_search")
async def web_search_tool(query: str, count: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for real-time information using Brave Search.
    Useful for finding current news, facts, or any information not in the model's training data.
    """
    return await web_search(query, count)

@tool("gmail_list_recent")
async def gmail_list_emails(query: Optional[str] = None, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    List recent emails from Gmail.
    The query parameter follows Gmail search syntax (e.g., 'from:someone', 'is:unread').
    """
    return await list_recent_emails(query, max_results)

@tool("gmail_read")
async def gmail_read_message(message_id: str) -> Dict[str, Any]:
    """
    Read the full content of a specific Gmail message by its ID.
    """
    return await read_email(message_id)

@tool("gmail_send")
async def gmail_send_message(to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Send an email via Gmail.
    Takes a recipient email address, subject, and message body.
    """
    # Mutating actions return a confirmation request that the UI handles
    return {
        "requires_confirmation": True,
        "message": f"I'm ready to send this email to {to}. Confirm?",
        "tool": "gmail_send",
        "args": {"to": to, "subject": subject, "body": body}
    }

async def gmail_send_actual(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Actual implementation for confirmed send."""
    return await gmail_send_impl(to, subject, body)

@tool("gcal_list_events")
async def google_calendar_list_events(
    max_results: int = 5,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List upcoming events from Google Calendar.
    time_min and time_max can be RFC3339 strings or natural language like 'today', 'tomorrow'.
    """
    return await list_events(max_results, time_min, time_max)

@tool("gcal_create_event")
async def google_calendar_create_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    location: str = "",
    description: str = "",
    attendees: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a new event in Google Calendar.
    start_iso and end_iso should be RFC3339 strings or natural language (e.g., 'tomorrow 3pm').
    """
    return {
        "requires_confirmation": True,
        "message": f"I'm ready to create the event '{summary}' on {start_iso}. Confirm?",
        "tool": "gcal_create_event",
        "args": {
            "summary": summary, "start_iso": start_iso, "end_iso": end_iso,
            "location": location, "description": description, "attendees": attendees
        }
    }

async def gcal_create_actual(summary: str, start_iso: str, end_iso: str, location: str, description: str, attendees: Optional[List[str]]) -> Dict[str, Any]:
    return await create_event(summary, start_iso, end_iso, location, description, attendees)

@tool("gcal_update_event")
async def google_calendar_update_event(
    event_id: str,
    summary: Optional[str] = None,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing Google Calendar event.
    Only provided fields will be updated.
    """
    return {
        "requires_confirmation": True,
        "message": f"I'm ready to update event {event_id}. Confirm?",
        "tool": "gcal_update_event",
        "args": {
            "event_id": event_id, "summary": summary, "start_iso": start_iso,
            "end_iso": end_iso, "location": location, "description": description
        }
    }

async def gcal_update_actual(event_id: str, summary: Optional[str], start_iso: Optional[str], end_iso: Optional[str], location: Optional[str], description: Optional[str]) -> Dict[str, Any]:
    return await update_event(event_id, summary, start_iso, end_iso, location, description)

@tool("gcal_delete_event")
async def google_calendar_delete_event(event_id: str) -> Dict[str, Any]:
    """
    Delete a Google Calendar event by its ID.
    """
    return {
        "requires_confirmation": True,
        "message": f"I'm ready to delete event {event_id}. Confirm?",
        "tool": "gcal_delete_event",
        "args": {"event_id": event_id}
    }

async def gcal_delete_actual(event_id: str) -> bool:
    return await delete_event(event_id)

def get_all_tools() -> List[Any]:
    """Returns a list of all tools for the LangChain agent."""
    return [
        web_search_tool,
        gmail_list_emails,
        gmail_read_message,
        gmail_send_message,
        google_calendar_list_events,
        google_calendar_create_event,
        google_calendar_update_event,
        google_calendar_delete_event,
    ]
