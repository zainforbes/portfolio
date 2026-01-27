import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.mcp_integration.calendar_server import (
    list_events,
    create_event,
    _norm_iso_datetime,
    _nl_window,
    _normalize_bounds
)

@pytest.fixture
def mock_google_auth():
    with patch("src.mcp_integration.calendar_server.get_credentials") as mock_creds:
        mock_creds.return_value = MagicMock()
        yield mock_creds

@pytest.fixture
def mock_calendar_service():
    with patch("src.mcp_integration.calendar_server.build") as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        yield mock_service

def test_norm_iso_datetime():
    # Test date only
    assert _norm_iso_datetime("2024-01-01").endswith("Z")

    # Test full ISO with Z
    assert _norm_iso_datetime("2024-01-01T12:00:00Z") == "2024-01-01T12:00:00Z"

    # Test without timezone (should assume local then convert to Z)
    res = _norm_iso_datetime("2024-01-01T12:00:00")
    assert res.endswith("Z")

def test_nl_window():
    # Basic check for some phrases
    assert _nl_window("today") is not None
    assert _nl_window("tomorrow") is not None
    assert _nl_window("now") is not None
    assert _nl_window("invalid") is None

def test_normalize_bounds():
    tmin, tmax = _normalize_bounds("today", None)
    assert tmin.endswith("Z")
    assert tmax.endswith("Z")

@pytest.mark.asyncio
async def test_list_events(mock_google_auth, mock_calendar_service):
    mock_list_req = MagicMock()
    mock_calendar_service.events().list.return_value = mock_list_req
    mock_list_req.execute.return_value = {
        "items": [
            {
                "id": "ev1",
                "summary": "Meeting",
                "start": {"dateTime": "2024-01-01T10:00:00Z"},
                "end": {"dateTime": "2024-01-01T11:00:00Z"}
            }
        ]
    }

    events = await list_events(max_results=1)
    assert len(events) == 1
    assert events[0]["summary"] == "Meeting"
    assert events[0]["id"] == "ev1"

@pytest.mark.asyncio
async def test_create_event(mock_google_auth, mock_calendar_service):
    mock_insert_req = MagicMock()
    mock_calendar_service.events().insert.return_value = mock_insert_req
    mock_insert_req.execute.return_value = {
        "id": "new_ev",
        "summary": "Lunch",
        "start": {"dateTime": "2024-01-01T12:00:00Z"},
        "end": {"dateTime": "2024-01-01T13:00:00Z"}
    }

    event = await create_event(
        summary="Lunch",
        start_iso="2024-01-01T12:00:00Z",
        end_iso="2024-01-01T13:00:00Z"
    )

    assert event["id"] == "new_ev"
    assert event["summary"] == "Lunch"
    mock_calendar_service.events().insert.assert_called_once()
