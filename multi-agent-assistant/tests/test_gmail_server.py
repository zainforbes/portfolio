import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.mcp_integration.gmail_server import list_recent_emails, read_email, send_email

@pytest.fixture
def mock_google_auth():
    with patch("src.mcp_integration.gmail_server.get_credentials") as mock_creds:
        mock_creds.return_value = MagicMock()
        yield mock_creds

@pytest.fixture
def mock_gmail_service():
    with patch("src.mcp_integration.gmail_server.build") as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        yield mock_service

@pytest.mark.asyncio
async def test_list_recent_emails(mock_google_auth, mock_gmail_service):
    # Mock messages().list().execute()
    mock_list_req = MagicMock()
    mock_gmail_service.users().messages().list.return_value = mock_list_req
    mock_list_req.execute.return_value = {"messages": [{"id": "123"}]}

    # Mock messages().get().execute()
    mock_get_req = MagicMock()
    mock_gmail_service.users().messages().get.return_value = mock_get_req
    mock_get_req.execute.return_value = {
        "id": "123",
        "snippet": "Test snippet",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@test.com"},
                {"name": "Subject", "value": "Test Subject"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"}
            ]
        }
    }

    emails = await list_recent_emails(max_results=1)

    assert len(emails) == 1
    assert emails[0]["id"] == "123"
    assert emails[0]["from"] == "sender@test.com"
    assert emails[0]["subject"] == "Test Subject"
    assert emails[0]["snippet"] == "Test snippet"

@pytest.mark.asyncio
async def test_read_email(mock_google_auth, mock_gmail_service):
    mock_get_req = MagicMock()
    mock_gmail_service.users().messages().get.return_value = mock_get_req
    mock_get_req.execute.return_value = {"id": "123", "snippet": "Test snippet"}

    email = await read_email("123")

    assert email["id"] == "123"
    assert email["snippet"] == "Test snippet"

@pytest.mark.asyncio
async def test_send_email(mock_google_auth, mock_gmail_service):
    mock_send_req = MagicMock()
    mock_gmail_service.users().messages().send.return_value = mock_send_req
    mock_send_req.execute.return_value = {"id": "sent_123"}

    result = await send_email(to="recipient@test.com", subject="Hello", body="World")

    assert result["id"] == "sent_123"
    assert result["status"] == "sent"
    mock_gmail_service.users().messages().send.assert_called_once()

@pytest.mark.asyncio
async def test_send_email_list(mock_google_auth, mock_gmail_service):
    mock_send_req = MagicMock()
    mock_gmail_service.users().messages().send.return_value = mock_send_req
    mock_send_req.execute.return_value = {"id": "sent_456"}

    result = await send_email(to=["a@test.com", "b@test.com"], subject="Hello", body="World")

    assert result["id"] == "sent_456"
    assert result["status"] == "sent"
