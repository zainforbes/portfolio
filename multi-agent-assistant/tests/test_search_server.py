import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from src.mcp_integration.search_server import web_search, BRAVE_API_URL, _CACHE

@pytest.fixture(autouse=True)
def clear_cache():
    _CACHE.clear()

@pytest.mark.asyncio
async def test_web_search_success(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "fake_key")

    mock_response_data = {
        "web": {
            "results": [
                {
                    "title": "Test Title",
                    "url": "https://test.com",
                    "description": "Test Description",
                    "meta_url": {"hostname": "test.com"}
                }
            ]
        }
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        results = await web_search("success query", count=1)

        assert len(results) == 1
        assert results[0]["title"] == "Test Title"
        assert results[0]["url"] == "https://test.com"
        assert results[0]["snippet"] == "Test Description"
        assert results[0]["source"] == "test.com"

@pytest.mark.asyncio
async def test_web_search_no_results(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "fake_key")

    mock_response_data = {"web": {"results": []}}
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        results = await web_search("no results query")
        assert len(results) == 0

@pytest.mark.asyncio
async def test_web_search_api_error(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "fake_key")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_response)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            await web_search("error query")

@pytest.mark.asyncio
async def test_web_search_not_configured(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("X_SUBSCRIPTION_TOKEN", raising=False)

    from src.mcp_integration.search_server import NotConfigured
    with pytest.raises(NotConfigured):
        await web_search("not configured query")
