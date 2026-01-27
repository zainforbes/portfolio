from src.core.langchain_tools import get_all_tools

def test_get_all_tools():
    tools = get_all_tools()
    assert len(tools) > 0

    tool_names = [t.name for t in tools]
    assert "web_search" in tool_names
    assert "gmail_list_recent" in tool_names
    assert "gmail_read" in tool_names
    assert "gmail_send" in tool_names
    assert "gcal_list_events" in tool_names
    assert "gcal_create_event" in tool_names
    assert "gcal_update_event" in tool_names
    assert "gcal_delete_event" in tool_names

def test_tool_descriptions():
    tools = get_all_tools()
    for tool in tools:
        assert tool.description is not None
        assert len(tool.description) > 0
