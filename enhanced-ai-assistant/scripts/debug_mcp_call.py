# scripts/debug_mcp_call.py
import os
from src.mcp_integration.mcp_client import MCPClient, MCPError

def main():
    url = os.getenv("BRAVE_MCP_URL", "http://localhost:8080/mcp")
    client = MCPClient(server_url=url, timeout=30)
    try:
        res = client.call_tool("brave_web_search", {"query": "openai", "count": 3})
        print("NORMALIZED RESULT:")
        import pprint; pprint.pprint(res)
    except MCPError as e:
        print("MCPError:", e)
    except Exception as e:
        print("Unhandled error:", e)

if __name__ == "__main__":
    main()
