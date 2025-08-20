from src.core.brave_client import BraveSearchClient

brave = BraveSearchClient()
results = brave.search("LangGraph MCP Gemini", 3)

if not results:
    print("No results found.")
else:
    print("Top results:")
    for r in results:
        print(f"- {r['title']} ({r['url']})")
