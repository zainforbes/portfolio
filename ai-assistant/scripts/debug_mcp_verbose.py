# scripts/debug_mcp_verbose.py
import os, json, time
import requests

SERVER = os.getenv("BRAVE_MCP_URL", "http://localhost:8080/mcp")

payload = {
    "jsonrpc": "2.0",
    "id": int(time.time() * 1000) % (2**31),
    "method": "tools/call",
    "params": {
        "name": "brave_web_search",
        "arguments": {"query": "openai", "count": 2}
    }
}

headers = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json"
}

print("POST ->", SERVER)
print("HEADERS ->", headers)
print("PAYLOAD ->", json.dumps(payload, indent=2))
print("="*60)
try:
    r = requests.post(SERVER, json=payload, headers=headers, timeout=30, stream=False)
    print("STATUS:", r.status_code)
    print("RESPONSE HEADERS:")
    for k, v in r.headers.items():
        print(f"  {k}: {v}")
    print("-"*60)
    # print full (but safe-limit) response body
    text = r.text
    print("RESPONSE BODY (first 4000 chars):\n")
    print(text[:4000])
    print("\n(END BODY)")
except Exception as e:
    print("REQUEST ERROR:", repr(e))
