# scripts/debug_classify.py
from src.utils.gemini_client import GeminiClient

g = GeminiClient()
res = g.classify("Add task: buy groceries")
print("CLASSIFY RESULT:", res)
print("RAW OUTPUT:", g.last_raw)
