from src.core.llm_client import GeminiClient

gemini = GeminiClient()
resp = gemini.chat("Say 'Hello from Gemini!'")
print(resp)
