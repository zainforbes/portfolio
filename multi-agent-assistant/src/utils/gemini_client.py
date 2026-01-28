# src/utils/gemini_client.py
import os, json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def _strip_fences(s: str) -> str:
    if not s: return s
    s = s.strip()
    if s.startswith("```"): s = "\n".join(s.splitlines()[1:])
    if s.endswith("```"):   s = "\n".join(s.splitlines()[:-1])
    return s.strip()

class GeminiClient:
    def __init__(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
        self.model_json = genai.GenerativeModel(
            "gemini-2.0-flash-lite",
            generation_config={"response_mime_type": "application/json"}
        )

    def chat(self, prompt: str) -> str:
        try:
            r = self.model.generate_content(prompt)
            return (r.text or "").strip()
        except Exception as e:
            return f"Error: {e}"

    def chat_json_obj(self, prompt: str):
        try:
            r = self.model_json.generate_content(prompt)
            raw = _strip_fences(r.text or "")
            return json.loads(raw)
        except Exception:
            try:
                r2 = self.model_json.generate_content("Return ONLY minified JSON for:\n" + prompt)
                return json.loads(_strip_fences(r2.text or ""))
            except Exception:
                return None
