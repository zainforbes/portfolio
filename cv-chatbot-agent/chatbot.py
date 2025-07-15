import requests
from utils import get_api_key

class CVChatbot:
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL   = "llama3-8b-8192"

    def __init__(self):
        self.api_key = get_api_key()
        self.cv_text = ""

    def test_connection(self) -> bool:
        """
        Ping Groq API to verify credentials.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=5)
        return resp.status_code == 200

    def set_cv(self, text: str):
        """Store the full CV text (extracted from PDF)."""
        self.cv_text = text.strip()

    def generate_response(self, question: str) -> str:
        """Send the user’s question + CV to Groq and return the answer."""
        if not self.cv_text:
            return "Please upload your CV first."

        prompt = (
            "You are an AI assistant for interview prep. "
            "Use the following CV content to answer professionally and concisely.\n\n"
            f"CV CONTENT:\n{self.cv_text}\n\n"
            f"QUESTION: {question}\n\n"
            "If the CV has no relevant info, respond generally but professionally."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        data = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        resp = requests.post(self.API_URL, headers=headers, json=data, timeout=30)
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Error {resp.status_code}: {resp.text}"
