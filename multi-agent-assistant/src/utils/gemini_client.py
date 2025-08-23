# src/utils/gemini_client.py
import os
import time
from typing import Optional, List

import google.generativeai as genai
from dotenv import load_dotenv

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 2):
        # Load .env so GEMINI_API_KEY is available
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create a .env with GEMINI_API_KEY=... "
                "or set it in your environment."
            )

        genai.configure(api_key=api_key)
        self.model_name = model
        self.max_retries = max_retries
        self.model = genai.GenerativeModel(self.model_name)

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[List[dict]] = None,
        temperature: float = 0.7,
        timeout_s: float = 60.0,
    ) -> str:
        """
        Minimal, resilient wrapper around generate_content.
        - `system`: optional system preamble
        - `history`: [{'role':'user'|'model','parts':[str]}] if you keep chat state
        """
        messages = []
        if system:
            messages.append({"role": "user", "parts": [f"System:\n{system}"]})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "parts": [prompt]})

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.model.generate_content(
                    messages,
                    generation_config={"temperature": temperature},
                    request_options={"timeout": timeout_s},
                )
                return getattr(resp, "text", "").strip()
            except Exception as e:
                last_err = e
                # simple backoff for transient issues (rate limits, timeouts)
                if attempt < self.max_retries:
                    time.sleep(0.6 * (attempt + 1))
                else:
                    return f"Error: {type(e).__name__}: {e}"

        return f"Error: {last_err}"  # defensive fallback

