# src/utils/gemini_client.py
import os
import json
import re
import time
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger("gemini_client")
logger.setLevel(logging.INFO)


class GeminiClient:
    """
    Gemini client wrapper using google.generativeai.GenerativeModel.

    This version is compatible with SDK variants where generate_content()
    accepts only positional or the single prompt argument (no temperature/max_output_tokens kwargs).
    """
    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 2, backoff_base: float = 0.6):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in .env")
        genai.configure(api_key=api_key)

        # instantiate model handle (retain your previous usage pattern)
        try:
            self.model = genai.GenerativeModel(model)
        except Exception:
            # If construction fails for some SDK versions, keep self.model = None
            self.model = None

        self.model_name = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.last_raw: Optional[str] = None

        logger.info("GeminiClient initialized (model=%s)", model)

    # -------------------------
    # Helpers for prompt + JSON extraction
    # -------------------------
    def _build_classify_prompt(self, user_text: str) -> str:
        examples = [
            {"text": "Add task: buy milk", "agent": "task", "confidence": 0.98},
            {"text": "Schedule meeting with Alice next Tuesday", "agent": "calendar", "confidence": 0.97},
            {"text": "Reply to John's email and attach the report", "agent": "email", "confidence": 0.96},
            {"text": "Research competitor pricing, create brief and schedule meeting", "agent": "coordinator", "confidence": 0.75},
        ]
        example_text = ""
        for ex in examples:
            example_text += f"User: {ex['text']}\nJSON: {{\"agent\": \"{ex['agent']}\", \"confidence\": {ex['confidence']}}}\n\n"

        instruction = (
            "You are a routing assistant. Given the user's request, return a JSON object with exactly two keys:\n"
            "  - \"agent\": one of the following literal strings: \"email\", \"calendar\", \"task\", \"coordinator\"\n"
            "  - \"confidence\": a number between 0.0 and 1.0 representing classifier confidence\n\n"
            "Return ONLY the JSON object and nothing else. Do not add commentary or code fences.\n"
            "If the user text contains multiple distinct actions (e.g., both schedule and email), choose \"coordinator\".\n\n"
            "Examples:\n\n"
            f"{example_text}\n"
            f"User: {user_text}\nJSON:"
        )
        return instruction

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        self.last_raw = text

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return None

        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except Exception:
            sanitized = re.sub(r"'", '"', json_str)
            sanitized = re.sub(r",\s*}", "}", sanitized)
            sanitized = re.sub(r",\s*]", "]", sanitized)
            try:
                return json.loads(sanitized)
            except Exception:
                return None

    # -------------------------
    # Public: classify()
    # -------------------------
    def classify(self, user_text: str) -> Dict[str, Any]:
        prompt = self._build_classify_prompt(user_text)

        attempt = 0
        last_exc = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                # Use the simplest, SDK-compatible call: pass prompt as positional argument
                if self.model is not None:
                    # many SDK versions accept a single positional argument
                    try:
                        resp = self.model.generate_content(prompt)
                    except TypeError:
                        # If the SDK expects keyword name 'prompt'
                        resp = self.model.generate_content(prompt=prompt)
                else:
                    # fallback to top-level genai.generate_content using minimal args
                    try:
                        resp = genai.generate_content(prompt=prompt, model=self.model_name)
                    except TypeError:
                        resp = genai.generate_content(prompt)

                raw = getattr(resp, "text", None) or str(resp)
                raw = raw.strip()
                self.last_raw = raw

                parsed = self._extract_json_from_text(raw)
                if parsed is None:
                    raise ValueError(f"Could not parse JSON from model output: {raw!r}")

                agent = parsed.get("agent")
                if not agent or not isinstance(agent, str):
                    raise ValueError(f"Invalid or missing 'agent' in parsed JSON: {parsed!r}")
                agent = agent.strip().lower()

                if agent not in {"email", "calendar", "task", "coordinator"}:
                    if "mail" in agent or "email" in agent:
                        agent = "email"
                    elif "cal" in agent or "meet" in agent or "schedule" in agent:
                        agent = "calendar"
                    elif "task" in agent or "todo" in agent or "remind" in agent:
                        agent = "task"
                    else:
                        agent = "coordinator"

                raw_conf = parsed.get("confidence", None)
                try:
                    confidence = float(raw_conf) if raw_conf is not None else 0.6
                except Exception:
                    confidence = 0.6

                return {"agent": agent, "confidence": confidence, "raw": raw, "reason": "model_success"}

            except Exception as e:
                last_exc = e
                logger.warning("Gemini classify attempt %d failed: %s", attempt, e)
                sleep_t = self.backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep_t)
                continue

        logger.error("Gemini classify failed after %d attempts: %s", attempt - 1, last_exc)
        return {"agent": "coordinator", "confidence": 0.5, "raw": getattr(self, "last_raw", None), "reason": f"classify_failed:{last_exc}"}

    # -------------------------
    # Public: chat()
    # -------------------------
    def chat(self, prompt: str) -> str:
        try:
            if self.model is not None:
                try:
                    resp = self.model.generate_content(prompt)
                except TypeError:
                    resp = self.model.generate_content(prompt=prompt)
            else:
                try:
                    resp = genai.generate_content(prompt=prompt, model=self.model_name)
                except TypeError:
                    resp = genai.generate_content(prompt)
            return getattr(resp, "text", str(resp))
        except Exception as e:
            logger.exception("Gemini chat failed: %s", e)
            return f"❌ Gemini error: {e}"
