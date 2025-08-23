# router.py
import json
import re
from typing import Dict, Any, Tuple
from src.utils.gemini_client import GeminiClient

_SYS = ("Classify the user's request. Output ONLY JSON: "
        '{"agent":"email|calendar|search|default","confidence":0..1,'
        '"filters":{"sender":"","unread":false,"newer_than_days":0,"subject":"","query_extra":""}}')

def _fallback(text: str) -> Tuple[str, float]:
    t = (text or "").lower()
    if any(k in t for k in ("email","gmail","inbox","message","mail")): return "email", 0.8
    if any(k in t for k in ("calendar","meeting","event","schedule")):  return "calendar", 0.8
    if any(k in t for k in ("search","look up","news","web","find")):   return "search", 0.7
    return "default", 0.5

def _extract_json(s: str) -> str:
    s = (s or "").strip()
    # remove code fences ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I|re.S)
    # grab first {...} block if extra text exists
    start = s.find("{")
    end = s.rfind("}")
    return s[start:end+1] if start != -1 and end != -1 else s

def classify_request(text: str, gemini: GeminiClient) -> Dict[str, Any]:
    prompt = (f"{_SYS}\n\nUser:\n{text}\n\nReturn strictly JSON. Example:\n"
              '{"agent":"email","confidence":0.85,'
              '"filters":{"sender":"alice@example.com","unread":true,"newer_than_days":7,"subject":"status","query_extra":""}}')
    out = gemini.chat(prompt)
    try:
        data = json.loads(_extract_json(out))
        agent = (data.get("agent") or "default").strip()
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))  # clamp
        filters = data.get("filters") or {}
        # sanity: ensure expected keys exist
        filters.setdefault("sender", "")
        filters.setdefault("unread", False)
        filters.setdefault("newer_than_days", 0)
        filters.setdefault("subject", "")
        filters.setdefault("query_extra", "")
        return {"agent": agent, "confidence": conf, "filters": filters}
    except Exception:
        a, c = _fallback(text)
        return {"agent": a, "confidence": c, "filters": {}}
