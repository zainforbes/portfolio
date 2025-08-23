from typing import Tuple
import httpx
from googleapiclient.errors import HttpError

RETRYABLE = {408, 429, 500, 502, 503, 504}

def classify(e: Exception) -> str:
    s = str(e)
    if "BRAVE_API_KEY" in s or "X-Subscription-Token" in s:
        return "auth_brave"
    if "insufficientPermissions" in s or "Insufficient Permission" in s:
        return "scope_google"
    if isinstance(e, HttpError):
        if e.resp.status in RETRYABLE: return "retryable_http"
        if e.resp.status == 401:       return "auth_google"
        return "http_google"
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        # Treat Brave auth/config errors as auth issues (Brave sometimes uses 422)
        if code in (401, 403, 422):    return "auth_brave"
        if code in RETRYABLE:          return "retryable_http"
        if 400 <= code < 500:          return "http_brave"
        return "http_brave"
    if isinstance(e, (httpx.TimeoutException, TimeoutError)):
        return "timeout"
    return "unknown"

def should_retry(e: Exception, attempt: int) -> Tuple[bool, float]:
    t = classify(e)
    if t in {"retryable_http","timeout"} and attempt < 3:
        return True, min(2**attempt, 8)  # 1s, 2s, 4s
    return False, 0.0

def explain(e: Exception) -> str:
    t = classify(e)
    if t == "auth_brave":
        return ("Brave Search authentication failed (401/403/422). "
                "Check BRAVE_API_KEY in your .env (X-Subscription-Token) and restart the shell.")
    if t == "auth_google":
        return "Google auth failed (401). Delete config/token.json and re-authenticate."
    if t == "scope_google":
        return "Google token lacks required Gmail/Calendar scopes. Delete config/token.json and re-auth with the correct scopes."
    if t == "retryable_http":
        return "Temporary server issue; retried a few times and then stopped."
    if t == "timeout":
        return "Network timeout while calling the service."
    if t == "http_brave":
        return "Brave rejected the request (HTTP 4xx). Try a simpler query and ensure parameters are valid."
    return "Unexpected error; see logs for details."
