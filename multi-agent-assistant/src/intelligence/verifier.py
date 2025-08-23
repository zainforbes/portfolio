from __future__ import annotations
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse

def _domain(u: str) -> str:
    try:
        return urlparse(u).hostname or ""
    except Exception:
        return ""

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _search_confidence(payload: Dict[str, Any]) -> Tuple[float, List[str]]:
    items = payload.get("items") or []
    if not items:
        return 0.35, ["no results"]
    # diversity by domain
    doms = []
    for it in items[:5]:
        doms.append(it.get("source") or _domain(it.get("url", "")))
    uniq = len(set([d for d in doms if d]))
    score = 0.55
    score += min(0.2, 0.05 * len(items))       # more items -> higher
    score += min(0.25, 0.10 * max(0, uniq-1))  # source diversity
    issues: List[str] = []
    if uniq <= 1:
        issues.append("low source diversity")
    return _clamp(score), issues

def _email_confidence(payload: Dict[str, Any]) -> Tuple[float, List[str]]:
    count = int(payload.get("count") or 0)
    q = (payload.get("query") or "").strip()
    score = 0.45
    if count > 0: score += 0.25
    if q:         score += 0.15
    issues: List[str] = []
    if count == 0:
        issues.append("no matching emails")
    return _clamp(score), issues

def _calendar_confidence(payload: Dict[str, Any]) -> Tuple[float, List[str]]:
    items = payload.get("items") or []
    issues: List[str] = []
    score = 0.5 + min(0.3, 0.05 * len(items))
    # slight penalty if conflicts exist
    conflicts = payload.get("conflicts") or []
    if conflicts:
        score -= 0.05
    return _clamp(score), issues

def verify_response(agent: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if agent == "search":
        s, issues = _search_confidence(payload)
    elif agent == "email":
        s, issues = _email_confidence(payload)
    elif agent == "calendar":
        s, issues = _calendar_confidence(payload)
    else:
        s, issues = 0.6, []  # default
    return {"confidence": round(float(s), 2), "issues": issues}
