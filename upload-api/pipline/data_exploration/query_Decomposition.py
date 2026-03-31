from __future__ import annotations

import re


def decompose_natural_language_query(text: str) -> list[str]:
    """
    Turn a short natural-language goal into a checklist (heuristic; no LLM).
    Useful for UI hints or future planner agents.
    """
    q = (text or "").strip()
    if not q:
        return []
    parts = re.split(r"[.;]\s*|\n+", q)
    steps = [p.strip() for p in parts if p.strip()]
    if not steps:
        steps = [q]
    out: list[str] = []
    for i, s in enumerate(steps, start=1):
        out.append(f"{i}. {s}")
    return out
