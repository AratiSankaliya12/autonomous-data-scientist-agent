"""Evaluation step — delegates to `agents.evaluation_agent`."""

from __future__ import annotations

from typing import Any

from agents.evaluation_agent import run as _run


def run_evaluation(modeling_payload: dict[str, Any]) -> tuple[str, dict]:
    return _run(modeling_payload)
