"""Evaluation agent — wraps `pipline.evaluation`."""

from __future__ import annotations

from typing import Any

from pipline.evaluation import format_evaluation_from_modeling_payload


def run(modeling_payload: dict[str, Any]) -> tuple[str, dict]:
    return format_evaluation_from_modeling_payload(modeling_payload)
