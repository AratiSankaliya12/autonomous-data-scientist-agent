"""Modeling step — delegates to `agents.model_agent`."""

from __future__ import annotations

import pandas as pd

from agents.model_agent import run as _run


def run_modeling(df: pd.DataFrame, target: str) -> tuple[str, dict]:
    return _run(df, target)
