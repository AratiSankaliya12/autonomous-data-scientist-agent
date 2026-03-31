"""Exploration step — delegates to `agents.data_agent`."""

from __future__ import annotations

import pandas as pd

from agents.data_agent import run as _run


def run_exploration(df: pd.DataFrame) -> tuple[str, dict]:
    return _run(df)
