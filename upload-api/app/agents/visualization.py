"""Visualization step — delegates to `agents.visualization_agent`."""

from __future__ import annotations

import pandas as pd

from agents.visualization_agent import run as _run


def run_visualization(df: pd.DataFrame) -> tuple[str, dict]:
    return _run(df)
