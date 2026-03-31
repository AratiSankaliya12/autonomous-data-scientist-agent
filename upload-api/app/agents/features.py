"""Feature profiling step — delegates to `agents.feature_agent`."""

from __future__ import annotations

import pandas as pd

from agents.feature_agent import run as _run


def run_features(df: pd.DataFrame) -> tuple[str, dict]:
    return _run(df)
