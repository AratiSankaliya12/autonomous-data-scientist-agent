"""Cleaning step — delegates to `agents.cleaning_agent`."""

from __future__ import annotations

import pandas as pd

from agents.cleaning_agent import run as _run


def run_cleaning(df: pd.DataFrame, apply: bool) -> tuple[pd.DataFrame, str, dict]:
    return _run(df, apply=apply)
