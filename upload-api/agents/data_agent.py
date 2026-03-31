"""Data exploration agent — wraps `pipline.data_exploration`."""

from __future__ import annotations

import pandas as pd

from pipline.data_exploration import DataExploration


def run(df: pd.DataFrame) -> tuple[str, dict]:
    payload = DataExploration().explore(df)
    summary = (
        f"{payload['row_count']} rows, {payload['column_count']} columns; "
        f"{payload['duplicate_rows']} duplicate rows."
    )
    return summary, payload
