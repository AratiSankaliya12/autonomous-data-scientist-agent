"""Visualization agent — wraps `pipline.visualization`."""

from __future__ import annotations

import pandas as pd

from pipline.visualization import build_chart_specs


def run(df: pd.DataFrame) -> tuple[str, dict]:
    charts = build_chart_specs(df)
    summary = f"Prepared {len(charts)} chart dataset(s) for the UI."
    return summary, {"charts": charts}
