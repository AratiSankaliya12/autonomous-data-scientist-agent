from __future__ import annotations

import numpy as np
import pandas as pd


def build_chart_specs(df: pd.DataFrame) -> list[dict]:
    charts: list[dict] = []

    nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}
    null_items = sorted(
        ((k, v) for k, v in nulls.items() if v > 0),
        key=lambda x: -x[1],
    )[:15]
    if null_items:
        charts.append(
            {
                "id": "missing_counts",
                "title": "Missing values by column",
                "type": "bar",
                "labels": [k for k, _ in null_items],
                "values": [v for _, v in null_items],
            }
        )

    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        col = numeric.columns[0]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0:
            hist, edges = np.histogram(s.values, bins=min(20, max(5, len(s) // 10)))
            charts.append(
                {
                    "id": f"histogram_{col}",
                    "title": f"Distribution of {col}",
                    "type": "histogram",
                    "bin_edges": [float(x) for x in edges],
                    "counts": [int(x) for x in hist],
                }
            )

    return charts
