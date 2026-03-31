from __future__ import annotations

import pandas as pd


def dtype_breakdown(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in df.columns:
        k = str(df[col].dtype)
        counts[k] = counts.get(k, 0) + 1
    return counts


def column_profiles(df: pd.DataFrame, max_uniques: int = 12) -> list[dict]:
    rows: list[dict] = []
    for c in df.columns:
        s = df[c]
        rows.append(
            {
                "column": str(c),
                "dtype": str(s.dtype),
                "null_count": int(s.isna().sum()),
                "n_unique": int(s.nunique(dropna=True)),
                "sample_values": s.dropna().astype(str).head(max_uniques).tolist(),
            }
        )
    return rows
