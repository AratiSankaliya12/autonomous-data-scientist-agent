from __future__ import annotations

import pandas as pd


def null_counts_by_column(df: pd.DataFrame) -> dict[str, int]:
    return {
        str(c): int(df[c].isna().sum())
        for c in df.columns
        if df[c].isna().any()
    }


def fill_missing_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Median for numeric columns; mode (first) for others. Does not drop duplicates."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            mode = out[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) > 0 else None
            out[col] = out[col].fillna(fill)
    return out
