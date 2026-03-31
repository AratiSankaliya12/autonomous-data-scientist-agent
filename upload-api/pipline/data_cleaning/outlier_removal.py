from __future__ import annotations

import numpy as np
import pandas as pd


def count_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> dict[str, int]:
    """Count rows that fall outside [Q1 - k*IQR, Q3 + k*IQR] per numeric column."""
    num = df.select_dtypes(include=[np.number])
    cols = [c for c in (columns or num.columns.tolist()) if c in num.columns]
    counts: dict[str, int] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 4:
            counts[str(c)] = 0
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        mask = (df[c] < low) | (df[c] > high)
        counts[str(c)] = int(mask.fillna(False).sum())
    return counts


def clip_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    out = df.copy()
    num = out.select_dtypes(include=[np.number])
    cols = [c for c in (columns or num.columns.tolist()) if c in num.columns]
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().sum() < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        out[c] = s.clip(lower=low, upper=high)
    return out
