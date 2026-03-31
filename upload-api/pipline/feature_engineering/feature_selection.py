from __future__ import annotations

import numpy as np
import pandas as pd


def constant_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]


def high_cardinality_columns(df: pd.DataFrame, threshold: int = 50) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    out: list[str] = []
    for c in df.columns:
        if c in num:
            continue
        if df[c].nunique(dropna=True) > threshold:
            out.append(str(c))
    return out


def feature_profile(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in df.columns if c not in num]
    hc = high_cardinality_columns(df)
    return {
        "numeric_columns": [str(c) for c in num],
        "non_numeric_columns": [str(c) for c in cat],
        "high_cardinality_columns": hc[:20],
        "constant_columns": constant_columns(df),
    }
