from __future__ import annotations

import numpy as np
import pandas as pd


def suggest_encodings(df: pd.DataFrame) -> list[dict]:
    """
    Heuristic encoding suggestions for non-numeric columns.
    """
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    suggestions: list[dict] = []
    for c in df.columns:
        if c in num:
            continue
        n = int(df[c].nunique(dropna=True))
        col = str(c)
        if n <= 2:
            suggestions.append(
                {"column": col, "method": "binary_or_one_hot", "n_categories": n}
            )
        elif n <= 25:
            suggestions.append(
                {"column": col, "method": "one_hot_or_ordinal", "n_categories": n}
            )
        else:
            suggestions.append(
                {
                    "column": col,
                    "method": "target_or_embedding_or_hash",
                    "n_categories": n,
                }
            )
    return suggestions
