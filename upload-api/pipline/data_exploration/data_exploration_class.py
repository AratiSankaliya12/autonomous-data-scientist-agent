from __future__ import annotations

import json

import numpy as np
import pandas as pd


class DataExploration:
    """Compute dataset-level statistics and lightweight categorical samples."""

    def explore(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include=[np.number])
        payload: dict = {
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": list(df.columns.astype(str)),
        }
        if len(numeric.columns) > 0:
            desc = numeric.describe()
            payload["numeric_describe"] = json.loads(desc.to_json())
        cat_cols = [
            c
            for c in df.columns
            if c not in numeric.columns and df[c].nunique(dropna=True) <= 20
        ]
        payload["low_cardinality_columns_sample"] = {
            str(c): df[c].value_counts(dropna=True).head(10).to_dict()
            for c in cat_cols[:5]
        }
        return payload
