from __future__ import annotations

import pandas as pd

from pipline.data_cleaning.missing_value_handling import (
    fill_missing_simple,
    null_counts_by_column,
)
from pipline.data_cleaning.outlier_removal import count_outliers_iqr


class DataCleaning:
    """Analyze table quality and optionally apply standard fixes."""

    def analyze(self, df: pd.DataFrame) -> dict:
        dups = int(df.duplicated().sum())
        nulls = null_counts_by_column(df)
        outlier_hint = count_outliers_iqr(df)
        outlier_cols = {k: v for k, v in outlier_hint.items() if v > 0}
        reco: list[str] = []
        if dups:
            reco.append(f"Found {dups} duplicate rows; drop_duplicates recommended.")
        if nulls:
            reco.append(
                f"Missing values in {len(nulls)} column(s); imputation or drop recommended."
            )
        if outlier_cols:
            reco.append(
                f"Potential outliers (IQR rule) in {len(outlier_cols)} numeric column(s)."
            )
        if not reco:
            reco.append("No obvious structural issues detected.")
        return {
            "recommendations": reco,
            "null_columns": nulls,
            "duplicate_rows": dups,
            "outlier_counts_iqr": outlier_cols,
        }

    def transform(self, df: pd.DataFrame, *, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
        out = df.copy()
        before = len(out)
        dropped = 0
        if drop_duplicates:
            out = out.drop_duplicates()
            dropped = before - len(out)
        out = fill_missing_simple(out)
        meta = {
            "rows_before": before,
            "rows_after": len(out),
            "dropped_duplicates": dropped,
        }
        return out, meta
