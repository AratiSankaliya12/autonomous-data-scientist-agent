"""Data cleaning agent — wraps `pipline.data_cleaning`."""

from __future__ import annotations

import pandas as pd

from pipline.data_cleaning import DataCleaning


def run(df: pd.DataFrame, *, apply: bool) -> tuple[pd.DataFrame, str, dict]:
    dc = DataCleaning()
    analyzed = dc.analyze(df)
    payload: dict = {
        "recommendations": analyzed["recommendations"],
        "null_columns": analyzed["null_columns"],
        "duplicate_rows": analyzed["duplicate_rows"],
        "outlier_counts_iqr": analyzed["outlier_counts_iqr"],
        "applied": apply,
    }
    before_rows = len(df)
    if apply:
        cleaned, meta = dc.transform(df, drop_duplicates=True)
        payload["rows_after_clean"] = meta["rows_after"]
        payload["dropped_duplicates"] = meta["dropped_duplicates"]
        summary = (
            f"Applied cleaning: {before_rows} → {len(cleaned)} rows; "
            f"removed {meta['dropped_duplicates']} duplicate row(s) and imputed missing values."
        )
        return cleaned, summary, payload

    n_null_cols = len(analyzed["null_columns"])
    summary = (
        f"Analysis only: {analyzed['duplicate_rows']} duplicate rows; "
        f"{n_null_cols} columns with nulls; "
        f"IQR outliers flagged in {len(analyzed['outlier_counts_iqr'])} numeric column(s). "
        "Set apply_cleaning=true to transform."
    )
    return df.copy(), summary, payload
