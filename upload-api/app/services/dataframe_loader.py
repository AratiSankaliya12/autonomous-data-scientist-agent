import json
from pathlib import Path

import pandas as pd


def preview_dataframe(df: pd.DataFrame, limit: int = 8) -> dict:
    nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    head = df.head(limit)
    sample_rows = json.loads(head.to_json(orient="records", date_format="iso"))
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
        "dtypes": dtypes,
        "null_counts": nulls,
        "sample_rows": sample_rows,
    }


def read_dataframe(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported suffix: {suffix}")


def try_preview(path: Path, suffix: str) -> dict | None:
    try:
        df = read_dataframe(path, suffix)
    except Exception:
        return None
    return preview_dataframe(df)
