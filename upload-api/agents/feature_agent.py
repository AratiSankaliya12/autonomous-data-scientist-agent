"""Feature engineering agent — wraps `pipline.feature_engineering`."""

from __future__ import annotations

import pandas as pd

from pipline.feature_engineering import feature_profile, suggest_encodings


def run(df: pd.DataFrame) -> tuple[str, dict]:
    profile = feature_profile(df)
    enc = suggest_encodings(df)
    payload = {
        **profile,
        "encoding_suggestions": enc[:40],
    }
    summary = (
        f"{len(profile['numeric_columns'])} numeric, "
        f"{len(profile['non_numeric_columns'])} non-numeric features; "
        f"{len(profile['high_cardinality_columns'])} high-cardinality categorical columns."
    )
    return summary, payload
