"""Model training agent — wraps `pipline.model_training`."""

from __future__ import annotations

import pandas as pd

from pipline.model_training import train_baseline_random_forest


def run(df: pd.DataFrame, target: str) -> tuple[str, dict]:
    return train_baseline_random_forest(df, target)
