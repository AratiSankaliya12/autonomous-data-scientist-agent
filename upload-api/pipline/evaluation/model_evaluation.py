from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def format_evaluation_from_modeling_payload(
    modeling_payload: dict[str, Any],
) -> tuple[str, dict]:
    task = modeling_payload.get("task")
    payload = {
        "task": task,
        "metrics": {
            k: modeling_payload[k]
            for k in ("accuracy", "f1_macro", "r2", "mae")
            if k in modeling_payload
        },
        "n_train": modeling_payload.get("n_train"),
        "n_test": modeling_payload.get("n_test"),
    }
    if task == "classification":
        summary = (
            "Evaluation: hold-out classification metrics "
            f"(accuracy={payload['metrics'].get('accuracy')}, "
            f"macro-F1={payload['metrics'].get('f1_macro')})."
        )
    elif task == "regression":
        summary = (
            "Evaluation: hold-out regression metrics "
            f"(R²={payload['metrics'].get('r2')}, MAE={payload['metrics'].get('mae')})."
        )
    else:
        summary = "No modeling metrics to summarize."
    return summary, payload
