from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _is_classification(y: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(y):
        return True
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return y.nunique(dropna=True) <= 25
    if pd.api.types.is_integer_dtype(y) and y.nunique(dropna=True) <= 15:
        return True
    return False


def train_baseline_random_forest(df: pd.DataFrame, target: str) -> tuple[str, dict]:
    """
    Fit a sklearn RandomForest baseline with numeric imputation + one-hot for categoricals.
    Returns (summary_text, metrics_and_artifacts_dict).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe.")

    work = df.dropna(subset=[target]).copy()
    if len(work) < 5:
        raise ValueError("Need at least 5 rows with non-null target for a baseline model.")

    y = work[target]
    X = work.drop(columns=[target])

    if X.shape[1] == 0:
        raise ValueError("No feature columns after removing target.")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers: list[tuple] = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                max_categories=30,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        )
    if not transformers:
        raise ValueError("No usable feature columns after removing target.")

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    classify = _is_classification(y)
    if classify:
        y_enc = pd.factorize(y.astype(str))[0]
        if len(np.unique(y_enc)) < 2:
            raise ValueError("Target has only one class after encoding.")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            random_state=42,
            class_weight="balanced_subsample",
        )
        strat = y_enc if len(np.unique(y_enc)) > 1 else None
    else:
        y_enc = pd.to_numeric(y, errors="coerce")
        if y_enc.isna().all():
            raise ValueError("Target is not numeric for regression.")
        y_enc = y_enc.values
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            random_state=42,
        )
        strat = None

    pipe = Pipeline([("prep", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=strat,
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    payload: dict = {
        "task": "classification" if classify else "regression",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    if classify:
        payload["accuracy"] = float(accuracy_score(y_test, pred))
        payload["f1_macro"] = float(
            f1_score(y_test, pred, average="macro", zero_division=0)
        )
        summary = (
            f"RandomForest classifier — accuracy {payload['accuracy']:.3f}, "
            f"macro-F1 {payload['f1_macro']:.3f}."
        )
    else:
        payload["r2"] = float(r2_score(y_test, pred))
        payload["mae"] = float(mean_absolute_error(y_test, pred))
        summary = (
            f"RandomForest regressor — R² {payload['r2']:.3f}, MAE {payload['mae']:.4f}."
        )

    try:
        importances = pipe.named_steps["model"].feature_importances_
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        top = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:12]
        payload["feature_importance_top"] = [
            {"feature": str(a), "importance": float(b)} for a, b in top
        ]
    except Exception:
        pass

    return summary, payload
