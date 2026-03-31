from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def quick_tune_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: Literal["classification", "regression"],
    n_iter: int = 12,
    cv: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Lightweight randomized search over a small hyperparameter space.
    Intended for optional / offline use; the default API uses fixed params.
    """
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
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    if task == "classification":
        model = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        param_dist = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [6, 12, 20, None],
            "model__min_samples_leaf": [1, 2, 4],
        }
        scoring = "f1_macro"
    else:
        model = RandomForestRegressor(random_state=random_state)
        param_dist = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [6, 12, 20, None],
            "model__min_samples_leaf": [1, 2, 4],
        }
        scoring = "r2"

    pipe = Pipeline([("prep", preprocess), ("model", model)])
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(n_iter, 24),
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)

    best = search.best_estimator_
    prep = best.named_steps["prep"]
    Xt = prep.transform(X)

    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "scoring": scoring,
        "transformed_shape": getattr(Xt, "shape", None),
    }
