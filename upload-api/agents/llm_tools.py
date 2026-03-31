"""
LangChain tools for the autonomous data-scientist agent.

Tools close over `upload_id` so the LLM cannot target the wrong dataset.
"""

from __future__ import annotations

import json
from typing import Callable

from langchain_core.tools import tool

from app.services.dataset_access import load_dataframe
from agents.feature_agent import run as run_feature_agent
from pipline.data_exploration import DataExploration


def _truncate(s: str, limit: int = 14_000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n... (truncated for context)"


def build_ds_tools(upload_id: str) -> list[Callable]:
    @tool
    def describe_dataset() -> str:
        """Column names, dtypes, row/column counts, and missing-value counts per column."""
        df = load_dataframe(upload_id)
        nulls = {
            str(c): int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()
        }
        meta = {
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "columns_with_missing": nulls,
        }
        return _truncate(json.dumps(meta, indent=2, default=str))

    @tool
    def exploration_summary() -> str:
        """Duplicate rows, numeric describe(), and value-count samples for low-cardinality columns."""
        df = load_dataframe(upload_id)
        payload = DataExploration().explore(df)
        return _truncate(json.dumps(payload, indent=2, default=str))

    @tool
    def feature_engineering_snapshot() -> str:
        """Feature roles: numeric vs categorical, high-cardinality columns, encoding suggestions."""
        df = load_dataframe(upload_id)
        summary, payload = run_feature_agent(df)
        blob = {"summary": summary, "payload": payload}
        return _truncate(json.dumps(blob, indent=2, default=str))

    @tool
    def execute_full_pipeline(
        apply_cleaning: bool,
        target_column: str,
        reasoning: str,
    ) -> str:
        """Run the full stack (explore → clean? → features → model? → eval → charts) once.

        Args:
            apply_cleaning: True to dedupe rows and impute missing values before later stages.
            target_column: Exact column name to predict; use an empty string to skip modeling.
            reasoning: Short rationale (stored for the UI audit trail).
        """
        from app.agents.orchestrator import run_pipeline
        from app.schemas.agent import AnalyzeRequest

        tgt = (target_column or "").strip() or None
        body = AnalyzeRequest(
            target_column=tgt,
            steps=None,
            apply_cleaning=apply_cleaning,
        )
        result = run_pipeline(upload_id, body)
        blob = result.model_dump()
        blob["agent_reasoning"] = (reasoning or "").strip()
        return _truncate(json.dumps(blob, indent=2, default=str), limit=200_000)

    return [describe_dataset, exploration_summary, feature_engineering_snapshot, execute_full_pipeline]
