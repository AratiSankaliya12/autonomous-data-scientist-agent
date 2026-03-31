from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AgentContext:
    """Shared context for future LLM or tool-using agents."""

    upload_id: str
    dataframe: pd.DataFrame
    target_column: str | None = None
    extras: dict[str, Any] | None = None
