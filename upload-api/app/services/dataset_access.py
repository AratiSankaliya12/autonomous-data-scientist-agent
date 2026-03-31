from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.dataframe_loader import read_dataframe
from app.services.upload_store import get_record, resolve_path


def load_dataframe(upload_id: str) -> pd.DataFrame:
    rec = get_record(upload_id)
    if rec is None:
        raise ValueError(f"Upload not found: {upload_id}")
    path = resolve_path(rec["stored_path"])
    suffix = Path(rec.get("original_name", "")).suffix.lower() or path.suffix.lower()
    return read_dataframe(path, suffix)
