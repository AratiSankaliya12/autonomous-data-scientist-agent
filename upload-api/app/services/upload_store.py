import json
from pathlib import Path

from app.config import INDEX_PATH, UPLOAD_DIR


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def save_index(items: list[dict]) -> None:
    INDEX_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def get_record(file_id: str) -> dict | None:
    for x in load_index():
        if x.get("id") == file_id:
            return x
    return None


def resolve_path(stored_name: str) -> Path:
    return UPLOAD_DIR / stored_name
