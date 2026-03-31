import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 50 * 1024 * 1024
ALLOWED_EXT = frozenset({".csv", ".xlsx", ".xls", ".json", ".parquet"})
INDEX_PATH = UPLOAD_DIR / "index.json"

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]
# Preview / production frontends on Vercel (`*.vercel.app`); override with `CORS_ORIGIN_REGEX=""` to disable.
CORS_ORIGIN_REGEX = os.environ.get(
    "CORS_ORIGIN_REGEX",
    r"https://.*\.vercel\.app",
).strip() or None

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
LANGGRAPH_RECURSION_LIMIT = int(os.environ.get("LANGGRAPH_RECURSION_LIMIT", "40"))
