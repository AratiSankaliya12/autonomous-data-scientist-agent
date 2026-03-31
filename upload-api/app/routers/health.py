from fastapi import APIRouter

from app.config import OPENAI_API_KEY, OPENAI_MODEL

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(OPENAI_API_KEY),
        "llm_model": OPENAI_MODEL if OPENAI_API_KEY else None,
    }
