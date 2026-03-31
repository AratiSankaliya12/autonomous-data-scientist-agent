from fastapi import APIRouter, HTTPException

from app.agents.orchestrator import run_pipeline
from app.config import OPENAI_API_KEY
from app.schemas.agent import AnalyzeRequest, AnalyzeResponse, LLMAnalyzeRequest, LLMAnalyzeResponse

router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/uploads/{file_id}/analyze", response_model=AnalyzeResponse)
def analyze_upload(file_id: str, body: AnalyzeRequest):
    try:
        return run_pipeline(file_id, body)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Upload not found") from None


@router.post("/uploads/{file_id}/analyze-llm", response_model=LLMAnalyzeResponse)
async def analyze_upload_llm(file_id: str, body: LLMAnalyzeRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured. Set OPENAI_API_KEY on the API server.",
        )
    try:
        from agents.llm_orchestrator import run_llm_pipeline

        return await run_llm_pipeline(file_id, body.user_goal)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Upload not found") from None
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "LangGraph is missing or too old (no langgraph.prebuilt). "
                f"Install deps from upload-api/requirements.txt. Import error: {e!s}"
            ),
        ) from None
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LangGraph / OpenAI error: {e!s}",
        ) from None
