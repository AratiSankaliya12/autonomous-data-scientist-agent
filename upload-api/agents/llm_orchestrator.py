"""High-level entry: LangGraph agent + pydantic validation of pipeline output."""

from __future__ import annotations

from app.config import LANGGRAPH_RECURSION_LIMIT, OPENAI_API_KEY, OPENAI_MODEL
from app.schemas.agent import AnalyzeResponse, LLMAnalyzeResponse, MessageTraceItem
from app.services.upload_store import get_record


async def run_llm_pipeline(upload_id: str, user_goal: str) -> LLMAnalyzeResponse:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    if get_record(upload_id) is None:
        raise FileNotFoundError(upload_id)

    # Deferred import so FastAPI can boot even if LangGraph is missing/broken in the env.
    from agents.langgraph_runner import run_langgraph_agent

    raw = await run_langgraph_agent(
        upload_id=upload_id,
        user_goal=user_goal,
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        recursion_limit=LANGGRAPH_RECURSION_LIMIT,
    )

    trace_items = [MessageTraceItem.model_validate(row) for row in raw["trace"]]

    pipeline: AnalyzeResponse | None = None
    blob = raw.get("pipeline_blob")
    if isinstance(blob, dict):
        clean = {k: v for k, v in blob.items() if k != "agent_reasoning"}
        try:
            pipeline = AnalyzeResponse.model_validate(clean)
        except Exception:
            pipeline = None

    return LLMAnalyzeResponse(
        upload_id=upload_id,
        user_goal=user_goal.strip(),
        model=OPENAI_MODEL,
        final_reply=raw.get("final_reply") or "",
        pipeline=pipeline,
        trace=trace_items,
    )
