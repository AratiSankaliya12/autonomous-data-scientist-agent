"""
LangGraph ReAct agent (LangChain OpenAI chat model + tool loop).

`langgraph.prebuilt` exists only in recent `langgraph` releases; we lazy-load and try
fallback import paths so the rest of the API can start without a full LLM stack.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agents.llm_tools import build_ds_tools

SYSTEM_PROMPT = """You are an autonomous senior data scientist working inside an analytics product.

You have read-only inspection tools and one execution tool:
- Use describe_dataset, exploration_summary, and feature_engineering_snapshot to understand the table.
- When you are ready, call execute_full_pipeline exactly ONCE with your best choices.
- Set apply_cleaning=true if you see duplicate rows or meaningful missingness; otherwise false.
- Set target_column to the exact column name the user wants to predict, or to empty string \"\" if unclear or not a prediction task.
- Always include a short reasoning string explaining your decisions.

After execute_full_pipeline returns, write a concise, non-technical summary of findings and next steps for the user.
Do not call execute_full_pipeline more than once unless a tool error explicitly asks you to retry."""


def _load_create_react_agent() -> Callable[..., Any]:
    """Resolve create_react_agent across LangGraph versions."""
    attempts: list[str] = []
    for mod_path, attr in (
        ("langgraph.prebuilt", "create_react_agent"),
        ("langgraph.prebuilt.chat_agent_executor", "create_react_agent"),
    ):
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
            attempts.append(f"{mod_path}.{attr}: missing or not callable")
        except ImportError as e:
            attempts.append(f"{mod_path}: {e!s}")
    raise ImportError(
        "Could not import create_react_agent from LangGraph. Your environment likely has an "
        "outdated or incomplete `langgraph` install (missing `langgraph.prebuilt`). "
        "Fix: pip uninstall -y langgraph langgraph-checkpoint langgraph-sdk; "
        'pip install "langgraph>=0.2.55" "langchain-openai>=0.2.0" '
        '"langgraph-checkpoint>=2.0.0". '
        f"Attempts: {'; '.join(attempts)}"
    )


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            trace.append({"role": "system", "content": _clip(str(m.content))})
        elif isinstance(m, HumanMessage):
            trace.append({"role": "user", "content": _clip(str(m.content))})
        elif isinstance(m, AIMessage):
            tool_calls = None
            if getattr(m, "tool_calls", None):
                tool_calls = []
                for tc in m.tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(
                            {
                                "name": tc.get("name"),
                                "args": tc.get("args") or tc.get("arguments"),
                            }
                        )
                    else:
                        tool_calls.append(
                            {
                                "name": getattr(tc, "name", None),
                                "args": getattr(tc, "args", None),
                            }
                        )
            trace.append(
                {
                    "role": "assistant",
                    "content": _clip(str(m.content or "")),
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(m, ToolMessage):
            trace.append(
                {
                    "role": "tool",
                    "tool_name": getattr(m, "name", "") or "",
                    "content": _clip(str(m.content)),
                }
            )
        else:
            trace.append({"role": type(m).__name__, "content": _clip(str(m))})
    return trace


def _clip(s: str, n: int = 12_000) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "\n... (truncated)"


def _final_assistant_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            if getattr(m, "tool_calls", None):
                continue
            return str(m.content).strip()
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            return str(m.content).strip()
    return ""


def _extract_pipeline_blob(messages: list[Any]) -> dict[str, Any] | None:
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = json.loads(str(m.content))
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, dict)
            and "steps" in data
            and "upload_id" in data
            and isinstance(data.get("steps"), list)
        ):
            return data
    return None


def build_graph(model: str, api_key: str | None, upload_id: str):
    create_react_agent = _load_create_react_agent()
    tools = build_ds_tools(upload_id)
    llm = ChatOpenAI(
        model=model,
        temperature=0.15,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )
    return create_react_agent(llm, tools)


async def run_langgraph_agent(
    *,
    upload_id: str,
    user_goal: str,
    model: str,
    api_key: str | None,
    recursion_limit: int,
) -> dict[str, Any]:
    graph = build_graph(model, api_key, upload_id)
    user_block = (
        f"Dataset session id: {upload_id}\n"
        f"User goal / question:\n{user_goal.strip()}\n"
        "Inspect the dataset with tools, then run execute_full_pipeline once, then summarize."
    )
    state = await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_block),
            ]
        },
        config={"recursion_limit": recursion_limit},
    )
    messages = state.get("messages", [])
    return {
        "messages": messages,
        "trace": _serialize_messages(messages),
        "final_reply": _final_assistant_text(messages),
        "pipeline_blob": _extract_pipeline_blob(messages),
    }
