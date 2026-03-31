from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AgentName = Literal[
    "exploration",
    "cleaning",
    "features",
    "modeling",
    "evaluation",
    "visualization",
]


class AnalyzeRequest(BaseModel):
    target_column: str | None = None
    steps: list[AgentName] | None = Field(
        default=None,
        description="Subset of agents to run; default runs all applicable steps.",
    )
    apply_cleaning: bool = False


class AgentStepResult(BaseModel):
    agent: AgentName
    status: Literal["ok", "skipped", "error"]
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AnalyzeResponse(BaseModel):
    upload_id: str
    started_at: str
    finished_at: str
    target_column: str | None
    steps: list[AgentStepResult]


class LLMAnalyzeRequest(BaseModel):
    user_goal: str = Field(
        ...,
        min_length=3,
        max_length=8000,
        description="What the user wants to learn or predict from the dataset.",
    )


class MessageTraceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None


class LLMAnalyzeResponse(BaseModel):
    upload_id: str
    user_goal: str
    model: str
    final_reply: str
    pipeline: AnalyzeResponse | None = None
    trace: list[MessageTraceItem]
