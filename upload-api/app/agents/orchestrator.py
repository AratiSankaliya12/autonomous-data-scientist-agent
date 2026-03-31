from datetime import datetime, timezone
from pathlib import Path

from app.agents.cleaning import run_cleaning
from app.agents.evaluation import run_evaluation
from app.agents.exploration import run_exploration
from app.agents.features import run_features
from app.agents.modeling import run_modeling
from app.agents.visualization import run_visualization
from app.schemas.agent import AgentName, AgentStepResult, AnalyzeRequest, AnalyzeResponse
from app.services.dataframe_loader import read_dataframe
from app.services.upload_store import get_record, resolve_path

def _should_run(name: AgentName, requested: list[AgentName] | None) -> bool:
    if requested is None:
        return True
    return name in requested


def run_pipeline(upload_id: str, body: AnalyzeRequest) -> AnalyzeResponse:
    started = datetime.now(timezone.utc).isoformat()
    rec = get_record(upload_id)
    if rec is None:
        raise FileNotFoundError(upload_id)

    path = resolve_path(rec["stored_path"])
    suffix = Path(rec.get("original_name", "")).suffix.lower() or path.suffix.lower()
    df = read_dataframe(path, suffix)

    steps_out: list[AgentStepResult] = []
    modeling_payload: dict | None = None
    current_df = df.copy()
    req = body.steps

    if _should_run("exploration", req):
        try:
            summary, payload = run_exploration(current_df)
            steps_out.append(
                AgentStepResult(
                    agent="exploration",
                    status="ok",
                    summary=summary,
                    payload=payload,
                )
            )
        except Exception as e:
            steps_out.append(
                AgentStepResult(
                    agent="exploration",
                    status="error",
                    summary="Exploration failed.",
                    error=str(e),
                )
            )

    if _should_run("cleaning", req):
        try:
            current_df, summary, payload = run_cleaning(
                current_df,
                apply=body.apply_cleaning,
            )
            steps_out.append(
                AgentStepResult(
                    agent="cleaning",
                    status="ok",
                    summary=summary,
                    payload=payload,
                )
            )
        except Exception as e:
            steps_out.append(
                AgentStepResult(
                    agent="cleaning",
                    status="error",
                    summary="Cleaning failed.",
                    error=str(e),
                )
            )

    if _should_run("features", req):
        try:
            summary, payload = run_features(current_df)
            steps_out.append(
                AgentStepResult(
                    agent="features",
                    status="ok",
                    summary=summary,
                    payload=payload,
                )
            )
        except Exception as e:
            steps_out.append(
                AgentStepResult(
                    agent="features",
                    status="error",
                    summary="Feature analysis failed.",
                    error=str(e),
                )
            )

    target = (body.target_column or "").strip() or None

    if _should_run("modeling", req):
        if not target:
            steps_out.append(
                AgentStepResult(
                    agent="modeling",
                    status="skipped",
                    summary="No target_column provided; skipping baseline model.",
                    payload={},
                )
            )
        else:
            try:
                summary, payload = run_modeling(current_df, target)
                modeling_payload = payload
                steps_out.append(
                    AgentStepResult(
                        agent="modeling",
                        status="ok",
                        summary=summary,
                        payload=payload,
                    )
                )
            except Exception as e:
                steps_out.append(
                    AgentStepResult(
                        agent="modeling",
                        status="error",
                        summary="Modeling failed.",
                        error=str(e),
                        payload={},
                    )
                )

    if _should_run("evaluation", req):
        if modeling_payload is None:
            steps_out.append(
                AgentStepResult(
                    agent="evaluation",
                    status="skipped",
                    summary="No successful modeling step; evaluation skipped.",
                    payload={},
                )
            )
        else:
            try:
                summary, payload = run_evaluation(modeling_payload)
                steps_out.append(
                    AgentStepResult(
                        agent="evaluation",
                        status="ok",
                        summary=summary,
                        payload=payload,
                    )
                )
            except Exception as e:
                steps_out.append(
                    AgentStepResult(
                        agent="evaluation",
                        status="error",
                        summary="Evaluation step failed.",
                        error=str(e),
                    )
                )

    if _should_run("visualization", req):
        try:
            summary, payload = run_visualization(current_df)
            steps_out.append(
                AgentStepResult(
                    agent="visualization",
                    status="ok",
                    summary=summary,
                    payload=payload,
                )
            )
        except Exception as e:
            steps_out.append(
                AgentStepResult(
                    agent="visualization",
                    status="error",
                    summary="Visualization prep failed.",
                    error=str(e),
                )
            )

    finished = datetime.now(timezone.utc).isoformat()
    return AnalyzeResponse(
        upload_id=upload_id,
        started_at=started,
        finished_at=finished,
        target_column=target,
        steps=steps_out,
    )
