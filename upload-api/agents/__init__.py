"""
Agent entrypoints that wrap `pipline` for the FastAPI orchestrator.

Import submodules explicitly, e.g. `from agents.data_agent import run`.
"""

__all__ = [
    "cleaning_agent",
    "data_agent",
    "evaluation_agent",
    "feature_agent",
    "model_agent",
    "visualization_agent",
]
