"""Run: uvicorn main:app --reload (from the upload-api directory)."""

from app.main import app

__all__ = ["app"]
