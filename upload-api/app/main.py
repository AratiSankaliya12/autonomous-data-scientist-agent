from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, CORS_ORIGIN_REGEX
from app.routers import analysis_router, health_router, uploads_router

app = FastAPI(
    title="Autonomous Data Scientist API",
    description="Upload datasets and run the multi-step agentic analysis pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(uploads_router)
app.include_router(analysis_router)
