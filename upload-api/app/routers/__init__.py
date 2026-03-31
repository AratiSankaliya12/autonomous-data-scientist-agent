from app.routers.analysis import router as analysis_router
from app.routers.health import router as health_router
from app.routers.uploads import router as uploads_router

__all__ = ["analysis_router", "health_router", "uploads_router"]
