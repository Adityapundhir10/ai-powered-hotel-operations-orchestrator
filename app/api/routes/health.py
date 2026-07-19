from fastapi import APIRouter
from app.config import settings
from app.infrastructure.event_bus import event_bus
from app.infrastructure.state_store import state_store

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "author": settings.author_name,
        "event_backend": "Kafka" if event_bus._producer is not None else "in-memory",
        "state_backend": state_store.backend,
    }


@router.get("/")
def root():
    return {
        "project": settings.app_name,
        "author": settings.author_name,
        "portfolio_reconstruction": True,
        "docs": "/docs",
        "dashboard": "/dashboard",
        "capabilities": [
            "occupancy_forecasting",
            "dynamic_pricing",
            "complaint_classification",
            "hybrid_rag",
            "workflow_orchestration",
            "document_extraction",
        ],
    }
