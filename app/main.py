from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.infrastructure.audit_store import audit_store
from app.api.routes import health, workflows, revenue, rag, documents, events


@asynccontextmanager
async def lifespan(_: FastAPI):
    await audit_store.initialize()
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Portfolio reconstruction by Aditya Pundhir: hotel occupancy forecasting, "
        "dynamic pricing, multilingual complaints, hybrid RAG, document extraction, "
        "and event-driven workflow orchestration."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["System"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["Hotel workflows"])
app.include_router(revenue.router, prefix="/api/v1/revenue", tags=["Revenue management"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["Hybrid RAG"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Document intelligence"])
app.include_router(events.router, prefix="/api/v1/events", tags=["Events"])


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard():
    path = Path(__file__).parent / "static" / "dashboard.html"
    return path.read_text(encoding="utf-8")
