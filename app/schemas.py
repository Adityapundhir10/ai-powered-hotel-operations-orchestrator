from datetime import date, datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class ComplaintRequest(BaseModel):
    complaint_id: str = Field(min_length=1, max_length=80)
    hotel_id: str = Field(min_length=1, max_length=80)
    room_number: str | None = Field(default=None, max_length=20)
    language: str = Field(default="en", max_length=12)
    text: str = Field(min_length=3, max_length=5000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    intent: str
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0, le=1)
    language: str
    model_backend: str


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)
    metadata_filter: dict[str, str] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    document_id: str
    title: str
    text: str
    metadata: dict[str, str]
    bm25_score: float
    vector_score: float
    fused_score: float
    rerank_score: float | None = None


class WorkflowResult(BaseModel):
    complaint_id: str
    status: str
    classification: ClassificationResult
    assigned_team: str
    sla_minutes: int
    escalation_required: bool
    human_approval_required: bool
    retrieved_sops: list[RetrievalHit]
    event_id: str
    audit_steps: list[str]
    created_at: datetime


class ForecastRequest(BaseModel):
    hotel_id: str
    forecast_date: date
    lead_time: int = Field(ge=0, le=365)
    booking_pickup_7d: float = Field(ge=0)
    rolling_demand_14d: float = Field(ge=0, le=1.5)
    cancellation_rate_28d: float = Field(ge=0, le=1)
    available_inventory: int = Field(ge=0)
    base_adr: float = Field(gt=0)
    event_intensity: float = Field(default=0, ge=0, le=1)
    lag_occupancy_7d: float = Field(default=0.6, ge=0, le=1)
    lag_occupancy_28d: float = Field(default=0.6, ge=0, le=1)


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    hotel_id: str
    forecast_date: date
    predicted_occupancy: float
    recommended_adr: float
    estimated_revpar: float
    demand_multiplier: float
    model_backend: str
    feature_contributions: dict[str, float]


class DocumentExtractionResponse(BaseModel):
    document_type: str
    fields: dict[str, Any]
    field_confidences: dict[str, float]
    average_confidence: float
    backend: str
    warnings: list[str]


class EventRecord(BaseModel):
    event_id: str
    event_type: str
    payload: dict[str, Any]
    produced_at: datetime
    backend: str
