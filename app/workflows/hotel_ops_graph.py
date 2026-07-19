from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from app.config import settings
from app.ml.complaint_classifier import MultilingualComplaintClassifier
from app.rag.hybrid_retriever import HybridRetriever, SearchHit
from app.infrastructure.event_bus import event_bus
from app.infrastructure.state_store import state_store
from app.infrastructure.audit_store import audit_store
from app.schemas import ComplaintRequest, ClassificationResult, RetrievalHit, WorkflowResult


TEAM_BY_INTENT = {
    "maintenance": "Engineering & Maintenance",
    "housekeeping": "Housekeeping",
    "noise": "Front Office / Security",
    "billing": "Finance & Front Office",
    "food": "Food & Beverage",
    "safety": "Security & Manager on Duty",
    "service": "Front Office",
}
SLA_BY_SEVERITY = {"low": 120, "medium": 60, "high": 20, "critical": 5}


class WorkflowState(TypedDict, total=False):
    request: ComplaintRequest
    classification: ClassificationResult
    retrieved_sops: list[RetrievalHit]
    assigned_team: str
    sla_minutes: int
    escalation_required: bool
    human_approval_required: bool
    event_id: str
    audit_steps: list[str]
    status: str


class HotelOperationsWorkflow:
    def __init__(self):
        self.classifier = MultilingualComplaintClassifier()
        self.retriever = HybridRetriever()
        self.graph = self._build_langgraph() if settings.use_langgraph else None

    @staticmethod
    def _hit_schema(hit: SearchHit) -> RetrievalHit:
        return RetrievalHit(
            document_id=hit.document.document_id,
            title=hit.document.title,
            text=hit.document.text,
            metadata=hit.document.metadata,
            bm25_score=hit.bm25_score,
            vector_score=hit.vector_score,
            fused_score=hit.fused_score,
            rerank_score=hit.rerank_score,
        )

    def classify_node(self, state: WorkflowState) -> WorkflowState:
        request = state["request"]
        prediction = self.classifier.predict(request.text, request.language)
        state["classification"] = ClassificationResult(
            intent=prediction.intent,
            severity=prediction.severity,
            confidence=prediction.confidence,
            language=prediction.language,
            model_backend=prediction.backend,
        )
        state.setdefault("audit_steps", []).append("classified complaint intent and severity")
        return state

    def retrieve_node(self, state: WorkflowState) -> WorkflowState:
        request, classification = state["request"], state["classification"]
        query = f"{classification.intent} {classification.severity} {request.text}"
        hits = self.retriever.search(query, top_k=3)
        state["retrieved_sops"] = [self._hit_schema(hit) for hit in hits]
        state.setdefault("audit_steps", []).append("retrieved and ranked relevant SOPs")
        return state

    def route_node(self, state: WorkflowState) -> WorkflowState:
        classification = state["classification"]
        state["assigned_team"] = TEAM_BY_INTENT.get(classification.intent, "Front Office")
        state["sla_minutes"] = SLA_BY_SEVERITY.get(classification.severity, settings.default_sla_minutes)
        state["escalation_required"] = classification.severity in {"high", "critical"}
        state["human_approval_required"] = classification.severity == settings.human_approval_severity
        state.setdefault("audit_steps", []).append("assigned responsible team and calculated SLA")
        return state

    def event_node(self, state: WorkflowState) -> WorkflowState:
        request = state["request"]
        event = event_bus.publish(
            "hotel.complaint.workflow.created",
            {
                "complaint_id": request.complaint_id,
                "hotel_id": request.hotel_id,
                "assigned_team": state["assigned_team"],
                "severity": state["classification"].severity,
                "sla_minutes": state["sla_minutes"],
            },
        )
        state["event_id"] = event.event_id
        state["status"] = "awaiting_human_approval" if state["human_approval_required"] else "assigned"
        state_store.set(f"complaint:{request.complaint_id}", self._serializable_state(state))
        state.setdefault("audit_steps", []).append(f"published event using {event.backend}")
        return state

    @staticmethod
    def _serializable_state(state: WorkflowState) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in state.items():
            if hasattr(value, "model_dump"):
                result[key] = value.model_dump(mode="json")
            elif isinstance(value, list):
                result[key] = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value]
            else:
                result[key] = value
        return result

    def _build_langgraph(self):
        try:
            from langgraph.graph import StateGraph, START, END
            builder = StateGraph(WorkflowState)
            builder.add_node("classify", self.classify_node)
            builder.add_node("retrieve_sop", self.retrieve_node)
            builder.add_node("assign_and_sla", self.route_node)
            builder.add_node("publish_event", self.event_node)
            builder.add_edge(START, "classify")
            builder.add_edge("classify", "retrieve_sop")
            builder.add_edge("retrieve_sop", "assign_and_sla")
            builder.add_edge("assign_and_sla", "publish_event")
            builder.add_edge("publish_event", END)
            return builder.compile()
        except Exception:
            return None

    async def run(self, request: ComplaintRequest) -> WorkflowResult:
        initial: WorkflowState = {"request": request, "audit_steps": []}
        if self.graph is not None:
            state = self.graph.invoke(initial)
        else:
            state = self.event_node(self.route_node(self.retrieve_node(self.classify_node(initial))))
        await audit_store.write(request.complaint_id, "workflow_completed", self._serializable_state(state))
        return WorkflowResult(
            complaint_id=request.complaint_id,
            status=state["status"],
            classification=state["classification"],
            assigned_team=state["assigned_team"],
            sla_minutes=state["sla_minutes"],
            escalation_required=state["escalation_required"],
            human_approval_required=state["human_approval_required"],
            retrieved_sops=state["retrieved_sops"],
            event_id=state["event_id"],
            audit_steps=state["audit_steps"],
            created_at=datetime.now(timezone.utc),
        )


workflow_engine = HotelOperationsWorkflow()
