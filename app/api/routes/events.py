from fastapi import APIRouter, Query
from app.infrastructure.event_bus import event_bus
from app.schemas import EventRecord

router = APIRouter()


@router.get("/recent", response_model=list[EventRecord])
def recent_events(limit: int = Query(default=20, ge=1, le=200)):
    return event_bus.recent(limit)
