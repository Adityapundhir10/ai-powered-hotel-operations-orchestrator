from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4
import json

from app.config import settings
from app.schemas import EventRecord


class EventBus:
    """Kafka publisher with a thread-safe in-memory fallback."""

    def __init__(self):
        self._producer = None
        self._events: list[EventRecord] = []
        self._lock = Lock()
        if settings.kafka_bootstrap_servers:
            try:
                from kafka import KafkaProducer
                self._producer = KafkaProducer(
                    bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
                    value_serializer=lambda value: json.dumps(value, default=str).encode("utf-8"),
                    acks="all",
                    retries=3,
                )
            except Exception:
                self._producer = None

    def publish(self, event_type: str, payload: dict[str, Any]) -> EventRecord:
        record = EventRecord(
            event_id=str(uuid4()),
            event_type=event_type,
            payload=payload,
            produced_at=datetime.now(timezone.utc),
            backend="Kafka" if self._producer is not None else "in-memory",
        )
        if self._producer is not None:
            self._producer.send(settings.kafka_topic, record.model_dump(mode="json"))
            self._producer.flush(timeout=5)
        with self._lock:
            self._events.append(record)
            self._events = self._events[-10000:]
        return record

    def recent(self, limit: int = 20) -> list[EventRecord]:
        with self._lock:
            return list(reversed(self._events[-limit:]))


event_bus = EventBus()
