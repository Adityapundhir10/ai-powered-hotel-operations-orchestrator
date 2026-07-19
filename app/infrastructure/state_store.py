from __future__ import annotations
from threading import Lock
from typing import Any
import json

from app.config import settings


class WorkflowStateStore:
    """Redis-backed JSON state with an in-memory fallback."""

    def __init__(self):
        self._redis = None
        self._memory: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        if settings.redis_url:
            try:
                import redis
                client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=1)
                client.ping()
                self._redis = client
            except Exception:
                self._redis = None

    @property
    def backend(self) -> str:
        return "Redis" if self._redis is not None else "in-memory"

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 86400):
        if self._redis is not None:
            self._redis.setex(key, ttl_seconds, json.dumps(value, default=str))
            return
        with self._lock:
            self._memory[key] = value

    def get(self, key: str) -> dict[str, Any] | None:
        if self._redis is not None:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        with self._lock:
            return self._memory.get(key)


state_store = WorkflowStateStore()
