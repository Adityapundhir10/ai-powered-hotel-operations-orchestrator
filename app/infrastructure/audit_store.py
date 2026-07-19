from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from app.config import settings


class AuditStore:
    """PostgreSQL/SQLAlchemy-ready audit sink with a zero-setup memory mode."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self.engine = None
        self.session_factory = None

    async def initialize(self):
        if not settings.database_url:
            return
        try:
            from sqlalchemy import JSON, DateTime, Integer, String
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
            from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

            class Base(DeclarativeBase):
                pass

            class WorkflowAudit(Base):
                __tablename__ = "workflow_audits"
                id: Mapped[int] = mapped_column(Integer, primary_key=True)
                complaint_id: Mapped[str] = mapped_column(String(80), index=True)
                event_type: Mapped[str] = mapped_column(String(80), index=True)
                payload: Mapped[dict] = mapped_column(JSON)
                created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

            self.WorkflowAudit = WorkflowAudit
            self.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
            self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
            async with self.engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
        except Exception:
            self.engine = None
            self.session_factory = None

    async def write(self, complaint_id: str, event_type: str, payload: dict[str, Any]):
        record = {
            "complaint_id": complaint_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.records.append(record)
        self.records = self.records[-10000:]
        if self.session_factory is not None:
            async with self.session_factory() as session:
                session.add(self.WorkflowAudit(complaint_id=complaint_id, event_type=event_type, payload=payload))
                await session.commit()


audit_store = AuditStore()
