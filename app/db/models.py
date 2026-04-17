import uuid
import enum
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, relationship


# ── Status Enum ───────────────────────────────────────────────────────────────

class RunStatus(str, enum.Enum):
    pending   = "pending"
    running   = "running"
    completed = "completed"
    failed    = "failed"


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Tables ────────────────────────────────────────────────────────────────────

class ResearchRun(Base):
    __tablename__ = "research_runs"

    id              = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    goal            = Column(Text, nullable=False)
    status          = Column(SAEnum(RunStatus), nullable=False, default=RunStatus.pending)
    report          = Column(Text, nullable=True)
    error           = Column(Text, nullable=True)
    iteration_count = Column(Integer, default=0)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                             onupdate=lambda: datetime.now(timezone.utc))

    steps = relationship(
        "RunStep",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunStep.timestamp",
    )


class RunStep(Base):
    __tablename__ = "run_steps"

    id          = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id      = Column(String(36), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    node        = Column(String(64), nullable=False)
    tool_called = Column(String(64), nullable=True)
    input       = Column(Text, nullable=True)
    output      = Column(Text, nullable=True)
    timestamp   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    run = relationship("ResearchRun", back_populates="steps")
