import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import ResearchRun, RunStep, RunStatus


class RunRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Create ────────────────────────────────────────────────────────────────

    async def create(self, run_id: str, goal: str) -> ResearchRun:
        run = ResearchRun(id=run_id, goal=goal, status=RunStatus.pending)
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        return run

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, run_id: str) -> Optional[ResearchRun]:
        result = await self.session.execute(
            select(ResearchRun)
            .options(selectinload(ResearchRun.steps))
            .where(ResearchRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def list(self, limit: int = 20, offset: int = 0) -> list[ResearchRun]:
        result = await self.session.execute(
            select(ResearchRun)
            .order_by(desc(ResearchRun.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    # ── Status updates ────────────────────────────────────────────────────────

    async def set_running(self, run_id: str) -> None:
        run = await self._get_or_raise(run_id)
        run.status = RunStatus.running
        run.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

    async def set_completed(self, run_id: str, report: str, iteration_count: int) -> None:
        run = await self._get_or_raise(run_id)
        run.status = RunStatus.completed
        run.report = report
        run.iteration_count = iteration_count
        run.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

    async def set_failed(self, run_id: str, error: str) -> None:
        run = await self._get_or_raise(run_id)
        run.status = RunStatus.failed
        run.error = error
        run.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete(self, run_id: str) -> bool:
        result = await self.session.execute(
            select(ResearchRun).where(ResearchRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if run is None:
            return False
        await self.session.delete(run)
        await self.session.commit()
        return True

    # ── Steps ─────────────────────────────────────────────────────────────────

    async def add_step(
        self,
        run_id: str,
        node: str,
        input_data: dict,
        output_data: dict,
        tool_called: Optional[str] = None,
    ) -> RunStep:
        step = RunStep(
            run_id=run_id,
            node=node,
            tool_called=tool_called,
            input=json.dumps(input_data),
            output=json.dumps(output_data),
        )
        self.session.add(step)
        await self.session.commit()
        return step

    # ── Private helper ────────────────────────────────────────────────────────

    async def _get_or_raise(self, run_id: str) -> ResearchRun:
        result = await self.session.execute(
            select(ResearchRun).where(ResearchRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if run is None:
            raise ValueError(f"Run '{run_id}' not found")
        return run
