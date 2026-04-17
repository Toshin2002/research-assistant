import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.db.repository import RunRepository
from app.schemas.run import StartRunRequest, RunOut, RunDetailOut
from app.agent.runner import run_agent

router = APIRouter(prefix="/runs", tags=["runs"])


# ── POST /runs ─────────────────────────────────────────────────────────────────

@router.post("", response_model=RunOut, status_code=202)
async def start_run(
    body:             StartRunRequest,
    background_tasks: BackgroundTasks,
    session:          AsyncSession = Depends(get_session),
):
    run_id = str(uuid.uuid4())
    repo   = RunRepository(session)
    run    = await repo.create(run_id=run_id, goal=body.goal)

    background_tasks.add_task(
        run_agent,
        run_id=run_id,
        goal=body.goal,
        max_iterations=body.max_iterations,
    )

    return run


# ── GET /runs ──────────────────────────────────────────────────────────────────

@router.get("", response_model=list[RunOut])
async def list_runs(
    limit:   int          = Query(default=20, ge=1, le=100),
    offset:  int          = Query(default=0,  ge=0),
    session: AsyncSession = Depends(get_session),
):
    return await RunRepository(session).list(limit=limit, offset=offset)


# ── GET /runs/{run_id} ────────────────────────────────────────────────────────

@router.get("/{run_id}", response_model=RunDetailOut)
async def get_run(
    run_id:  str,
    session: AsyncSession = Depends(get_session),
):
    run = await RunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


# ── DELETE /runs/{run_id} ─────────────────────────────────────────────────────

@router.delete("/{run_id}", status_code=204)
async def delete_run(
    run_id:  str,
    session: AsyncSession = Depends(get_session),
):
    deleted = await RunRepository(session).delete(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
