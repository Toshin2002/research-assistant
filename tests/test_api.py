"""
Integration tests for the FastAPI layer.

- Database : in-memory SQLite (fresh per test, no files on disk)
- Agent    : mocked — we test the API contract, not the agent logic
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db.models import Base
from app.db.session import get_session
from app.main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    """
    Spin up a fresh in-memory SQLite database for each test.
    Creates all tables before the test, drops them after.
    """
    engine = create_async_engine(TEST_DB_URL)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with SessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    """
    HTTP test client wired to the FastAPI app.
    Overrides the database dependency to use the test session.
    """
    app.dependency_overrides[get_session] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Health check ──────────────────────────────────────────────────────────────

async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── POST /runs ────────────────────────────────────────────────────────────────

@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_start_run_returns_202(mock_runner, client):
    resp = await client.post("/runs", json={"goal": "Analyze AI coding assistants"})

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "pending"
    assert data["goal"]   == "Analyze AI coding assistants"
    assert "id"           in data
    assert "created_at"   in data


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_start_run_schedules_agent(mock_runner, client):
    await client.post("/runs", json={"goal": "Test goal"})

    mock_runner.assert_called_once()
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["goal"] == "Test goal"


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_start_run_passes_custom_iterations(mock_runner, client):
    await client.post("/runs", json={"goal": "Test goal", "max_iterations": 3})

    kwargs = mock_runner.call_args.kwargs
    assert kwargs["max_iterations"] == 3


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_start_run_missing_goal_returns_422(mock_runner, client):
    resp = await client.post("/runs", json={})

    assert resp.status_code == 422


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_start_run_invalid_iterations_returns_422(mock_runner, client):
    resp = await client.post("/runs", json={"goal": "Test", "max_iterations": "five"})

    assert resp.status_code == 422


# ── GET /runs ─────────────────────────────────────────────────────────────────

@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_list_runs_empty(mock_runner, client):
    resp = await client.get("/runs")

    assert resp.status_code == 200
    assert resp.json() == []


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_list_runs_returns_all(mock_runner, client):
    await client.post("/runs", json={"goal": "Goal A"})
    await client.post("/runs", json={"goal": "Goal B"})

    resp = await client.get("/runs")

    assert resp.status_code == 200
    assert len(resp.json()) == 2


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_list_runs_newest_first(mock_runner, client):
    await client.post("/runs", json={"goal": "First"})
    await client.post("/runs", json={"goal": "Second"})

    resp  = await client.get("/runs")
    goals = [r["goal"] for r in resp.json()]

    assert goals[0] == "Second"
    assert goals[1] == "First"


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_list_runs_pagination(mock_runner, client):
    for i in range(5):
        await client.post("/runs", json={"goal": f"Goal {i}"})

    resp = await client.get("/runs?limit=2&offset=0")
    assert len(resp.json()) == 2

    resp = await client.get("/runs?limit=2&offset=2")
    assert len(resp.json()) == 2

    resp = await client.get("/runs?limit=2&offset=4")
    assert len(resp.json()) == 1


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_list_runs_limit_validation(mock_runner, client):
    resp = await client.get("/runs?limit=200")
    assert resp.status_code == 422


# ── GET /runs/{run_id} ────────────────────────────────────────────────────────

@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_get_run_returns_detail(mock_runner, client):
    create_resp = await client.post("/runs", json={"goal": "Research quantum computing"})
    run_id      = create_resp.json()["id"]

    resp = await client.get(f"/runs/{run_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"]     == run_id
    assert data["goal"]   == "Research quantum computing"
    assert data["status"] == "pending"
    assert "steps"        in data
    assert isinstance(data["steps"], list)


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_get_run_not_found(mock_runner, client):
    resp = await client.get("/runs/00000000-0000-0000-0000-000000000000")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"


# ── DELETE /runs/{run_id} ─────────────────────────────────────────────────────

@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_delete_run_succeeds(mock_runner, client):
    create_resp = await client.post("/runs", json={"goal": "To be deleted"})
    run_id      = create_resp.json()["id"]

    del_resp = await client.delete(f"/runs/{run_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/runs/{run_id}")
    assert get_resp.status_code == 404


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_delete_run_not_found(mock_runner, client):
    resp = await client.delete("/runs/00000000-0000-0000-0000-000000000000")

    assert resp.status_code == 404


@patch("app.api.routes.run_agent", new_callable=AsyncMock)
async def test_delete_removes_from_list(mock_runner, client):
    r1 = await client.post("/runs", json={"goal": "Keep this"})
    r2 = await client.post("/runs", json={"goal": "Delete this"})

    await client.delete(f"/runs/{r2.json()['id']}")

    resp  = await client.get("/runs")
    goals = [r["goal"] for r in resp.json()]

    assert "Keep this"   in goals
    assert "Delete this" not in goals
