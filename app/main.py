import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.session import init_db
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Autonomous Research Agent",
    description="An agentic AI that autonomously plans, researches, and reflects to answer any goal.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
