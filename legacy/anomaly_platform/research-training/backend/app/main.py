import sys
import os

# Ensure the app package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import async_session, init_db
from auth import create_default_admin
from config import settings
from routers import auth as auth_router
from routers import experiments as experiments_router
from routers import training as training_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[startup] Initializing database...")
    await init_db()
    print("[startup] Database tables created")

    # Create default admin user
    async with async_session() as db:
        await create_default_admin(db)

    print(f"[startup] Project root: {settings.PROJECT_ROOT}")
    print(f"[startup] CORS origins: {settings.CORS_ORIGINS}")
    yield
    # Shutdown
    from training import runner
    if runner.is_running:
        print("[shutdown] Stopping running training...")
        await runner.stop()
    print("[shutdown] Server stopped")


app = FastAPI(
    title="FedTAD Training Manager",
    description="Web UI for managing FedTAD federated graph learning training",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins = [
    o.strip()
    for o in settings.CORS_ORIGINS.split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router.router)
app.include_router(experiments_router.router)
app.include_router(training_router.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
