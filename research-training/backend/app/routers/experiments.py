import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Experiment, ExperimentStatus, TrainingLog, TrainingMetric, User
from schemas import (
    ExperimentCreate, ExperimentResponse,
    TrainingLogResponse, TrainingMetricResponse,
)
from auth import get_current_user

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/", response_model=list[ExperimentResponse])
async def list_experiments(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Experiment)
        .order_by(desc(Experiment.created_at))
        .offset(offset)
        .limit(limit)
    )
    experiments = result.scalars().all()
    return experiments


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.get("/{experiment_id}/logs", response_model=list[TrainingLogResponse])
async def get_experiment_logs(
    experiment_id: int,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(TrainingLog)
        .where(TrainingLog.experiment_id == experiment_id)
        .order_by(TrainingLog.id)
        .offset(offset)
        .limit(limit)
    )
    logs = result.scalars().all()
    return logs


@router.get("/{experiment_id}/metrics", response_model=list[TrainingMetricResponse])
async def get_experiment_metrics(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(TrainingMetric)
        .where(TrainingMetric.experiment_id == experiment_id)
        .order_by(TrainingMetric.id)
    )
    metrics = result.scalars().all()
    return metrics


@router.get("/{experiment_id}/config")
async def get_experiment_config(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """完整配置快照 (用于复制为新任务 / 导出配置 / 复现)。"""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "status": exp.status.value if hasattr(exp.status, "value") else exp.status,
        "raw_parameters": exp.parameters,
        "canonical_config": exp.canonical_config,
        "git_commit": exp.git_commit,
        "environment": exp.environment,
        "task_dir": exp.task_dir,
        "command": exp.command,
        "best_round": exp.best_round,
        "best_val": exp.best_val,
        "best_test": exp.best_test,
        "last_round": exp.last_round,
        "exit_code": exp.exit_code,
        "start_time": str(exp.start_time) if exp.start_time else None,
        "end_time": str(exp.end_time) if exp.end_time else None,
        "resume_hint": {
            "resume_checkpoint":
                (exp.canonical_config or {}).get("resume_checkpoint") or "",
            "checkpoint_dir":
                (exp.canonical_config or {}).get("checkpoint_dir") or
                (f"{exp.task_dir}/checkpoints" if exp.task_dir else ""),
        },
    }
