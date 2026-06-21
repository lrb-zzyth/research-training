import json
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Experiment, User
from schemas import ExperimentCreate, ExperimentResponse
from auth import get_current_user
from training import runner
from websocket_manager import manager

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", response_model=ExperimentResponse)
async def start_training(
    request: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if runner.is_running:
        raise HTTPException(
            status_code=400,
            detail=f"Training already running (experiment #{runner.experiment_id})",
        )

    # Build command from parameters
    params = request.parameters
    cmd_parts = ["python", "train_fedtad.py"]

    # Map frontend keys to CLI flags (exclude non-CLI keys)
    param_to_flag = {
        "root": "--root",
        "dataset": "--dataset",
        "num_clients": "--num_clients",
        "partition": "--partition",
        "gpu_id": "--gpu_id",
        "num_rounds": "--num_rounds",
        "num_epochs": "--num_epochs",
        "num_dims": "--num_dims",
        "lr": "--lr",
        "weight_decay": "--weight_decay",
        "dropout": "--dropout",
        "hid_dim": "--hid_dim",
        "glb_epochs": "--glb_epochs",
        "it_g": "--it_g",
        "it_d": "--it_d",
        "lr_g": "--lr_g",
        "lr_d": "--lr_d",
        "fedtad_mode": "--fedtad_mode",
        "num_gen": "--num_gen",
        "lam1": "--lam1",
        "lam2": "--lam2",
        "topk": "--topk",
        "use_weighted_ce": "--use_weighted_ce",
        "class_weight_method": "--class_weight_method",
        "beta": "--beta",
        "use_contrastive": "--use_contrastive",
        "lambda_cl": "--lambda_cl",
        "cl_tau": "--cl_tau",
        "generator_type": "--generator_type",
        "diffusion_steps": "--diffusion_steps",
        "diffusion_hidden": "--diffusion_hidden",
        "diffusion_beta_start": "--diffusion_beta_start",
        "diffusion_beta_end": "--diffusion_beta_end",
        "seed": "--seed",
        "part_delta": "--part_delta",
    }

    for key, flag in param_to_flag.items():
        if key in params and params[key] is not None and params[key] != "":
            value = params[key]
            # Boolean flags (store_true)
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(flag)
            else:
                cmd_parts.append(flag)
                cmd_parts.append(str(value))

    command = " ".join(cmd_parts)

    # Create experiment record
    exp = Experiment(
        name=request.name,
        description=request.description,
        status="pending",
        command=command,
        parameters=params,
        created_by=current_user.id,
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    # Start training asynchronously
    await runner.start(exp.id, command)

    # Refresh to get updated status
    await db.refresh(exp)
    return exp


@router.post("/stop")
async def stop_training(current_user: User = Depends(get_current_user)):
    if not runner.is_running:
        raise HTTPException(status_code=400, detail="No training is running")
    await runner.stop()
    return {"message": "Training stop signal sent"}


@router.get("/status")
async def training_status(current_user: User = Depends(get_current_user)):
    return {
        "is_running": runner.is_running,
        "experiment_id": runner.experiment_id,
    }


@router.websocket("/ws/{experiment_id}")
async def training_websocket(websocket: WebSocket, experiment_id: int):
    await manager.connect(experiment_id, websocket)
    try:
        while True:
            # Keep the connection alive, receive pings if any
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(experiment_id, websocket)
    except Exception:
        await manager.disconnect(experiment_id, websocket)
