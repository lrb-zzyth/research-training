import json
import os
import platform
import subprocess
import sys
import uuid

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from database import get_db
from models import Experiment, ParameterUpdate, User
from schemas import (ExperimentCreate, ExperimentResponse,
                     ParameterUpdateRequest, ParameterUpdateResponse)
from auth import get_current_user
from training import runner
from websocket_manager import manager
from config import settings
from config_schema import (validate_and_normalize, COMMAND_FIELDS,
                           BOOLEAN_OPTIONAL, HOT_UPDATABLE, ALIAS_TO_PRIMARY)

router = APIRouter(prefix="/api/training", tags=["training"])


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=settings.PROJECT_ROOT,
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip()[:64] if out.returncode == 0 else ""
    except Exception:
        return ""


def _environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import torch_geometric
        env["torch_geometric"] = torch_geometric.__version__
    except Exception:
        pass
    return env


def _build_argv(canonical: dict, task_dir: str, experiment_id: int) -> list:
    """canonical config -> train_fedtad.py argv 数组 (无 shell, 防注入)。

    任务产物默认落盘到 task_dir: checkpoint / 日志 / 指标 / 事件 / 热更新目录。
    """
    argv = [sys.executable, "train_fedtad.py"]
    # 任务目录产物 (用户未显式提供时)
    checkpoints_dir = os.path.join(task_dir, "checkpoints")
    if not canonical.get("checkpoint_dir"):
        argv += ["--checkpoint_dir", checkpoints_dir]
        argv += ["--save_last_checkpoint"]
    else:
        argv += ["--checkpoint_dir", canonical["checkpoint_dir"]]
        if canonical.get("save_last_checkpoint"):
            argv += ["--save_last_checkpoint"]
    argv += ["--log_dir", os.path.join(task_dir, "logs")]
    argv += ["--metrics_jsonl", os.path.join(task_dir, "metrics.jsonl")]
    argv += ["--events_jsonl", os.path.join(task_dir, "events.jsonl")]
    argv += ["--final_metrics_json", os.path.join(task_dir, "final_metrics.json")]
    argv += ["--evaluation_report_json", os.path.join(task_dir, "evaluation_report.json")]
    argv += ["--resource_usage_json", os.path.join(task_dir, "resource_usage.json")]
    argv += ["--emit_events"]
    argv += ["--param_update_dir", task_dir]
    argv += ["--task_id", str(experiment_id)]

    for key in COMMAND_FIELDS:
        if key not in canonical:
            continue
        value = canonical[key]
        # 空字符串视为未提供 (避免覆盖上面显式设置的路径参数)
        if isinstance(value, str) and value == "":
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if key in BOOLEAN_OPTIONAL:
                argv.append(flag if value else f"--no-{key}")
            elif value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))
    return argv


def _task_dir(experiment_id: int) -> str:
    """每个任务的产物目录: <PROJECT_ROOT>/runs/platform/exp_<id>/ (gitignored)。"""
    d = os.path.join(settings.PROJECT_ROOT, "runs", "platform", f"exp_{experiment_id}")
    os.makedirs(d, exist_ok=True)
    return d


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

    # ---- canonical 校验: 未知/非法参数 422 拒绝 ----
    canonical, errors = validate_and_normalize(request.parameters)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"errors": errors[:50],
                    "hint": "请检查参数名称/类型/范围; 未知参数不被接受"})

    # Create experiment record (配置快照: 原始参数 + canonical + git + 环境)
    exp = Experiment(
        name=request.name,
        description=request.description,
        status="pending",
        command="",
        parameters=request.parameters,
        canonical_config=canonical,
        git_commit=_git_commit(),
        environment=_environment(),
        created_by=current_user.id,
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    # 任务产物目录
    task_dir = _task_dir(exp.id)
    exp.task_dir = task_dir

    # 构建 argv (数组直传, 无 shell 拼接)
    argv = _build_argv(canonical, task_dir, exp.id)
    command_str = " ".join(argv)
    exp.command = command_str
    await db.commit()
    await db.refresh(exp)

    # Start training asynchronously
    await runner.start(exp.id, argv)

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


# ---------------------------------------------------------------------------
#  参数热更新 (安全轮次边界生效; 非白名单参数返回"需要重启")
# ---------------------------------------------------------------------------

@router.post("/update", response_model=ParameterUpdateResponse)
async def update_training_param(
    request: ParameterUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not runner.is_running or runner.experiment_id is None:
        raise HTTPException(status_code=400,
                            detail="No training is running; 参数只能在训练运行中热更新")
    experiment_id = runner.experiment_id
    param = request.parameter

    meta = HOT_UPDATABLE.get(param)
    if meta is None:
        raise HTTPException(
            status_code=400,
            detail=f"参数 {param} 不支持热更新; 该参数需要重新启动训练任务。"
                   f"可热更新参数: {sorted(HOT_UPDATABLE)}")

    # 类型/范围校验
    try:
        vtype = {"int": int, "float": float, "str": str, "bool": bool}[meta["type"]]
        value = vtype(request.value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422,
                            detail=f"{param}: invalid value {request.value!r}")
    if "min" in meta and value < meta["min"]:
        raise HTTPException(status_code=422,
                            detail=f"{param}: {value} < min {meta['min']}")
    if "max" in meta and value > meta["max"]:
        raise HTTPException(status_code=422,
                            detail=f"{param}: {value} > max {meta['max']}")
    if "choices" in meta and value not in meta["choices"]:
        raise HTTPException(status_code=422,
                            detail=f"{param}: {value!r} not in choices {meta['choices']}")

    # 当前生效值 (canonical 快照只保留主键, 别名映射回主键)
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id))
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    primary_key = ALIAS_TO_PRIMARY.get(param, param)
    cur = (exp.canonical_config or {}).get(primary_key)

    request_id = uuid.uuid4().hex[:12]
    upd = ParameterUpdate(
        experiment_id=experiment_id,
        request_id=request_id,
        parameter=param,
        old_value=cur,
        new_value=value,
        status="pending",
    )
    db.add(upd)
    await db.commit()
    await db.refresh(upd)

    # 写入训练进程轮次边界检查的 pending 文件
    task_dir = exp.task_dir or _task_dir(experiment_id)
    pending_path = os.path.join(task_dir, "pending_updates.json")
    try:
        existing = []
        if os.path.exists(pending_path):
            with open(pending_path) as f:
                existing = json.load(f)
        existing.append({
            "parameter": param, "value": value,
            "request_id": request_id,
        })
        with open(pending_path, "w") as f:
            json.dump(existing, f)
    except Exception as e:
        # 文件写失败 -> 回滚审计记录, 明确失败原因
        upd.status = "rejected"
        upd.reason = f"pending 文件写入失败: {e}"
        await db.commit()
        raise HTTPException(status_code=500,
                            detail=f"无法写入热更新请求: {e}")

    await manager.broadcast_status(
        experiment_id, "running",
        f"参数更新请求: {param}={value} (待下一轮生效)")
    return upd


@router.get("/updates", response_model=list[ParameterUpdateResponse])
async def list_param_updates(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(ParameterUpdate)
        .where(ParameterUpdate.experiment_id == experiment_id)
        .order_by(ParameterUpdate.id.desc()).limit(200))
    return result.scalars().all()


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
