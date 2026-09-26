import asyncio
import os
import signal
import sys
from collections import deque
from typing import Optional

from sqlalchemy import select
from sqlalchemy.sql import func

from database import async_session
from models import (Experiment, ExperimentStatus, ParameterUpdate,
                    TrainingLog, TrainingMetric)
from log_parser import (parse_line, parse_event_line, event_to_metrics,
                        ROUND_PATTERN)
from websocket_manager import manager
from config import settings


class TrainingRunner:
    """
    Manages a single training subprocess with async I/O.
    使用 argv 数组直传子进程 (不做 shell 字符串拼接, 防注入)。
    """

    def __init__(self):
        self._process: Optional[asyncio.subprocess.Process] = None
        self._experiment_id: Optional[int] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._recent_lines: deque = deque(maxlen=500)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def experiment_id(self) -> Optional[int]:
        return self._experiment_id

    async def start(self, experiment_id: int, argv: list):
        """Start training in a subprocess (argv list, 不经过 shell)."""
        self._experiment_id = experiment_id
        self._stop_event.clear()
        self._recent_lines.clear()

        # Update experiment status
        async with async_session() as db:
            result = await db.execute(
                select(Experiment).where(Experiment.id == experiment_id)
            )
            exp = result.scalar_one_or_none()
            if exp:
                exp.status = ExperimentStatus.running
                exp.start_time = func.now()
                await db.commit()

        await manager.broadcast_status(experiment_id, "running",
                                       "Training started")

        # Start subprocess (argv 数组直传, 无 shell)
        project_root = settings.PROJECT_ROOT
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=project_root,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            preexec_fn=lambda: os.setsid() if sys.platform != "win32" else None,
        )

        # Read output in a background task
        self._task = asyncio.create_task(self._read_output())
        # Also wait for process to complete
        asyncio.create_task(self._wait_process())

    async def _read_output(self):
        """Read subprocess stdout line by line.

        结构化 [EVENT] 行是第一数据源; 自然语言正则仅作 fallback。
        """
        assert self._process is not None
        assert self._experiment_id is not None

        experiment_id = self._experiment_id
        current_round = None

        async for line_bytes in self._process.stdout:
            if self._stop_event.is_set():
                break

            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
            self._recent_lines.append(line)

            # Track current round from "[Round N]" headers (fallback)
            m_round = ROUND_PATTERN.search(line)
            if m_round:
                current_round = int(m_round.group(1))

            # Store log to DB (in a separate session to avoid blocking)
            asyncio.create_task(self._store_log(experiment_id, line))

            # Broadcast raw log via WebSocket
            await manager.broadcast_log(experiment_id, line)

            # 结构化事件优先
            ev = parse_event_line(line)
            if ev is not None:
                parsed_metrics, lifecycle, payload = event_to_metrics(ev)
                if lifecycle is not None:
                    asyncio.create_task(
                        self._handle_lifecycle(experiment_id, lifecycle, payload)
                    )
            else:
                parsed_metrics, lifecycle, payload = [], None, None
                parsed_metrics = parse_line(line, current_round)

            if parsed_metrics:
                asyncio.create_task(
                    self._store_metrics(experiment_id, parsed_metrics)
                )
                # Broadcast metrics
                metrics_data = [
                    {
                        "metric_name": m.metric_name,
                        "metric_value": m.metric_value,
                        "source": m.source,
                        "round": m.round,
                    }
                    for m in parsed_metrics
                ]
                await manager.broadcast_metrics(experiment_id, metrics_data)

    async def _handle_lifecycle(self, experiment_id: int,
                                event_type: str, payload: dict):
        """生命周期事件: last_round / 参数热更新审计 / 失败原因。"""
        try:
            async with async_session() as db:
                result = await db.execute(
                    select(Experiment).where(Experiment.id == experiment_id)
                )
                exp = result.scalar_one_or_none()
                if exp is None:
                    return
                if event_type == "round_started" and payload.get("round") is not None:
                    exp.last_round = int(payload["round"])
                elif event_type == "task_started":
                    exp.last_round = 0
                elif event_type == "task_finished":
                    if payload.get("round") is not None:
                        exp.last_round = int(payload["round"])
                elif event_type == "parameter_update_applied":
                    parameter = payload.get("parameter")
                    new_value = payload.get("new_value")
                    effective_round = payload.get("effective_round")
                    # 匹配 pending 审计记录
                    res = await db.execute(
                        select(ParameterUpdate).where(
                            ParameterUpdate.experiment_id == experiment_id,
                            ParameterUpdate.parameter == parameter,
                            ParameterUpdate.status == "pending",
                        ).order_by(ParameterUpdate.id.desc()).limit(1)
                    )
                    upd = res.scalar_one_or_none()
                    if upd is not None:
                        upd.status = "applied"
                        upd.effective_round = effective_round
                        upd.applied_at = func.now()
                        if upd.old_value is None:
                            upd.old_value = payload.get("old_value")
                        if new_value is not None:
                            upd.new_value = new_value
                await db.commit()
        except Exception as e:
            print(f"[training] Failed to handle lifecycle event "
                  f"{event_type}: {e}", file=sys.stderr)

    async def _store_log(self, experiment_id: int, line: str):
        try:
            async with async_session() as db:
                log_entry = TrainingLog(
                    experiment_id=experiment_id,
                    line=line,
                )
                db.add(log_entry)
                await db.commit()
        except Exception as e:
            print(f"[training] Failed to store log: {e}", file=sys.stderr)

    async def _store_metrics(self, experiment_id: int,
                             parsed_metrics: list):
        try:
            async with async_session() as db:
                for pm in parsed_metrics:
                    metric = TrainingMetric(
                        experiment_id=experiment_id,
                        round=pm.round,
                        metric_name=pm.metric_name,
                        metric_value=pm.metric_value,
                        source=pm.source,
                    )
                    db.add(metric)

                    # Update experiment best if applicable
                    if pm.metric_name == "best_val":
                        result = await db.execute(
                            select(Experiment).where(
                                Experiment.id == experiment_id)
                        )
                        exp = result.scalar_one_or_none()
                        if exp:
                            exp.best_val = pm.metric_value
                    elif pm.metric_name == "best_test":
                        result = await db.execute(
                            select(Experiment).where(
                                Experiment.id == experiment_id)
                        )
                        exp = result.scalar_one_or_none()
                        if exp:
                            exp.best_test = pm.metric_value
                    elif pm.metric_name == "current_round":
                        result = await db.execute(
                            select(Experiment).where(
                                Experiment.id == experiment_id)
                        )
                        exp = result.scalar_one_or_none()
                        if exp and pm.round is not None:
                            exp.best_round = pm.round

                await db.commit()
        except Exception as e:
            print(f"[training] Failed to store metrics: {e}",
                  file=sys.stderr)

    async def _wait_process(self):
        """Wait for process to finish and update status.

        失败时保存 exit code 与日志尾部 (stderr 已合并进 stdout)。
        """
        assert self._process is not None
        assert self._experiment_id is not None

        returncode = await self._process.wait()
        self._process = None

        # Determine final status
        if self._stop_event.is_set():
            final_status = ExperimentStatus.stopped
        elif returncode == 0:
            final_status = ExperimentStatus.finished
        else:
            final_status = ExperimentStatus.failed

        # Update DB
        error_tail = None
        if final_status == ExperimentStatus.failed:
            error_tail = "\n".join(list(self._recent_lines)[-200:])[-4000:]
        async with async_session() as db:
            result = await db.execute(
                select(Experiment).where(
                    Experiment.id == self._experiment_id)
            )
            exp = result.scalar_one_or_none()
            if exp:
                exp.status = final_status
                exp.end_time = func.now()
                exp.exit_code = int(returncode)
                if error_tail:
                    exp.error_tail = error_tail
                await db.commit()

        await manager.broadcast_status(
            self._experiment_id, final_status.value,
            f"Process exited with code {returncode}"
        )

        self._experiment_id = None

    async def stop(self):
        """Gracefully stop the training process (进程组, 不留孤儿)。"""
        self._stop_event.set()

        if self._process is not None and self._process.returncode is None:
            try:
                # Send SIGTERM to the process group
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                else:
                    self._process.terminate()

                # Wait up to 5 seconds for graceful shutdown
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    else:
                        self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Already dead

        self._process = None

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Update DB status if not already updated
        if self._experiment_id is not None:
            async with async_session() as db:
                result = await db.execute(
                    select(Experiment).where(
                        Experiment.id == self._experiment_id)
                )
                exp = result.scalar_one_or_none()
                if exp and exp.status == ExperimentStatus.running:
                    exp.status = ExperimentStatus.stopped
                    exp.end_time = func.now()
                    await db.commit()

            await manager.broadcast_status(
                self._experiment_id, "stopped", "Training stopped by user"
            )
            self._experiment_id = None


# Singleton
runner = TrainingRunner()
