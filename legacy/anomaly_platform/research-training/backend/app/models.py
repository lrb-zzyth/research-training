import enum
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, JSON,
    ForeignKey, Enum as SAEnum
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from database import Base


class ExperimentStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    stopped = "stopped"
    finished = "finished"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    experiments = relationship("Experiment", back_populates="creator")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    status = Column(SAEnum(ExperimentStatus), default=ExperimentStatus.pending)
    command = Column(Text, nullable=False)
    parameters = Column(JSON, default=dict)
    # 平台集成 (配置快照与任务生命周期)
    canonical_config = Column(JSON, default=dict)
    git_commit = Column(String(64), default="")
    environment = Column(JSON, default=dict)
    task_dir = Column(String(512), default="")
    last_round = Column(Integer, nullable=True)
    exit_code = Column(Integer, nullable=True)
    error_tail = Column(Text, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    best_round = Column(Integer, nullable=True)
    best_val = Column(Float, nullable=True)
    best_test = Column(Float, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    creator = relationship("User", back_populates="experiments")
    logs = relationship("TrainingLog", back_populates="experiment",
                        cascade="all, delete-orphan")
    metrics = relationship("TrainingMetric", back_populates="experiment",
                           cascade="all, delete-orphan")
    param_updates = relationship("ParameterUpdate", back_populates="experiment",
                                 cascade="all, delete-orphan")


class ParameterUpdate(Base):
    """参数热更新审计记录 (任务启动后, 轮次边界生效)。"""

    __tablename__ = "parameter_updates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"),
                           nullable=False, index=True)
    request_id = Column(String(64), default="")
    parameter = Column(String(100), nullable=False)
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=False)
    effective_round = Column(Integer, nullable=True)
    status = Column(String(20), default="pending")  # pending / applied / rejected
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    applied_at = Column(DateTime(timezone=True), nullable=True)

    experiment = relationship("Experiment", back_populates="param_updates")


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"),
                           nullable=False, index=True)
    line = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    experiment = relationship("Experiment", back_populates="logs")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"),
                           nullable=False, index=True)
    round = Column(Integer, nullable=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    source = Column(String(100), default="server")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    experiment = relationship("Experiment", back_populates="metrics")
