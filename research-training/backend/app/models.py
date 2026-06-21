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
