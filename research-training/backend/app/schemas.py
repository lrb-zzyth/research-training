from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str

    @field_validator("username")
    @classmethod
    def username_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Username must not be empty")
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


class RegisterResponse(BaseModel):
    id: int
    username: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserInfo(BaseModel):
    id: int
    username: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Experiment ────────────────────────────────────────────────────────────────

class ExperimentCreate(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}


class ExperimentUpdate(BaseModel):
    status: Optional[str] = None
    best_round: Optional[int] = None
    best_val: Optional[float] = None
    best_test: Optional[float] = None
    end_time: Optional[datetime] = None


class ExperimentResponse(BaseModel):
    id: int
    name: str
    description: str
    status: str
    command: str
    parameters: dict
    # 平台集成: 配置快照与任务生命周期
    # (Optional: 兼容迁移前的历史记录)
    canonical_config: Optional[dict] = None
    git_commit: Optional[str] = None
    environment: Optional[dict] = None
    task_dir: Optional[str] = None
    last_round: Optional[int] = None
    exit_code: Optional[int] = None
    error_tail: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    best_round: Optional[int] = None
    best_val: Optional[float] = None
    best_test: Optional[float] = None
    created_by: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Training Log ──────────────────────────────────────────────────────────────

class TrainingLogResponse(BaseModel):
    id: int
    experiment_id: int
    line: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Training Metric ───────────────────────────────────────────────────────────

class TrainingMetricResponse(BaseModel):
    id: int
    experiment_id: int
    round: Optional[int] = None
    metric_name: str
    metric_value: float
    source: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── 参数热更新 ────────────────────────────────────────────────────────────────

class ParameterUpdateRequest(BaseModel):
    parameter: str
    value: object = None


class ParameterUpdateResponse(BaseModel):
    id: int
    experiment_id: int
    request_id: str = ""
    parameter: str
    old_value: Optional[object] = None
    new_value: object = None
    effective_round: Optional[int] = None
    status: str = "pending"
    reason: Optional[str] = None
    created_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None

    class Config:
        from_attributes = True
