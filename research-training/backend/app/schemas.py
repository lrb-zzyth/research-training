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
