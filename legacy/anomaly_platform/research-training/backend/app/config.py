import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Database (占位默认值; 实际配置来自 backend/.env, 该文件 gitignored)
    DATABASE_URL: str = "postgresql+asyncpg://fedtad_user:change-me@127.0.0.1:5432/fedtad"

    # JWT
    SECRET_KEY: str = "change-this-to-a-random-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Project paths
    # This file is at research-training/backend/app/config.py
    # Project root is three levels up
    PROJECT_ROOT: str = str(Path(__file__).resolve().parents[3])

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:8000"

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        env_file_encoding = "utf-8"


settings = Settings()
