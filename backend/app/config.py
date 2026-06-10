from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "UngerFink-TREND API"
    debug: bool = False
    api_prefix: str = "/api"

    # Database (use SQLite for local dev without Docker, PostgreSQL for production)
    database_url: str = "sqlite+aiosqlite:///./ungerfink_trend.db"
    database_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Data paths (existing project)
    project_root: str = r"P:\MCH\UngerFink-TREND"
    data_dir: str = r"P:\MCH\UngerFink-TREND\data"

    # Paper trading
    paper_polling_interval_seconds: int = 900  # 15 minutes

    # Auth (future)
    secret_key: str = "dev-secret-change-in-production"
    access_token_expire_minutes: int = 1440  # 24 hours

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
