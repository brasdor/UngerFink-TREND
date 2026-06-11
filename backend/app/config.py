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

    # --- Live execution (Binance spot) -------------------------------------
    # Keys come from backend/.env and are NEVER committed. Start with TESTNET
    # keys. The master switch and testnet flag are the primary safety gates.
    exchange_api_key: str = ""
    exchange_secret: str = ""

    # Safety posture: default to fake money, with live placement disabled.
    exchange_testnet: bool = True          # ccxt set_sandbox_mode(True)
    live_trading_enabled: bool = False     # master switch — must be flipped on purpose

    # Risk / sizing
    default_risk_pct: float = 0.0025       # 0.25% of equity per trade (frozen config)

    # Hard caps (kill-switches)
    max_order_usdt: float = 100.0          # reject any single order above this notional
    daily_loss_limit_usdt: float = 200.0   # block new orders once breached for the day

    # --- Telegram alerts ---------------------------------------------------
    # From @BotFather. Empty = alerts silently disabled. Set in backend/.env.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Auth (future)
    secret_key: str = "dev-secret-change-in-production"
    access_token_expire_minutes: int = 1440  # 24 hours

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
