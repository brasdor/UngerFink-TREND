-- TimescaleDB initialization for UngerFink-TREND
-- This runs automatically on first docker-compose up

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- After tables are created by Alembic, convert to hypertables:
-- SELECT create_hypertable('candles', 'timestamp', if_not_exists => TRUE);
-- SELECT create_hypertable('equity_snapshots', 'timestamp', if_not_exists => TRUE);
