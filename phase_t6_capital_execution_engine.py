import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PHASE T6 — CAPITAL & EXECUTION ENGINE
# ============================================================

INITIAL_CAPITAL = 10000
RISK_PER_TRADE = 0.01
MAX_PORTFOLIO_HEAT = 0.05
EXECUTION_COST_R = 0.02

INPUT_FILE = Path(
    'data/research_trend_t5/phase_t5_portfolio_trades.csv'
)

OUTPUT_DIR = Path(
    'data/research_trend_t6'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 80)
print('PHASE T6 — CAPITAL & EXECUTION ENGINE')
print('=' * 80)
print(f'Input: {INPUT_FILE}')
print(f'Output: {OUTPUT_DIR}')

if not INPUT_FILE.exists():
    raise FileNotFoundError(f'Missing input file: {INPUT_FILE}')

trades = pd.read_csv(INPUT_FILE)

# ------------------------------------------------------------
# Detect R column
# ------------------------------------------------------------
possible_r_cols = ['R', 'r_multiple', 'r', 'trade_r']
R_COL = None

for c in possible_r_cols:
    if c in trades.columns:
        R_COL = c
        break

if R_COL is None:
    numeric_cols = trades.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError('No numeric columns found for R values.')
    R_COL = numeric_cols[-1]

print(f'Using R column: {R_COL}')

# ------------------------------------------------------------
# Date column
# ------------------------------------------------------------
DATE_COL = None
for c in ['exit_time', 'close_time', 'timestamp', 'date']:
    if c in trades.columns:
        DATE_COL = c
        break

if DATE_COL is not None:
    trades[DATE_COL] = pd.to_datetime(trades[DATE_COL], utc=True)
    trades = trades.sort_values(DATE_COL).reset_index(drop=True)

# ------------------------------------------------------------
# Equity simulation
# ------------------------------------------------------------
capital = INITIAL_CAPITAL
peak = INITIAL_CAPITAL
heat = 0.0

records = []
heat_records = []

for i, row in trades.iterrows():

    r_value = float(row[R_COL])

    trade_risk = capital * RISK_PER_TRADE

    if heat + RISK_PER_TRADE > MAX_PORTFOLIO_HEAT:
        continue

    pnl = trade_risk * (r_value - EXECUTION_COST_R)

    capital += pnl

    peak = max(peak, capital)

    dd_pct = (capital - peak) / peak * 100

    heat = min(MAX_PORTFOLIO_HEAT, heat + RISK_PER_TRADE)

    records.append({
        'trade_number': i + 1,
        'capital': capital,
        'pnl': pnl,
        'r_value': r_value,
        'drawdown_pct': dd_pct,
    })

    heat_records.append({
        'trade_number': i + 1,
        'portfolio_heat': heat,
    })

    heat = max(0, heat - RISK_PER_TRADE)

# ------------------------------------------------------------
# Create outputs
# ------------------------------------------------------------

equity_df = pd.DataFrame(records)
heat_df = pd.DataFrame(heat_records)

if len(equity_df) == 0:
    raise ValueError('No executed trades after portfolio heat filtering.')

final_capital = equity_df['capital'].iloc[-1]
net_profit = final_capital - INITIAL_CAPITAL
max_dd = equity_df['drawdown_pct'].min()

summary = pd.DataFrame([
    {
        'initial_capital': INITIAL_CAPITAL,
        'final_capital': round(final_capital, 2),
        'net_profit': round(net_profit, 2),
        'return_pct': round((final_capital / INITIAL_CAPITAL - 1) * 100, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'executed_trades': len(equity_df),
        'risk_per_trade_pct': RISK_PER_TRADE * 100,
        'max_portfolio_heat_pct': MAX_PORTFOLIO_HEAT * 100,
    }
])

# ------------------------------------------------------------
# Save files
# ------------------------------------------------------------

equity_path = OUTPUT_DIR / 'phase_t6_equity_curve.csv'
trade_log_path = OUTPUT_DIR / 'phase_t6_trade_log.csv'
summary_path = OUTPUT_DIR / 'phase_t6_portfolio_summary.csv'
heat_path = OUTPUT_DIR / 'phase_t6_heat_timeline.csv'
report_path = OUTPUT_DIR / 'phase_t6_master_report.txt'

equity_df.to_csv(equity_path, index=False)
equity_df.to_csv(trade_log_path, index=False)
summary.to_csv(summary_path, index=False)
heat_df.to_csv(heat_path, index=False)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('PHASE T6 — CAPITAL & EXECUTION ENGINE\n')
    f.write('=' * 60 + '\n')
    f.write(f'Initial capital: {INITIAL_CAPITAL}\n')
    f.write(f'Final capital: {final_capital:.2f}\n')
    f.write(f'Net profit: {net_profit:.2f}\n')
    f.write(f'Return %: {(final_capital / INITIAL_CAPITAL - 1) * 100:.2f}%\n')
    f.write(f'Max drawdown %: {max_dd:.2f}%\n')
    f.write(f'Executed trades: {len(equity_df)}\n')

print('\n[OK] phase_t6_equity_curve.csv')
print('[OK] phase_t6_trade_log.csv')
print('[OK] phase_t6_portfolio_summary.csv')
print('[OK] phase_t6_heat_timeline.csv')
print('[OK] phase_t6_master_report.txt')

print('\nDone.')