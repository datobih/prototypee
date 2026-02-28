"""
vbt_best_combo.py

Single-combo backtest for: Fast40_Slow50_RSI_L55_S50
SMA crossover + RSI continuation with proper SL/TP using High/Low.

Reports detailed stats: PnL, Win Rate, Profit Factor, Max Drawdown,
worst single trade, max losing streak, and min capital needed.
"""
import pandas as pd
import numpy as np
import vectorbt as vbt

# --------------------------------------------------------------------------
# 1. Load Data
# --------------------------------------------------------------------------
print("Loading 15-minute price data...")
df = pd.read_csv(
    'data/raw/XAUUSD15.csv',
    sep='\t',
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df = df.set_index('Datetime').dropna(subset=['Close']).loc['2023-02-01':'2025-10-27']
print(f"Loaded {len(df)} bars.")

price = df['Close']
high  = df['High']
low   = df['Low']
open_ = df['Open']

# --------------------------------------------------------------------------
# 2. Compute Indicators — Fast40 / Slow50 / RSI14
# --------------------------------------------------------------------------
print("Computing indicators...")
sma_fast = vbt.MA.run(price, window=40).ma
sma_slow = vbt.MA.run(price, window=50).ma
rsi      = vbt.RSI.run(price, window=14).rsi

ma_f_prev = sma_fast.shift(1)
ma_s_prev = sma_slow.shift(1)

# --------------------------------------------------------------------------
# 3. Generate Signals — RSI L55 / S50
# --------------------------------------------------------------------------
long_entries  = (ma_f_prev <= ma_s_prev) & (sma_fast > sma_slow) & (rsi > 55)
short_entries = (ma_f_prev >= ma_s_prev) & (sma_fast < sma_slow) & (rsi < 50)

print(f"Long signals:  {long_entries.sum()}")
print(f"Short signals: {short_entries.sum()}")

# --------------------------------------------------------------------------
# 4. Portfolio Simulation
# --------------------------------------------------------------------------
print("Running portfolio simulation...")
pf = vbt.Portfolio.from_signals(
    price,
    open=open_,
    high=high,
    low=low,
    entries=long_entries,
    short_entries=short_entries,
    upon_opposite_entry='close',
    size=1.0,
    size_type='amount',
    init_cash=1000000,
    fees=0.0001,
    sl_stop=0.002,
    tp_stop=0.005,
    freq='15min'
)

# --------------------------------------------------------------------------
# 5. Detailed Results
# --------------------------------------------------------------------------
trades = pf.trades.records_readable

# Equity curve drawdown
equity = pf.value()
running_max = equity.cummax()
drawdown = equity - running_max
max_dd = drawdown.min()

# Max losing streak
streak, max_streak = 0, 0
for pnl in trades['PnL']:
    if pnl < 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0

# Per-direction breakdown
long_trades  = trades[trades['Direction'] == 'Long']
short_trades = trades[trades['Direction'] == 'Short']

print("\n" + "=" * 60)
print("BACKTEST RESULTS: Fast40_Slow50_RSI_L55_S50")
print("=" * 60)

print(f"\n--- Overall ---")
print(f"  Total PnL:           ${pf.total_profit():.2f}")
print(f"  Total Trades:        {pf.trades.count()}")
print(f"  Win Rate:            {pf.trades.win_rate()*100:.1f}%")
print(f"  Profit Factor:       {pf.trades.profit_factor():.3f}")
print(f"  Max Drawdown:        ${max_dd:.2f}")
print(f"  Worst Single Trade:  ${trades['PnL'].min():.2f}")
print(f"  Best Single Trade:   ${trades['PnL'].max():.2f}")
print(f"  Avg Win:             ${trades[trades['PnL']>0]['PnL'].mean():.2f}")
print(f"  Avg Loss:            ${trades[trades['PnL']<0]['PnL'].mean():.2f}")
print(f"  Max Losing Streak:   {max_streak} trades")

print(f"\n--- Long Trades ---")
print(f"  Count:     {len(long_trades)}")
print(f"  Win Rate:  {(long_trades['PnL']>0).mean()*100:.1f}%")
print(f"  PnL:       ${long_trades['PnL'].sum():.2f}")

print(f"\n--- Short Trades ---")
print(f"  Count:     {len(short_trades)}")
print(f"  Win Rate:  {(short_trades['PnL']>0).mean()*100:.1f}%")
print(f"  PnL:       ${short_trades['PnL'].sum():.2f}")

print(f"\n--- Exit Type Breakdown ---")
exit_col = [c for c in trades.columns if 'exit' in c.lower() or 'status' in c.lower()]
if exit_col:
    for exit_type, group in trades.groupby(exit_col[0]):
        print(f"  {exit_type}: {len(group)} trades, avg PnL ${group['PnL'].mean():.2f}")
else:
    print(f"  (Exit type column not found. Columns: {list(trades.columns)})")

print(f"\n--- Min Capital Guide (0.01 lot at 1:500 leverage) ---")
min_cap = abs(max_dd) * 1.5 + 5
print(f"  Max Drawdown (1 oz = 0.01 lot): ${abs(max_dd):.2f}")
print(f"  Min Capital (1.5x buffer + margin): ${min_cap:.2f}")

# Save trade log
trade_log_path = 'data/processed/vbt_best_combo_trades.csv'
trades.to_csv(trade_log_path, index=False)
print(f"\nTrade log saved to: {trade_log_path}")
