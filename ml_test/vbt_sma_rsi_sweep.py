"""
vbt_sma_rsi_sweep.py

Vectorized parameter sweep: SMA Crossover + RSI Continuation (Momentum Filter)

Logic:
- LONG:  Fast SMA crosses ABOVE Slow SMA  AND  RSI > threshold (momentum up)
- SHORT: Fast SMA crosses BELOW Slow SMA  AND  RSI < threshold (momentum down)

Uses proper VectorBT API:
- entries / short_entries (separate boolean arrays, no direction hack)
- upon_opposite_entry='close' (no overlapping positions)
- size=1.0 (fixed 1-unit sizing, no compounding illusion)
"""
import pandas as pd
import numpy as np
import vectorbt as vbt

# --------------------------------------------------------------------------
# 1. Load Data
# --------------------------------------------------------------------------
print("Loading raw 1-minute price data...")
df = pd.read_csv(
    'data/raw/VIX.csv',
    sep='\t',
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df = df.set_index('Datetime').dropna(subset=['Close']).loc['2025-05-01':'2025-05-14']
print(f"Loaded {len(df)} bars (2025).")

price = df['Close']
high  = df['High']
low   = df['Low']
open_ = df['Open']

# --------------------------------------------------------------------------
# 2. Define Parameter Ranges
# --------------------------------------------------------------------------
fast_windows = np.arange(10, 41, 10)   # [10, 20, 30, 40]
slow_windows = np.arange(50, 151, 50)  # [50, 100, 150]
rsi_long_thresholds  = [50, 55, 60]
rsi_short_thresholds = [50, 45, 40]

combinations = [
    (f, s, rl, rs)
    for f in fast_windows for s in slow_windows if f < s
    for rl in rsi_long_thresholds for rs in rsi_short_thresholds
]
n = len(combinations)
c_fast, c_slow, c_rsi_l, c_rsi_s = zip(*combinations)
print(f"\nSweeping {n} parameter combinations...")

# --------------------------------------------------------------------------
# 3. Compute Indicators
# --------------------------------------------------------------------------
print("Computing indicators...")

sma_fast = vbt.MA.run(price, window=list(c_fast), short_name='fast')
sma_slow = vbt.MA.run(price, window=list(c_slow), short_name='slow')
rsi = vbt.RSI.run(price, window=14).rsi

# Build aligned DataFrames
col_idx = pd.MultiIndex.from_tuples(combinations)

rsi_df = pd.DataFrame(np.tile(rsi.values, (n, 1)).T, index=price.index, columns=col_idx)
rsi_l_df = pd.DataFrame(np.tile(c_rsi_l, (len(price), 1)), index=price.index, columns=col_idx)
rsi_s_df = pd.DataFrame(np.tile(c_rsi_s, (len(price), 1)), index=price.index, columns=col_idx)

ma_f_df = sma_fast.ma.copy()
ma_s_df = sma_slow.ma.copy()
ma_f_df.columns = col_idx
ma_s_df.columns = col_idx

# --------------------------------------------------------------------------
# 4. Generate Signals — Continuation Logic
# --------------------------------------------------------------------------
print("Generating signals...")

ma_f_prev = ma_f_df.shift(1)
ma_s_prev = ma_s_df.shift(1)

# LONG: Fast crosses above Slow AND RSI confirms upward momentum
long_entries = (ma_f_prev <= ma_s_prev) & (ma_f_df > ma_s_df) & (rsi_df > rsi_l_df)

# SHORT: Fast crosses below Slow AND RSI confirms downward momentum
short_entries = (ma_f_prev >= ma_s_prev) & (ma_f_df < ma_s_df) & (rsi_df < rsi_s_df)

exits       = pd.DataFrame(False, index=price.index, columns=col_idx)
short_exits = pd.DataFrame(False, index=price.index, columns=col_idx)

# --------------------------------------------------------------------------
# 5. Portfolio Simulation — Honest Settings
# --------------------------------------------------------------------------
print("Running portfolio simulation...")
portfolio = vbt.Portfolio.from_signals(
    price,
    open=open_,
    high=high,                     # SL/TP check against intrabar highs
    low=low,                       # SL/TP check against intrabar lows
    entries=long_entries,
    exits=exits,
    short_entries=short_entries,
    short_exits=short_exits,
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
# 6. Results — with REAL dollar drawdown per combo
# --------------------------------------------------------------------------
print("Compiling results...")

# Portfolio value over time gives us real equity curve per column
value = portfolio.value()  # shape: [T x N_combos]

# Compute real $ drawdown per combo from the equity curve
max_dd_dollars = []
for col in value.columns:
    equity = value[col]
    running_max = equity.cummax()
    drawdown = equity - running_max  # negative values = drawdown
    max_dd_dollars.append(drawdown.min())  # worst drawdown in $

summary_df = pd.DataFrame({
    'Total_PnL':          portfolio.total_profit().values,
    'Win_Rate_%':         portfolio.trades.win_rate().values * 100,
    'Total_Trades':       portfolio.trades.count().values,
    'Profit_Factor':      portfolio.trades.profit_factor().values,
    'Max_Drawdown_$':     max_dd_dollars,
    'Min_Capital_0.01lot': [round(abs(dd) * 1.5 + 5, 2) for dd in max_dd_dollars],  # size=1oz IS 0.01 lot. 1.5x buffer + $5 margin
}, index=[f"Fast{c[0]}_Slow{c[1]}_RSI_L{c[2]}_S{c[3]}" for c in combinations])

summary_df = summary_df.dropna()
summary_df = summary_df.sort_values('Total_PnL', ascending=False)

print("\nTop 15 SMA + RSI Continuation Combinations (Flat 1oz Sizing):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
print(summary_df.head(15))

print("\n--- Min Capital Guide (for 0.01 lot / micro lot at 1:500 leverage) ---")
top = summary_df.head(5)
for name, row in top.iterrows():
    print(f"  {name}: Max DD ${abs(row['Max_Drawdown_$']):.2f} -> Min capital at 0.01 lot: ${row['Min_Capital_0.01lot']:.2f}")

output_path = 'data/processed/vbt_sma_rsi_cont_sweep.csv'
summary_df.to_csv(output_path)
print(f"\nFull sweep saved to: {output_path}")

