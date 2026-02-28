"""
vbt_indicator_sweep.py

A vectorized backtesting script using VectorBT to sweep through combinations of
indicators (like SMA Crossovers and RSI) on Gold (XAUUSD) 1-minute data.

This uses pure pandas/numpy arrays so we can test thousands of parameter
combinations simultaneously in seconds.
"""
import os
import pandas as pd
import numpy as np
import vectorbt as vbt

# --------------------------------------------------------------------------
# 1. Load Data
# --------------------------------------------------------------------------
print("Loading raw 1-minute price data...")
df = pd.read_csv(
    'data/raw/XAUUSD1.csv',
    sep='\t',
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df = df.set_index('Datetime')
df = df.dropna(subset=['Close'])

# Take 2 months to keep RAM usage reasonable for massive sweeps
df = df.loc['2025-11-01':'2025-12-31'] 
print(f"Loaded {len(df)} 1-minute bars for backtesting.")

price = df['Close']

# --------------------------------------------------------------------------
# 2. Define Parameter Ranges to Sweep
# --------------------------------------------------------------------------
# To properly broadcast in vectorbt, we need all combinations of fast/slow
fast_windows = np.arange(10, 51, 10)   # [10, 20, 30, 40, 50]
slow_windows = np.arange(50, 201, 50)  # [50, 100, 150, 200]

# Generate all valid combinations where fast < slow
combinations = [(f, s) for f in fast_windows for s in slow_windows if f < s]
fast_combos, slow_combos = zip(*combinations)

print(f"\nSweeping {len(combinations)} valid SMA parameter combinations...")

# --------------------------------------------------------------------------
# 3. Create Vectorized Indicators
# --------------------------------------------------------------------------
# Run `vbt.MA.run` with the exact matched arrays of combinations
sma_fast = vbt.MA.run(price, window=list(fast_combos), short_name='fast')
sma_slow = vbt.MA.run(price, window=list(slow_combos), short_name='slow')

# --------------------------------------------------------------------------
# 4. Generate Trade Signals
# --------------------------------------------------------------------------
# Now the indexes match perfectly, so they can cross
long_entries = sma_fast.ma_crossed_above(sma_slow)
short_entries = sma_fast.ma_crossed_below(sma_slow)

# Rely strictly on TP/SL for exits
exits = pd.DataFrame(False, index=long_entries.index, columns=long_entries.columns)
short_exits = pd.DataFrame(False, index=short_entries.index, columns=short_entries.columns)

# --------------------------------------------------------------------------
# 5. Run Vectorized Portfolio Simulation
# --------------------------------------------------------------------------
print("Running portfolio simulation...")
portfolio = vbt.Portfolio.from_signals(
    price,
    entries=long_entries,
    exits=exits,
    short_entries=short_entries,
    short_exits=short_exits,
    init_cash=10000,
    fees=0.0001, # Simulating spread/commission
    sl_stop=0.002, # 0.2% strict stop loss
    tp_stop=0.005, # 0.5% take profit
    freq='1min'
)

# --------------------------------------------------------------------------
# 6. Analyze Results
# --------------------------------------------------------------------------
stats = portfolio.total_return() * 100 # % return
trades = portfolio.trades.count()
win_rate = portfolio.trades.win_rate() * 100

summary_df = pd.DataFrame({
    'Total_Return_%': stats,
    'Win_Rate_%': win_rate,
    'Total_Trades': trades
})

# Flatten the multi-index for cleaner CSV output
summary_df.index = [f"Fast_{idx[0]}_Slow_{idx[1]}" for idx in summary_df.index]

# Sort by highest return
summary_df = summary_df.sort_values(by='Total_Return_%', ascending=False)

print("\nTop 10 Parameter Combinations (SMA Fast vs Slow):")
pd.set_option('display.max_columns', None)
print(summary_df.head(10))

# Save the full sweep data
output_path = 'data/processed/vbt_sma_sweep.csv'
summary_df.to_csv(output_path)
print(f"\nFull sweep matrix saved to: {output_path}")
