"""
vbt_breakout_sweep.py

A vectorized parameter sweep for Gold (XAUUSD) M1 data combining
Bollinger Bands breakout with RSI momentum confirmation.

Logic:
- LONG: Close crosses above Upper BB AND RSI > rsi_long_threshold
- SHORT: Close crosses below Lower BB AND RSI < rsi_short_threshold

This script sweeps across different BB windows, BB standard deviations,
and RSI thresholds simultaneously to find robust parameters.
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

# Take 3 months for a solid sample size
df = df.loc['2025-10-01':'2025-12-31'] 
print(f"Loaded {len(df)} 1-minute bars for backtesting.")

price = df['Close']

# --------------------------------------------------------------------------
# 2. Define Parameter Ranges to Sweep
# --------------------------------------------------------------------------
# Bollinger Bands
bb_windows   = [20, 30, 50]
bb_alphas    = [1.5, 2.0, 2.5]  # Std Dev multipliers

# RSI Thresholds for momentum filter
rsi_long_thresholds  = [55, 60, 65]
rsi_short_thresholds = [45, 40, 35]

print(f"\nSweeping {len(bb_windows) * len(bb_alphas) * len(rsi_long_thresholds) * len(rsi_short_thresholds)} parameter combinations...")

# To perfectly align multi-index broadcasting in vbt, we build all permutations
from itertools import product
combos = list(product(bb_windows, bb_alphas, rsi_long_thresholds, rsi_short_thresholds))

c_bb_w, c_bb_a, c_rsi_l, c_rsi_s = zip(*combos)

# --------------------------------------------------------------------------
# 3. Create Vectorized Indicators
# --------------------------------------------------------------------------
print("Computing indicators...")
# Bollinger Bands matrix
bb = vbt.BBANDS.run(
    price, 
    window=list(c_bb_w), 
    alpha=list(c_bb_a)
)

# RSI matrix - we need RSI to have the exact same shape/index as BB
# Since RSI only depends on price (window=14 hardcoded), we can compute it once 
rsi = vbt.RSI.run(price, window=14).rsi

# Instead of vbt.broadcast (which is picky about shape mismatches like [88780,] vs [81,]),
# we just construct the perfectly sized DataFrames manually using numpy tile
rsi_df = pd.DataFrame(
    np.tile(rsi.values, (len(combos), 1)).T, 
    index=price.index, 
    columns=pd.MultiIndex.from_tuples(combos)
)

# Broadcast the scalar threshold parameters into DataFrames matching rsi_df shape
rsi_l_df = pd.DataFrame(
    np.tile(c_rsi_l, (len(price), 1)),
    index=price.index,
    columns=rsi_df.columns
)

rsi_s_df = pd.DataFrame(
    np.tile(c_rsi_s, (len(price), 1)),
    index=price.index,
    columns=rsi_df.columns
)

bb_upper = bb.upper.copy()
bb_lower = bb.lower.copy()
bb_upper.columns = rsi_df.columns
bb_lower.columns = rsi_df.columns

price_df = pd.DataFrame(
    np.tile(price.values, (len(combos), 1)).T, 
    index=price.index, 
    columns=rsi_df.columns
)

# --------------------------------------------------------------------------
# 4. Generate Trade Signals
# --------------------------------------------------------------------------
print("Generating signals...")
# Previous bar close to check crossovers
price_prev = price_df.shift(1)
bb_u_prev  = bb_upper.shift(1)
bb_l_prev  = bb_lower.shift(1)

# LONG: Price crosses above Upper BB AND RSI > threshold
long_entries = (price_prev <= bb_u_prev) & (price_df > bb_upper)
long_entries = long_entries & (rsi_df > rsi_l_df)

# SHORT: Price crosses below Lower BB AND RSI < threshold
short_entries = (price_prev >= bb_l_prev) & (price_df < bb_lower)
short_entries = short_entries & (rsi_df < rsi_s_df)

# We don't use indicator exits, we rely strictly on TP/SL
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
    fees=0.0001,   # 1 pip spread/commission equivalent
    sl_stop=0.002, # 0.2% strict stop loss
    tp_stop=0.005, # 0.5% take profit
    freq='1min'
)

# --------------------------------------------------------------------------
# 6. Analyze Results
# --------------------------------------------------------------------------
print("Compiling results...")
stats = portfolio.total_return() * 100 # % return
trades = portfolio.trades.count()
win_rate = portfolio.trades.win_rate() * 100
pf = portfolio.trades.profit_factor()

summary_df = pd.DataFrame({
    'Total_Return_%': stats.values,
    'Win_Rate_%': win_rate.values,
    'Total_Trades': trades.values,
    'Profit_Factor': pf.values
}, index=[f"BBw{c[0]}_BBa{c[1]}_RSI_L{c[2]}_RSI_S{c[3]}" for c in combos])

# Drop NaN profit factors (0 trades)
summary_df = summary_df.dropna()

# Sort by Profit Factor and Total Return
summary_df = summary_df.sort_values(by=['Profit_Factor', 'Total_Return_%'], ascending=[False, False])

print("\nTop 15 Breakout Parameter Combinations:")
pd.set_option('display.max_columns', None)
print(summary_df.head(15))

# Save the full sweep data
output_path = 'data/processed/vbt_bb_breakout_sweep.csv'
summary_df.to_csv(output_path)
print(f"\nFull sweep matrix saved to: {output_path}")

# Pick the very best performing setup and plot the equity curve
best_params = summary_df.index[0]
best_combo_idx = list(summary_df.index).index(best_params)
print(f"\nBest Parameters: {best_params}")
