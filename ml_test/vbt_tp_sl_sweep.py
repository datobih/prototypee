"""
vbt_tp_sl_sweep.py

A vectorized TP/SL parameter sweep using the correct VectorBT API.
Locks in the best performing strategy:
  - SMA Fast: 40 / Slow: 150
  - RSI Long: > 60 / Short: < 40

Uses proper entries=long_entries + short_entries=short_entries boolean DataFrames,
NOT the broken direction-string workaround.
"""
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
price = df.set_index('Datetime').dropna(subset=['Close']).loc['2025-10-01':'2025-12-31']['Close']
print(f"Loaded {len(price)} bars.")

# --------------------------------------------------------------------------
# 2. Compute Base Signals (fixed parameters)
# --------------------------------------------------------------------------
print("Computing SMA 40/150 + RSI 60/40 signals...")

sma_fast = vbt.MA.run(price, window=40).ma
sma_slow = vbt.MA.run(price, window=150).ma
rsi      = vbt.RSI.run(price, window=14).rsi

ma_f_prev = sma_fast.shift(1)
ma_s_prev = sma_slow.shift(1)

long_entries  = (ma_f_prev <= ma_s_prev) & (sma_fast > sma_slow) & (rsi > 60)
short_entries = (ma_f_prev >= ma_s_prev) & (sma_fast < sma_slow) & (rsi < 40)

print(f"Longs: {long_entries.sum()} | Shorts: {short_entries.sum()}")

# --------------------------------------------------------------------------
# 3. Define TP/SL Sweep Ranges
# --------------------------------------------------------------------------
sl_values = np.round(np.arange(0.001, 0.0055, 0.0005), 4)   # 0.10% -> 0.50%
tp_values = np.round(np.arange(0.002, 0.0155, 0.001),  4)   # 0.20% -> 1.50%

combinations = [(sl, tp) for sl in sl_values for tp in tp_values]
c_sl, c_tp   = zip(*combinations)
n = len(combinations)
print(f"\nSweeping {n} TP/SL combinations...")

# --------------------------------------------------------------------------
# 4. Tile the 1D signal arrays into [T x N] DataFrames
#    Each column = one (SL, TP) combination
# --------------------------------------------------------------------------
col_idx = pd.MultiIndex.from_tuples(combinations, names=['SL', 'TP'])

long_entries_b  = pd.DataFrame(
    np.tile(long_entries.values,  (n, 1)).T,
    index=price.index, columns=col_idx
)
short_entries_b = pd.DataFrame(
    np.tile(short_entries.values, (n, 1)).T,
    index=price.index, columns=col_idx
)
price_b = pd.DataFrame(
    np.tile(price.values,         (n, 1)).T,
    index=price.index, columns=col_idx
)

exits_b       = pd.DataFrame(False, index=price.index, columns=col_idx)
short_exits_b = pd.DataFrame(False, index=price.index, columns=col_idx)

# --------------------------------------------------------------------------
# 5. Run Vectorized Portfolio Simulation
# --------------------------------------------------------------------------
print("Running portfolio simulation...")
portfolio = vbt.Portfolio.from_signals(
    price_b,
    entries=long_entries_b,
    exits=exits_b,
    short_entries=short_entries_b,
    short_exits=short_exits_b,
    upon_opposite_entry='close',  # Properly close position before flipping
    sl_stop=list(c_sl),
    tp_stop=list(c_tp),
    init_cash=10000,
    fees=0.0001,
    freq='1min'
)

# --------------------------------------------------------------------------
# 6. Compile Results
# --------------------------------------------------------------------------
print("Compiling results...")
summary_df = pd.DataFrame({
    'SL_%':          [round(c[0]*100, 2) for c in combinations],
    'TP_%':          [round(c[1]*100, 2) for c in combinations],
    'Total_Return_%': portfolio.total_return().values * 100,
    'Win_Rate_%':     portfolio.trades.win_rate().values * 100,
    'Total_Trades':   portfolio.trades.count().values,
    'Profit_Factor':  portfolio.trades.profit_factor().values,
}, index=[f"SL{round(c[0]*100,2)}_TP{round(c[1]*100,2)}" for c in combinations])

summary_df = summary_df.dropna()
summary_df = summary_df.sort_values('Total_Return_%', ascending=False)

print("\nTop 15 TP/SL Combinations:")
pd.set_option('display.max_columns', None)
print(summary_df.head(15))

output_path = 'data/processed/vbt_tp_sl_sweep.csv'
summary_df.to_csv(output_path)
print(f"\nFull sweep saved to: {output_path}")
