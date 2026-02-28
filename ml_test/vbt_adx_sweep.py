"""
vbt_adx_sweep.py

SMA Crossover + RSI Continuation + ADX Trend Strength Filter

Logic:
- LONG:  Fast SMA crosses ABOVE Slow SMA  AND  RSI > 60  AND  ADX > adx_min
- SHORT: Fast SMA crosses BELOW Slow SMA  AND  RSI < 40  AND  ADX > adx_min

ADX confirms the move is happening in a trending market, not a choppy range.
Sweeps ADX minimum threshold (15, 20, 25, 30) on top of the best SMA/RSI combo.

Uses pandas_ta for ADX calculation (not in core VectorBT).
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta

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
df = df.set_index('Datetime').dropna(subset=['Close'])
df = df.loc['2025-01-01':'2025-12-31']
print(f"Loaded {len(df)} bars.")

price  = df['Close']
high   = df['High']
low    = df['Low']

# --------------------------------------------------------------------------
# 2. Compute Base Indicators (fixed: best SMA+RSI combo)
# --------------------------------------------------------------------------
print("Computing indicators...")

# SMA crossover (Fast 40 / Slow 50 — best from previous sweep)
sma_fast = vbt.MA.run(price, window=40).ma
sma_slow = vbt.MA.run(price, window=50).ma
rsi      = vbt.RSI.run(price, window=14).rsi

ma_f_prev = sma_fast.shift(1)
ma_s_prev = sma_slow.shift(1)

# ADX via pandas_ta — needs High, Low, Close
# pandas_ta returns a DataFrame with columns: DMP_14, DMN_14, ADX_14
adx_period = 14
adx_df = ta.adx(high=high, low=low, close=price, length=adx_period)
adx = adx_df[f'ADX_{adx_period}']  # The actual ADX line (trend strength)

print(f"ADX computed. Range: {adx.min():.1f} - {adx.max():.1f}")

# --------------------------------------------------------------------------
# 3. Sweep Parameters
# --------------------------------------------------------------------------
# Fix the best SMA/RSI, only sweep ADX threshold
adx_thresholds = [0, 15, 20, 25, 30, 35]  # 0 = no filter (baseline)

# Also sweep RSI thresholds for comparison
rsi_long_vals  = [55, 60]
rsi_short_vals = [45, 40]

# Build combinations: (adx_min, rsi_long, rsi_short)
from itertools import product
combos = list(product(adx_thresholds, rsi_long_vals, rsi_short_vals))
print(f"\nSweeping {len(combos)} ADX + RSI combinations...")

# --------------------------------------------------------------------------
# 4. Run Each Combo (loop since ADX threshold creates different-shaped masks)
# --------------------------------------------------------------------------
results = []

for adx_min, rsi_l, rsi_s in combos:
    # Build signal masks
    long_entries  = (ma_f_prev <= ma_s_prev) & (sma_fast > sma_slow) & (rsi > rsi_l)
    short_entries = (ma_f_prev >= ma_s_prev) & (sma_fast < sma_slow) & (rsi < rsi_s)

    # ADX filter: only take signal if ADX is above minimum threshold
    if adx_min > 0:
        long_entries  = long_entries  & (adx > adx_min)
        short_entries = short_entries & (adx > adx_min)

    n_long  = long_entries.sum()
    n_short = short_entries.sum()
    total_signals = n_long + n_short

    if total_signals < 10:
        results.append({
            'ADX_Min': adx_min, 'RSI_L': rsi_l, 'RSI_S': rsi_s,
            'Total_Trades': 0, 'Win_Rate_%': np.nan,
            'Total_PnL': np.nan, 'Profit_Factor': np.nan,
            'Long_signals': n_long, 'Short_signals': n_short
        })
        continue

    pf = vbt.Portfolio.from_signals(
        price,
        entries=long_entries,
        short_entries=short_entries,
        upon_opposite_entry='close',
        size=1.0,
        size_type='amount',
        init_cash=100000,
        fees=0.0001,
        sl_stop=0.002,
        tp_stop=0.005,
        freq='15min'
    )

    results.append({
        'ADX_Min':       adx_min,
        'RSI_L':         rsi_l,
        'RSI_S':         rsi_s,
        'Total_Trades':  pf.trades.count(),
        'Win_Rate_%':    round(pf.trades.win_rate() * 100, 1),
        'Total_PnL':     round(pf.total_profit(), 2),
        'Profit_Factor': round(pf.trades.profit_factor(), 3),
        'Long_signals':  n_long,
        'Short_signals': n_short
    })

# --------------------------------------------------------------------------
# 5. Results
# --------------------------------------------------------------------------
summary = pd.DataFrame(results).sort_values('Profit_Factor', ascending=False)

print("\nFull ADX + RSI Confluence Results:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)
print(summary.to_string(index=False))

output_path = 'data/processed/vbt_adx_sweep.csv'
summary.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
