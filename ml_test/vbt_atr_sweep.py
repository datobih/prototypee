"""
vbt_atr_sweep.py

SMA Crossover + RSI Continuation + ATR Volatility Filter

Logic:
- LONG:  Fast SMA crosses ABOVE Slow SMA  AND  RSI > 60  AND  ATR > atr_min
- SHORT: Fast SMA crosses BELOW Slow SMA  AND  RSI < 40  AND  ATR > atr_min

ATR (Average True Range) filters for minimum volatility.
If ATR is too low, the market is too flat to realistically reach TP before SL.

Sweeps ATR minimum threshold as a percentile of the ATR distribution
to keep it scale-invariant as gold price changes over time.
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta

# --------------------------------------------------------------------------
# 1. Load Data
# --------------------------------------------------------------------------
print("Loading 5-minute price data...")
df = pd.read_csv(
    'data/raw/XAUUSD5.csv',
    sep='\t',
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df = df.set_index('Datetime').dropna(subset=['Close'])
df = df.loc['2025-01-01':'2025-12-31']
print(f"Loaded {len(df)} bars.")

price = df['Close']
high  = df['High']
low   = df['Low']

# --------------------------------------------------------------------------
# 2. Compute Base Indicators
# --------------------------------------------------------------------------
print("Computing indicators...")

sma_fast = vbt.MA.run(price, window=40).ma
sma_slow = vbt.MA.run(price, window=50).ma
rsi      = vbt.RSI.run(price, window=14).rsi

ma_f_prev = sma_fast.shift(1)
ma_s_prev = sma_slow.shift(1)

# ATR via pandas_ta
atr = ta.atr(high=high, low=low, close=price, length=14)

# Also compute ATR as % of price (normalised) so thresholds work regardless of price level
atr_pct = (atr / price) * 100  # ATR as % of current price

print(f"ATR range: {atr.min():.2f} - {atr.max():.2f} (raw $)")
print(f"ATR % range: {atr_pct.min():.3f}% - {atr_pct.max():.3f}%")
print(f"ATR % median: {atr_pct.median():.3f}%  |  mean: {atr_pct.mean():.3f}%")

# --------------------------------------------------------------------------
# 3. Sweep ATR % Minimums
# --------------------------------------------------------------------------
# We filter on ATR as % of price — scale invariant
# Low ATR% = dead market, High ATR% = volatile / news driven
# Thresholds tested: 0 = no filter, then increasing minimums
atr_pct_mins = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

# Also sweep RSI thresholds
rsi_configs = [(60, 40), (55, 45), (60, 45)]

from itertools import product
combos = list(product(atr_pct_mins, rsi_configs))
print(f"\nSweeping {len(combos)} ATR + RSI combinations...")

# --------------------------------------------------------------------------
# 4. Run Each Combo
# --------------------------------------------------------------------------
results = []

for atr_min_pct, (rsi_l, rsi_s) in combos:
    long_entries  = (ma_f_prev <= ma_s_prev) & (sma_fast > sma_slow) & (rsi > rsi_l)
    short_entries = (ma_f_prev >= ma_s_prev) & (sma_fast < sma_slow) & (rsi < rsi_s)

    if atr_min_pct > 0:
        long_entries  = long_entries  & (atr_pct > atr_min_pct)
        short_entries = short_entries & (atr_pct > atr_min_pct)

    total_signals = long_entries.sum() + short_entries.sum()
    if total_signals < 10:
        results.append({
            'ATR_%_Min': atr_min_pct, 'RSI_L': rsi_l, 'RSI_S': rsi_s,
            'Total_Trades': 0, 'Win_Rate_%': np.nan,
            'Total_PnL': np.nan, 'Profit_Factor': np.nan,
            'Signals_Filtered_Out': '(too few signals)'
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
        freq='5min'
    )

    pct_filtered = 100 * (1 - total_signals / (long_entries.notna().sum() + short_entries.notna().sum()))

    results.append({
        'ATR_%_Min':     atr_min_pct,
        'RSI_L':         rsi_l,
        'RSI_S':         rsi_s,
        'Total_Trades':  pf.trades.count(),
        'Win_Rate_%':    round(pf.trades.win_rate() * 100, 1),
        'Total_PnL':     round(pf.total_profit(), 2),
        'Profit_Factor': round(pf.trades.profit_factor(), 3),
        'Signals_Filtered_Out': f"{total_signals} signals"
    })

# --------------------------------------------------------------------------
# 5. Results
# --------------------------------------------------------------------------
summary = pd.DataFrame(results).sort_values('Profit_Factor', ascending=False)

print("\nATR + RSI Confluence Results (sorted by Profit Factor):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)
print(summary.to_string(index=False))

output_path = 'data/processed/vbt_atr_sweep.csv'
summary.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
