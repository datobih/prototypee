"""
vbt_ema_sweep.py

SMA Crossover + RSI Continuation + EMA Trend Direction Filter

Logic:
- LONG:  Fast SMA crosses ABOVE Slow SMA  AND  RSI > rsi_l  AND  Price > EMA(ema_period)
- SHORT: Fast SMA crosses BELOW Slow SMA  AND  RSI < rsi_s  AND  Price < EMA(ema_period)

EMA acts as a trend direction gate:
- Only take longs when price is ABOVE the long-term EMA (uptrend)
- Only take shorts when price is BELOW the long-term EMA (downtrend)
- Trades against the dominant trend are skipped entirely

Sweeps EMA period (50, 100, 200, 500) and RSI thresholds.
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
from itertools import product

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
price = df.set_index('Datetime').dropna(subset=['Close']).loc['2025-01-01':'2025-12-31']['Close']
print(f"Loaded {len(price)} bars.")

# --------------------------------------------------------------------------
# 2. Compute Base Indicators
# --------------------------------------------------------------------------
print("Computing indicators...")

sma_fast = vbt.MA.run(price, window=40).ma
sma_slow = vbt.MA.run(price, window=50).ma
rsi      = vbt.RSI.run(price, window=14).rsi

ma_f_prev = sma_fast.shift(1)
ma_s_prev = sma_slow.shift(1)

# Precompute all EMA periods we'll sweep
ema_periods = [50, 100, 150, 200, 300, 500]
emas = {p: vbt.MA.run(price, window=p, ewm=True).ma for p in ema_periods}

# RSI configs
rsi_configs = [(60, 40), (55, 45), (60, 45)]

# --------------------------------------------------------------------------
# 3. Baseline — no EMA filter
# --------------------------------------------------------------------------
combos = [('None', rsi_l, rsi_s) for _, rsi_l, rsi_s in [(0, *rc) for rc in rsi_configs]]
combos += [(p, rsi_l, rsi_s) for p, (rsi_l, rsi_s) in product(ema_periods, rsi_configs)]
print(f"Sweeping {len(combos)} EMA + RSI combinations...")

# --------------------------------------------------------------------------
# 4. Run Each Combo
# --------------------------------------------------------------------------
results = []

for ema_period, rsi_l, rsi_s in combos:
    long_entries  = (ma_f_prev <= ma_s_prev) & (sma_fast > sma_slow) & (rsi > rsi_l)
    short_entries = (ma_f_prev >= ma_s_prev) & (sma_fast < sma_slow) & (rsi < rsi_s)

    if ema_period != 'None':
        ema = emas[ema_period]
        long_entries  = long_entries  & (price > ema)   # Only long when price above EMA
        short_entries = short_entries & (price < ema)   # Only short when price below EMA

    total_signals = long_entries.sum() + short_entries.sum()

    if total_signals < 5:
        results.append({'EMA_Period': ema_period, 'RSI_L': rsi_l, 'RSI_S': rsi_s,
                        'Total_Trades': 0, 'Win_Rate_%': np.nan,
                        'Total_PnL': np.nan, 'Profit_Factor': np.nan,
                        'Long_Trades': 0, 'Short_Trades': 0})
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

    trades_df = pf.trades.records_readable
    long_pnl  = trades_df[trades_df['Direction'] == 'Long']['PnL'].sum()  if not trades_df.empty else 0
    short_pnl = trades_df[trades_df['Direction'] == 'Short']['PnL'].sum() if not trades_df.empty else 0

    results.append({
        'EMA_Period':    ema_period,
        'RSI_L':         rsi_l,
        'RSI_S':         rsi_s,
        'Total_Trades':  pf.trades.count(),
        'Win_Rate_%':    round(pf.trades.win_rate() * 100, 1),
        'Total_PnL':     round(pf.total_profit(), 2),
        'Profit_Factor': round(pf.trades.profit_factor(), 3),
        'Long_PnL':      round(long_pnl, 2),
        'Short_PnL':     round(short_pnl, 2),
    })

# --------------------------------------------------------------------------
# 5. Results
# --------------------------------------------------------------------------
summary = pd.DataFrame(results).sort_values('Profit_Factor', ascending=False)

print("\nEMA Trend Filter Results (sorted by Profit Factor):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
print(summary.to_string(index=False))

output_path = 'data/processed/vbt_ema_sweep.csv'
summary.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
