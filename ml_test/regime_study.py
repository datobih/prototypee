"""
regime_study.py
===============
Characterize each discovered regime by joining regime labels with
raw OHLCV data and outcome labels.

Outputs per-regime:
  - Outcome rates (move %, long %, short %)
  - Avg volatility (hl_range, TickVol)
  - Session distribution
  - Regime duration runs
  - Transition patterns
  - Profitability estimate (expected value per trade)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG — must match hedging_strategy_strict.py
# =============================================================================
RAW_DATA    = 'data/raw/XAUUSD1.csv'
REGIME_CSV  = 'data/processed/XAUUSD1_regimes.csv'
TARGET      = 3.0    # dollar TP
STOP        = 1.5    # dollar SL
HORIZON     = 30     # bars to look ahead
STRIDE      = 5      # stride used in engineer_features.py

# =============================================================================
# LOAD RAW DATA + REGIME LABELS
# =============================================================================
print('=' * 70)
print('REGIME STUDY — Per-regime characterization')
print('=' * 70)

print('\nLoading raw data...')
df = pd.read_csv(RAW_DATA, sep='\t',
                 names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'TickVol']].copy()
df['hl_range'] = df['High'] - df['Low']
print(f'Raw bars: {len(df):,}')

print('Loading regime labels...')
regimes = pd.read_csv(REGIME_CSV, index_col=0, parse_dates=True)
print(f'Regime labels: {len(regimes):,} bars, {regimes["regime"].nunique()} regimes')

# Join regime labels to raw data (forward-fill stride gaps)
df = df.join(regimes[['regime', 'regime_name']], how='left')
df['regime'] = df['regime'].ffill()
df['regime_name'] = df['regime_name'].ffill()
df = df.dropna(subset=['regime'])
df['regime'] = df['regime'].astype(int)
print(f'Bars with regime labels: {len(df):,}')

# =============================================================================
# LABEL OUTCOMES (vectorised where possible, loop for SL/TP logic)
# =============================================================================
print(f'\nLabeling outcomes ({HORIZON} bars, ${TARGET} TP, ${STOP} SL)...')

outcomes = np.zeros(len(df), dtype=int)
close = df['Close'].values
high = df['High'].values
low = df['Low'].values

for i in range(len(df) - HORIZON):
    if i % 50000 == 0:
        print(f'  {i:,}/{len(df) - HORIZON:,}...')
    entry = close[i]
    # Check LONG
    long_hit = False
    for j in range(i + 1, i + HORIZON + 1):
        if high[j] >= entry + TARGET:
            long_hit = True
            break
        if low[j] <= entry - STOP:
            break
    # Check SHORT
    short_hit = False
    if not long_hit:
        for j in range(i + 1, i + HORIZON + 1):
            if low[j] <= entry - TARGET:
                short_hit = True
                break
            if high[j] >= entry + STOP:
                break
    outcomes[i] = 1 if long_hit else (2 if short_hit else 0)

df = df.iloc[:len(df) - HORIZON].copy()
df['outcome'] = outcomes[:len(df)]
print(f'Labeled {len(df):,} bars')

# Add session labels
hour = df.index.hour
df['session'] = 'LATE'
df.loc[(hour >= 0) & (hour < 7), 'session'] = 'ASIAN'
df.loc[(hour >= 7) & (hour < 13), 'session'] = 'LONDON'
df.loc[(hour >= 13) & (hour < 17), 'session'] = 'NY_OVERLAP'
df.loc[(hour >= 17) & (hour < 20), 'session'] = 'NY'

regime_ids = sorted(df['regime'].unique())
n_regimes = len(regime_ids)

# =============================================================================
# TRAIN / TEST SPLIT — evaluate outcomes out-of-sample only
# Regimes were learned on all data, but outcome stats use last 40%
# =============================================================================
TRAIN_FRAC = 0.60
split_idx = int(len(df) * TRAIN_FRAC)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]
print(f'\nTrain/Test split: {len(df_train):,} train | {len(df_test):,} test (last {100-TRAIN_FRAC*100:.0f}%)')
print('All outcome stats below are OUT-OF-SAMPLE (test set only)\n')

# Use test set for all evaluations below
df = df_test

# =============================================================================
# 1. OUTCOME RATES PER REGIME
# =============================================================================
print('\n' + '=' * 70)
print('1. OUTCOME RATES PER REGIME')
print('=' * 70)
print(f'{"Regime":<22} {"Count":>8} {"Move%":>7} {"Long%":>7} {"Short%":>8} {"NoMove%":>8}')
print('-' * 65)

for r in regime_ids:
    sub = df[df['regime'] == r]
    name = sub['regime_name'].iloc[0]
    n = len(sub)
    move = (sub['outcome'] != 0).mean() * 100
    long = (sub['outcome'] == 1).mean() * 100
    short = (sub['outcome'] == 2).mean() * 100
    no_move = (sub['outcome'] == 0).mean() * 100
    print(f'{name:<22} {n:>8,} {move:>6.1f}% {long:>6.1f}% {short:>7.1f}% {no_move:>7.1f}%')

# Overall baseline
move_all = (df['outcome'] != 0).mean() * 100
print(f'{"OVERALL":<22} {len(df):>8,} {move_all:>6.1f}%')

# =============================================================================
# 2. EXPECTED VALUE PER REGIME
# =============================================================================
print('\n' + '=' * 70)
print('2. EXPECTED VALUE (EV) PER REGIME')
print(f'   Win = +${TARGET}, Loss = -${STOP}')
print('=' * 70)
print(f'{"Regime":<22} {"Trades":>8} {"WinRate":>8} {"EV/trade":>10} {"Total EV":>10} {"Verdict":>10}')
print('-' * 72)

regime_ev = {}
for r in regime_ids:
    sub = df[df['regime'] == r]
    name = sub['regime_name'].iloc[0]
    n = len(sub)
    wins = (sub['outcome'] != 0).sum()
    wr = wins / n
    ev = wr * TARGET - (1 - wr) * STOP
    total_ev = ev * n
    regime_ev[r] = {'name': name, 'ev': ev, 'wr': wr, 'n': n, 'total_ev': total_ev}
    verdict = 'TRADE' if ev > 0 else 'AVOID'
    print(f'{name:<22} {n:>8,} {wr:>7.1%} {ev:>9.4f} {total_ev:>10.1f} {verdict:>10}')

# =============================================================================
# 3. VOLATILITY PROFILE PER REGIME
# =============================================================================
print('\n' + '=' * 70)
print('3. VOLATILITY PROFILE PER REGIME')
print('=' * 70)
print(f'{"Regime":<22} {"AvgRange":>10} {"StdRange":>10} {"AvgTickVol":>12} {"StdTickVol":>12}')
print('-' * 70)

for r in regime_ids:
    sub = df[df['regime'] == r]
    name = sub['regime_name'].iloc[0]
    print(f'{name:<22} {sub["hl_range"].mean():>10.3f} {sub["hl_range"].std():>10.3f} '
          f'{sub["TickVol"].mean():>12.1f} {sub["TickVol"].std():>12.1f}')

# =============================================================================
# 4. SESSION DISTRIBUTION PER REGIME
# =============================================================================
print('\n' + '=' * 70)
print('4. SESSION DISTRIBUTION PER REGIME')
print('=' * 70)

sessions = ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']
header = f'{"Regime":<22}' + ''.join(f'{s:>12}' for s in sessions)
print(header)
print('-' * (22 + 12 * len(sessions)))

for r in regime_ids:
    sub = df[df['regime'] == r]
    name = sub['regime_name'].iloc[0]
    row = f'{name:<22}'
    for s in sessions:
        pct = (sub['session'] == s).mean() * 100
        row += f'{pct:>11.1f}%'
    print(row)

# =============================================================================
# 5. REGIME DURATION ANALYSIS (consecutive runs)
# =============================================================================
print('\n' + '=' * 70)
print('5. REGIME DURATION (consecutive bar runs)')
print('=' * 70)

regime_col = df['regime'].values
changes = np.where(np.diff(regime_col) != 0)[0] + 1
starts = np.concatenate([[0], changes])
ends = np.concatenate([changes, [len(regime_col)]])
run_regimes = regime_col[starts]
run_lengths = ends - starts

print(f'{"Regime":<22} {"Runs":>8} {"AvgLen":>8} {"MedianLen":>10} {"MaxLen":>8} {"P10":>6} {"P90":>6}')
print('-' * 70)

for r in regime_ids:
    mask = run_regimes == r
    lengths = run_lengths[mask]
    name = df[df['regime'] == r]['regime_name'].iloc[0]
    print(f'{name:<22} {len(lengths):>8,} {lengths.mean():>8.1f} {np.median(lengths):>10.1f} '
          f'{lengths.max():>8} {np.percentile(lengths, 10):>6.0f} {np.percentile(lengths, 90):>6.0f}')

# =============================================================================
# 6. REGIME TRANSITION MATRIX (empirical)
# =============================================================================
print('\n' + '=' * 70)
print('6. EMPIRICAL TRANSITION MATRIX')
print('=' * 70)

trans = np.zeros((n_regimes, n_regimes))
for i in range(len(run_regimes) - 1):
    fr = int(run_regimes[i])
    to = int(run_regimes[i + 1])
    fr_idx = regime_ids.index(fr)
    to_idx = regime_ids.index(to)
    trans[fr_idx, to_idx] += 1

# Normalise
trans_pct = trans / trans.sum(axis=1, keepdims=True) * 100

names_short = [df[df['regime'] == r]['regime_name'].iloc[0][:10] for r in regime_ids]
header = f'{"From \\ To":<12}' + ''.join(f'{n:>12}' for n in names_short)
print(header)
print('-' * (12 + 12 * n_regimes))
for i, r in enumerate(regime_ids):
    row = f'{names_short[i]:<12}'
    for j in range(n_regimes):
        row += f'{trans_pct[i, j]:>11.1f}%'
    print(row)

# =============================================================================
# 7. SUMMARY — TRADE / AVOID RECOMMENDATION
# =============================================================================
print('\n' + '=' * 70)
print('7. SUMMARY — REGIME TRADING RECOMMENDATION')
print('=' * 70)

profitable = [r for r in regime_ids if regime_ev[r]['ev'] > 0]
avoid = [r for r in regime_ids if regime_ev[r]['ev'] <= 0]

print(f'\nPROFITABLE regimes (EV > 0):')
for r in profitable:
    e = regime_ev[r]
    print(f'  {e["name"]:<22} EV=${e["ev"]:.4f}/trade  WR={e["wr"]:.1%}  n={e["n"]:,}')

print(f'\nAVOID regimes (EV <= 0):')
for r in avoid:
    e = regime_ev[r]
    print(f'  {e["name"]:<22} EV=${e["ev"]:.4f}/trade  WR={e["wr"]:.1%}  n={e["n"]:,}')

total_all = sum(regime_ev[r]['total_ev'] for r in regime_ids)
total_trade = sum(regime_ev[r]['total_ev'] for r in profitable)
total_avoid = sum(regime_ev[r]['total_ev'] for r in avoid)
n_trade = sum(regime_ev[r]['n'] for r in profitable)
n_avoid = sum(regime_ev[r]['n'] for r in avoid)

print(f'\nIf trading ALL regimes:     EV = ${total_all:,.1f} over {len(df):,} bars')
print(f'If trading ONLY profitable: EV = ${total_trade:,.1f} over {n_trade:,} bars')
print(f'Avoided loss:               EV = ${total_avoid:,.1f} over {n_avoid:,} bars')

print('\n' + '=' * 70)
print('REGIME STUDY COMPLETE')
print('=' * 70)
