"""
Audit fib retracement logic — trace exact computation for trade #20
and check all Fib 38-62% + FVG trades for degenerate cases.
"""
import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

# Load price data
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'raw', 'XAUUSD30.csv'),
                 sep='\t', header=0,
                 names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                        'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'TickVol']].copy()

# Load combo trades
combo = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_combos',
                                  'top1_Fib_38-62pct__FVG.csv'))
combo['ob_dt'] = pd.to_datetime(combo['ob_dt'])

# Also load ALL OBs
all_obs = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed',
                                    'ob_mitigation_study.csv'))
all_obs['ob_dt'] = pd.to_datetime(all_obs['ob_dt'])

print('=' * 90)
print('FIB LOGIC AUDIT')
print('=' * 90)

# ─────────────────────────────────────────────────────────────────
# TRACE TRADE #20 STEP BY STEP
# ─────────────────────────────────────────────────────────────────
print('\n>>> TRACING TRADE #20: bear OB at 2025-11-20 15:00')
print('-' * 90)

row = combo.iloc[19]
ob_dt = row['ob_dt']
ob_pos = df.index.get_loc(ob_dt)
ob_high = row['ob_high']
ob_low = row['ob_low']
bars_to_mit = int(row['bars_to_mitigate'])

print(f'OB candle (bar {ob_pos}): {ob_dt}')
ob_bar = df.iloc[ob_pos]
print(f'  O={ob_bar["Open"]:.2f}  H={ob_bar["High"]:.2f}  '
      f'L={ob_bar["Low"]:.2f}  C={ob_bar["Close"]:.2f}')
print(f'  Bullish: {ob_bar["Close"] > ob_bar["Open"]}  ← correct for bear OB (supply candle)')

# Impulse is the impulsive bearish candle AFTER the OB
# In ob_filtering.py the impulse is found first, then OB is looked back
# For this bear OB, impulse is the next bar (impulsive bearish)
imp_idx = ob_pos + 1  # approximation; the actual code finds impulse first
imp_bar = df.iloc[imp_idx]
print(f'\nImpulse candle (bar {imp_idx}): {df.index[imp_idx]}')
print(f'  O={imp_bar["Open"]:.2f}  H={imp_bar["High"]:.2f}  '
      f'L={imp_bar["Low"]:.2f}  C={imp_bar["Close"]:.2f}')
print(f'  Body: ${abs(imp_bar["Close"] - imp_bar["Open"]):.2f}')
print(f'  Bearish: {imp_bar["Close"] < imp_bar["Open"]}')

# Touch bar
touch_idx = imp_idx + bars_to_mit
touch_bar = df.iloc[touch_idx]
print(f'\nTouch candle (bar {touch_idx}, {bars_to_mit} bars after impulse): {df.index[touch_idx]}')
print(f'  O={touch_bar["Open"]:.2f}  H={touch_bar["High"]:.2f}  '
      f'L={touch_bar["Low"]:.2f}  C={touch_bar["Close"]:.2f}')
print(f'  High >= OB low? {touch_bar["High"]:.2f} >= {ob_low:.2f} → {touch_bar["High"] >= ob_low}')

# Now trace the fib computation (lines 306-310 of ob_filtering.py)
print(f'\n--- FIB COMPUTATION (bear OB, lines 306-310) ---')
print(f'  swing_low = df.iloc[{imp_idx}:{touch_idx}]["Low"].min()')

fib_range = df.iloc[imp_idx:touch_idx]
print(f'  This slice covers {len(fib_range)} bar(s):')
for j, (dt, r) in enumerate(fib_range.iterrows()):
    print(f'    Bar {imp_idx + j}: {dt}  Low=${r["Low"]:.2f}')

swing_low = fib_range['Low'].min()
full_move = ob_high - swing_low
retrace_dist = touch_bar['High'] - swing_low
fib_pct = retrace_dist / full_move * 100

print(f'\n  swing_low     = ${swing_low:.2f}')
print(f'  ob_high       = ${ob_high:.3f}')
print(f'  full_move     = ob_high - swing_low = ${full_move:.2f}')
print(f'  touch High    = ${touch_bar["High"]:.2f}')
print(f'  retrace_dist  = touch_High - swing_low = ${retrace_dist:.2f}')
print(f'  fib_pct       = {retrace_dist:.2f} / {full_move:.2f} * 100 = {fib_pct:.1f}%')
print(f'  In 38-62%?    → {38 <= fib_pct <= 62}')

print(f'\n>>> PROBLEM: The "swing" is computed on only {len(fib_range)} bar(s).')
print(f'   bars_to_mitigate = {bars_to_mit} means touch happens immediately.')
print(f'   There is NO real swing structure — the fib is measured on noise.')

# ─────────────────────────────────────────────────────────────────
# CHECK ALL 20 COMBO TRADES FOR THIS ISSUE
# ─────────────────────────────────────────────────────────────────
print('\n\n' + '=' * 90)
print('ALL 20 Fib 38-62% + FVG TRADES — SWING BAR COUNT')
print('=' * 90)
print(f'\n{"#":>3} {"Type":>5} {"OB Date":>22} {"Bars2Mit":>9} {"FibSwgBars":>11} '
      f'{"Fib%":>6} {"React":>8} {"RR":>5} {"Valid?":>7}')
print('-' * 90)

MIN_SWING_BARS = 3  # minimum bars needed for a meaningful fib

for idx, r in combo.iterrows():
    btm = int(r['bars_to_mitigate'])
    # Fib range = imp_idx to touch_bar, which is bars_to_mitigate bars
    # (the swing_low/high is found in df.iloc[imp_idx:touch_bar])
    swing_bars = btm  # the slice length
    valid = swing_bars >= MIN_SWING_BARS
    marker = '  ✓' if valid else '  ✗ DEGENERATE'
    print(f'{idx+1:>3} {r["type"]:>5} {str(r["ob_dt"]):>22} {btm:>9} '
          f'{swing_bars:>11} {r["fib_retracement_pct"]:>5.1f}% '
          f'{r["reaction"]:>8} {r["trade_rr"]:>5.1f}{marker}')

degen_count = (combo['bars_to_mitigate'] < MIN_SWING_BARS).sum()
print(f'\nDegenerate (swing < {MIN_SWING_BARS} bars): {degen_count}/{len(combo)} trades')

# ─────────────────────────────────────────────────────────────────
# CHECK ALL OBs — distribution of bars_to_mitigate
# ─────────────────────────────────────────────────────────────────
print('\n\n' + '=' * 90)
print('ALL OBs — bars_to_mitigate DISTRIBUTION (touched only)')
print('=' * 90)

touched = all_obs[all_obs['touched'] == True]
print(f'\nTotal touched: {len(touched)}')
for thresh in [1, 2, 3, 5, 10]:
    n = (touched['bars_to_mitigate'] <= thresh).sum()
    pct = n / len(touched) * 100
    print(f'  bars_to_mitigate <= {thresh:>2}: {n:>4} ({pct:>5.1f}%)')

# Check fib values for immediate touches
print('\n--- Fib retracement by bars_to_mitigate bucket ---')
for lo, hi, label in [(1, 1, '1 bar (immediate)'), (2, 2, '2 bars'),
                       (3, 5, '3-5 bars'), (6, 20, '6-20 bars'), (21, 999, '>20 bars')]:
    subset = touched[(touched['bars_to_mitigate'] >= lo) & (touched['bars_to_mitigate'] <= hi)]
    fib = subset['fib_retracement_pct'].dropna()
    if len(fib) < 3:
        continue
    in_golden = ((fib >= 38) & (fib <= 62)).sum()
    print(f'  {label:<20} N={len(fib):>4}  '
          f'median fib={fib.median():>5.1f}%  '
          f'in 38-62%: {in_golden} ({in_golden/len(fib)*100:.1f}%)')

# ─────────────────────────────────────────────────────────────────
# WHAT DOES THE FIB 38-62% COMBO LOOK LIKE WITHOUT DEGENERATES?
# ─────────────────────────────────────────────────────────────────
print('\n\n' + '=' * 90)
print(f'FIB 38-62% + FVG COMBO — WITHOUT DEGENERATE (bars_to_mitigate >= {MIN_SWING_BARS})')
print('=' * 90)

clean = combo[combo['bars_to_mitigate'] >= MIN_SWING_BARS]
n_clean = len(clean)
bounces_clean = (clean['reaction'] == 'BOUNCE').sum()
breaks_clean = (clean['reaction'] == 'BREAK').sum()
avg_rr_clean = clean[clean['reaction'] == 'BOUNCE']['trade_rr'].mean() if bounces_clean > 0 else 0

print(f'  Total trades:  {n_clean} (removed {len(combo) - n_clean} degenerate)')
print(f'  BOUNCE:        {bounces_clean} ({bounces_clean/n_clean*100:.1f}%)')
print(f'  BREAK:         {breaks_clean}')
print(f'  Avg RR (bnc):  {avg_rr_clean:.1f}')

# Also check with higher threshold
for min_bars in [5, 10]:
    c = combo[combo['bars_to_mitigate'] >= min_bars]
    if len(c) < 3:
        continue
    b = (c['reaction'] == 'BOUNCE').sum()
    rr = c[c['reaction'] == 'BOUNCE']['trade_rr'].mean() if b > 0 else 0
    print(f'\n  With min_bars >= {min_bars}:  N={len(c)}  '
          f'Bounce={b}/{len(c)} ({b/len(c)*100:.1f}%)  Avg RR={rr:.1f}')

print('\n' + '=' * 90)
print('DONE')
print('=' * 90)
