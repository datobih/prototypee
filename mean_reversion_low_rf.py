import pandas as pd
import numpy as np

print('='*80)
print('BTCUSD MEAN REVERSION - LOW RF EXPLORATION')
print('Looking for SHORT when RF is LOW (model says NO long)')
print('='*80)

df = pd.read_csv('data/processed/BTCUSD_feature_analysis.csv')
print(f'Total bars: {len(df)}')

# For mean reversion, low RF might be better for shorts
rf_low = df[df['rf_prob'] < 0.30].copy()
print(f'RF < 0.30 bars: {len(rf_low)}')

rf_low['long_win'] = (rf_low['outcome'] == 1).astype(int)
rf_low['short_win'] = (rf_low['outcome'] == 2).astype(int)

baseline_short = rf_low['short_win'].mean() * 100
baseline_long = rf_low['long_win'].mean() * 100
print(f'\nBaseline (RF<0.30):')
print(f'  SHORT win rate: {baseline_short:.1f}%')
print(f'  LONG win rate: {baseline_long:.1f}%')

results = []

# Mean reversion setups with LOW RF
print('\n' + '='*80)
print('BEARISH CONTINUATION + LOW RF')
print('='*80)

# 3 down bars
down_3 = rf_low[rf_low['up_count_3'] == 0]
if len(down_3) > 50:
    sr = down_3['short_win'].mean() * 100
    results.append(('RF<0.30 + 3down', sr, len(down_3)))
    print(f'RF<0.30 + 3down: SHORT WR={sr:.1f}% (n={len(down_3)})')

# Big body bearish
bb_bear = rf_low[(rf_low['big_body'] == 1) & (rf_low['close_position'] < 0.3)]
if len(bb_bear) > 50:
    sr = bb_bear['short_win'].mean() * 100
    results.append(('RF<0.30 + big_body_bearish', sr, len(bb_bear)))
    print(f'RF<0.30 + big_body_bearish: SHORT WR={sr:.1f}% (n={len(bb_bear)})')

# Trend down
trend_down = rf_low[rf_low['trend_align'] == -1]
if len(trend_down) > 50:
    sr = trend_down['short_win'].mean() * 100
    results.append(('RF<0.30 + trend_down', sr, len(trend_down)))
    print(f'RF<0.30 + trend_down: SHORT WR={sr:.1f}% (n={len(trend_down)})')

# Strong sell imbalance
sell_imb = rf_low[rf_low['imbalance_3'] <= -2]
if len(sell_imb) > 50:
    sr = sell_imb['short_win'].mean() * 100
    results.append(('RF<0.30 + imbalance<=-2', sr, len(sell_imb)))
    print(f'RF<0.30 + imbalance<=-2: SHORT WR={sr:.1f}% (n={len(sell_imb)})')

# 3 down + big body + close<0.3
combo1 = rf_low[(rf_low['up_count_3'] == 0) & (rf_low['big_body'] == 1) & (rf_low['close_position'] < 0.3)]
if len(combo1) > 20:
    sr = combo1['short_win'].mean() * 100
    results.append(('RF<0.30 + 3down + bb + low_close', sr, len(combo1)))
    print(f'RF<0.30 + 3down + bb + low_close: SHORT WR={sr:.1f}% (n={len(combo1)})')

# 3 down + trend down
combo2 = rf_low[(rf_low['up_count_3'] == 0) & (rf_low['trend_align'] == -1)]
if len(combo2) > 20:
    sr = combo2['short_win'].mean() * 100
    results.append(('RF<0.30 + 3down + trend_down', sr, len(combo2)))
    print(f'RF<0.30 + 3down + trend_down: SHORT WR={sr:.1f}% (n={len(combo2)})')

# Very low RF
very_low = df[df['rf_prob'] < 0.20].copy()
very_low['short_win'] = (very_low['outcome'] == 2).astype(int)
print(f'\nRF < 0.20 bars: {len(very_low)}')
if len(very_low) > 50:
    sr = very_low['short_win'].mean() * 100
    results.append(('RF<0.20 (baseline)', sr, len(very_low)))
    print(f'RF<0.20 SHORT WR: {sr:.1f}% (n={len(very_low)})')

# Very low RF + bearish pattern
vl_bear = very_low[(very_low['up_count_3'] == 0) & (very_low['big_body'] == 1) & (very_low['close_position'] < 0.3)]
if len(vl_bear) > 20:
    sr = vl_bear['short_win'].mean() * 100
    results.append(('RF<0.20 + bearish_pattern', sr, len(vl_bear)))
    print(f'RF<0.20 + bearish_pattern: SHORT WR={sr:.1f}% (n={len(vl_bear)})')

print('\n' + '='*80)
print('TOP SHORT STRATEGIES (Sorted by Win Rate)')
print('='*80)
results.sort(key=lambda x: x[1], reverse=True)
print(f'\n{"Strategy":<45} {"WR%":<10} {"Count"}')
print('-'*65)
for name, wr, count in results:
    print(f'{name:<45} {wr:>7.1f}% {count:>8}')
