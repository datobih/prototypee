import pandas as pd
import numpy as np

print('='*80)
print('BTCUSD MEAN REVERSION STRATEGY EXPLORATION')
print('RF >= 0.70 Filter Applied')
print('='*80)

# Load the data
df = pd.read_csv('data/processed/BTCUSD_feature_analysis.csv')
print(f'Total bars: {len(df)}')

# Filter for RF >= 0.70
rf_high = df[df['rf_prob'] >= 0.70].copy()
print(f'RF >= 0.70 bars: {len(rf_high)}')

# Outcome: 1=LONG win, 2=SHORT win, 0=no movement
# For mean reversion SHORT, we want outcome == 2
rf_high['long_win'] = (rf_high['outcome'] == 1).astype(int)
rf_high['short_win'] = (rf_high['outcome'] == 2).astype(int)
rf_high['any_win'] = (rf_high['outcome'] != 0).astype(int)

baseline_long = rf_high['long_win'].mean() * 100
baseline_short = rf_high['short_win'].mean() * 100
print(f'\nBaseline (RF>=0.70):')
print(f'  LONG win rate: {baseline_long:.1f}%')
print(f'  SHORT win rate: {baseline_short:.1f}%')

print('\n' + '='*80)
print('MEAN REVERSION INDICATORS (Looking for SHORT opportunities)')
print('='*80)

results = []

# 1. Overextended above EMA (dist_ema8 > 0)
over_ema = rf_high[rf_high['dist_ema8'] > 0.001]  # 0.1% above EMA
if len(over_ema) > 50:
    sr = over_ema['short_win'].mean() * 100
    results.append(('dist_ema8 > 0.1%', sr, len(over_ema), 'SHORT'))
    print(f'dist_ema8 > 0.1%: SHORT WR={sr:.1f}% (n={len(over_ema)})')

# 2. Upper rejection (wick rejection at highs)
upper_rej = rf_high[rf_high['upper_reject'] == 1]
if len(upper_rej) > 50:
    sr = upper_rej['short_win'].mean() * 100
    results.append(('upper_reject == 1', sr, len(upper_rej), 'SHORT'))
    print(f'upper_reject == 1: SHORT WR={sr:.1f}% (n={len(upper_rej)})')

# 3. At recent high
at_high = rf_high[rf_high['at_high'] == 1]
if len(at_high) > 50:
    sr = at_high['short_win'].mean() * 100
    results.append(('at_high == 1', sr, len(at_high), 'SHORT'))
    print(f'at_high == 1: SHORT WR={sr:.1f}% (n={len(at_high)})')

# 4. Negative flow after up move (momentum exhaustion)
neg_flow = rf_high[(rf_high['flow_momentum'] < -0.0005)]
if len(neg_flow) > 50:
    sr = neg_flow['short_win'].mean() * 100
    results.append(('flow_momentum < -0.0005', sr, len(neg_flow), 'SHORT'))
    print(f'flow_momentum < -0.0005: SHORT WR={sr:.1f}% (n={len(neg_flow)})')

# 5. Close position < 0.3 (close near low = bearish)
low_close = rf_high[rf_high['close_position'] < 0.3]
if len(low_close) > 50:
    sr = low_close['short_win'].mean() * 100
    results.append(('close_position < 0.3', sr, len(low_close), 'SHORT'))
    print(f'close_position < 0.3: SHORT WR={sr:.1f}% (n={len(low_close)})')

# 6. Sell imbalance
sell_imb = rf_high[rf_high['sell_imbalance'] == 1]
if len(sell_imb) > 50:
    sr = sell_imb['short_win'].mean() * 100
    results.append(('sell_imbalance == 1', sr, len(sell_imb), 'SHORT'))
    print(f'sell_imbalance == 1: SHORT WR={sr:.1f}% (n={len(sell_imb)})')

# 7. Strong negative imbalance
neg_imb = rf_high[rf_high['imbalance_3'] <= -2]
if len(neg_imb) > 50:
    sr = neg_imb['short_win'].mean() * 100
    results.append(('imbalance_3 <= -2', sr, len(neg_imb), 'SHORT'))
    print(f'imbalance_3 <= -2: SHORT WR={sr:.1f}% (n={len(neg_imb)})')

# 8. Trend misalignment (price below EMAs)
trend_down = rf_high[rf_high['trend_align'] == -1]
if len(trend_down) > 50:
    sr = trend_down['short_win'].mean() * 100
    results.append(('trend_align == -1', sr, len(trend_down), 'SHORT'))
    print(f'trend_align == -1: SHORT WR={sr:.1f}% (n={len(trend_down)})')

print('\n' + '='*80)
print('COMBINATION MEAN REVERSION STRATEGIES')
print('='*80)

# Combo 1: Upper rejection + close < 0.5
combo1 = rf_high[(rf_high['upper_reject'] == 1) & (rf_high['close_position'] < 0.5)]
if len(combo1) > 20:
    sr = combo1['short_win'].mean() * 100
    results.append(('upper_reject + close_pos<0.5', sr, len(combo1), 'SHORT'))
    print(f'upper_reject + close_pos<0.5: SHORT WR={sr:.1f}% (n={len(combo1)})')

# Combo 2: At high + negative flow
combo2 = rf_high[(rf_high['at_high'] == 1) & (rf_high['flow_momentum'] < 0)]
if len(combo2) > 20:
    sr = combo2['short_win'].mean() * 100
    results.append(('at_high + neg_flow', sr, len(combo2), 'SHORT'))
    print(f'at_high + neg_flow: SHORT WR={sr:.1f}% (n={len(combo2)})')

# Combo 3: Overextended + upper rejection
combo3 = rf_high[(rf_high['dist_ema8'] > 0.001) & (rf_high['upper_reject'] == 1)]
if len(combo3) > 20:
    sr = combo3['short_win'].mean() * 100
    results.append(('dist_ema8>0.1% + upper_reject', sr, len(combo3), 'SHORT'))
    print(f'dist_ema8>0.1% + upper_reject: SHORT WR={sr:.1f}% (n={len(combo3)})')

# Combo 4: Trend down + sell imbalance
combo4 = rf_high[(rf_high['trend_align'] == -1) & (rf_high['imbalance_3'] < 0)]
if len(combo4) > 20:
    sr = combo4['short_win'].mean() * 100
    results.append(('trend_down + neg_imbalance', sr, len(combo4), 'SHORT'))
    print(f'trend_down + neg_imbalance: SHORT WR={sr:.1f}% (n={len(combo4)})')

# Combo 5: Big body down (bearish continuation)
combo5 = rf_high[(rf_high['big_body'] == 1) & (rf_high['close_position'] < 0.3)]
if len(combo5) > 20:
    sr = combo5['short_win'].mean() * 100
    results.append(('big_body + close<0.3', sr, len(combo5), 'SHORT'))
    print(f'big_body + close<0.3: SHORT WR={sr:.1f}% (n={len(combo5)})')

# Combo 6: 3 consecutive down bars
down_3 = rf_high[rf_high['up_count_3'] == 0]
if len(down_3) > 20:
    sr = down_3['short_win'].mean() * 100
    results.append(('up_count_3 == 0 (3 down)', sr, len(down_3), 'SHORT'))
    print(f'up_count_3 == 0 (3 down): SHORT WR={sr:.1f}% (n={len(down_3)})')

# Combo 7: 3 down + big body + close near low
combo7 = rf_high[(rf_high['up_count_3'] == 0) & (rf_high['big_body'] == 1) & (rf_high['close_position'] < 0.3)]
if len(combo7) > 10:
    sr = combo7['short_win'].mean() * 100
    results.append(('3down + big_body + close<0.3', sr, len(combo7), 'SHORT'))
    print(f'3down + big_body + close<0.3: SHORT WR={sr:.1f}% (n={len(combo7)})')

print('\n' + '='*80)
print('TOP MEAN REVERSION STRATEGIES (Sorted by Win Rate)')
print('='*80)

results.sort(key=lambda x: x[1], reverse=True)
print(f'\n{\"Strategy\":<40} {\"WR%\":<10} {\"Count\":<10} {\"Direction\"}')
print('-'*70)
for name, wr, count, direction in results[:15]:
    if count >= 20:
        print(f'{name:<40} {wr:>7.1f}% {count:>8} {direction:>10}')

print(f'\nBaseline SHORT: {baseline_short:.1f}%')
