import pandas as pd
import numpy as np
import pickle
from itertools import combinations

print('Loading data...')
df = pd.read_csv('data/processed/XAUUSD1_feature_analysis.csv')
print(f'Total bars: {len(df)}')

with open('models/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour

def get_session(hour):
    if 0 <= hour < 8: return 'ASIAN'
    elif 8 <= hour < 13: return 'LONDON'
    elif 13 <= hour < 17: return 'NY_OVERLAP'
    elif 17 <= hour < 22: return 'NY'
    else: return 'LATE'

df['session'] = df['hour'].apply(get_session)

X = df[feature_names].values
X_scaled = scaler.transform(X)
df['rf_prob'] = rf_model.predict_proba(X_scaled)[:, 1]

# Only include RESOLVED trades (outcome 1 or 2, not 0)
# For 3:1 RR: outcome=1 is WIN (+3R), outcome=2 is LOSS (-1R)
test = df[df['outcome'] > 0].copy()
test['win'] = (test['outcome'] == 1).astype(int)
print(f'Resolved trades: {len(test)}, Base WR: {test["win"].mean()*100:.1f}%')
print(f'3:1 RR - Need >25% WR for profit (breakeven at 25%)')
print()

filters = {
    'up3': lambda d: d['up_count_3'] == 3,
    'up2': lambda d: d['up_count_3'] >= 2,
    'big_body': lambda d: d['big_body'] == 1,
    'strong_close': lambda d: d['close_position'] > 0.7,
    'mid_close': lambda d: d['close_position'] > 0.5,
    'trend_up': lambda d: d['trend_align'] == 1,
    'vol_expand': lambda d: d['vol_expansion'] == 1,
    'no_upper_rej': lambda d: d['upper_reject'] == 0,
    'lower_rej': lambda d: d['lower_reject'] == 1,
    'not_at_low': lambda d: d['at_low'] == 0,
    'at_high': lambda d: d['at_high'] == 1,
    'ny_overlap': lambda d: d['session'] == 'NY_OVERLAP',
    'ny_sessions': lambda d: d['session'].isin(['NY_OVERLAP', 'NY']),
    'london': lambda d: d['session'] == 'LONDON',
    'no_london': lambda d: d['session'] != 'LONDON',
    'asian': lambda d: d['session'] == 'ASIAN',
}

def calc_ev(wr, rr=3):
    """Calculate expected value for given win rate and RR"""
    return (wr * rr) - ((1 - wr) * 1)

print('=== RF THRESHOLD EXPLORATION (3:1 RR) ===')
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    sub = test[test['rf_prob'] >= thresh]
    if len(sub) >= 10:
        wr = sub['win'].mean()
        ev = calc_ev(wr)
        print(f'RF>={thresh}: {wr*100:5.1f}% WR, EV={ev:+.2f}R ({len(sub)} trades)')

print()
print('=== SINGLE FILTERS (RF >= 0.60) ===')
rf_base = test[test['rf_prob'] >= 0.60]
base_wr = rf_base['win'].mean()
print(f'{"RF>=0.60 only":25} {base_wr*100:5.1f}% WR, EV={calc_ev(base_wr):+.2f}R ({len(rf_base)} trades)')
for name, filt in filters.items():
    sub = test[(test['rf_prob'] >= 0.60) & filt(test)]
    if len(sub) >= 30:
        wr = sub['win'].mean()
        ev = calc_ev(wr)
        diff = (wr - base_wr) * 100
        print(f'{name:25} {wr*100:5.1f}% WR, EV={ev:+.2f}R ({len(sub):5} trades) [{diff:+.1f}%]')

print()
print('=== BEST 2-FILTER COMBOS (RF >= 0.60, min 30 trades, sorted by EV) ===')
results = []
for c in combinations(filters.keys(), 2):
    mask = (test['rf_prob'] >= 0.60)
    for f in c: mask = mask & filters[f](test)
    sub = test[mask]
    if len(sub) >= 30:
        wr = sub['win'].mean()
        ev = calc_ev(wr)
        results.append((c, wr*100, ev, len(sub)))
results.sort(key=lambda x: -x[2])  # Sort by EV
for c, wr, ev, n in results[:20]:
    print(f'{" + ".join(c):45} {wr:5.1f}% WR, EV={ev:+.2f}R ({n} trades)')

print()
print('=== BEST 3-FILTER COMBOS (RF >= 0.60, min 20 trades, sorted by EV) ===')
results3 = []
for c in combinations(filters.keys(), 3):
    mask = (test['rf_prob'] >= 0.60)
    for f in c: mask = mask & filters[f](test)
    sub = test[mask]
    if len(sub) >= 20:
        wr = sub['win'].mean()
        ev = calc_ev(wr)
        results3.append((c, wr*100, ev, len(sub)))
results3.sort(key=lambda x: -x[2])  # Sort by EV
for c, wr, ev, n in results3[:20]:
    print(f'{" + ".join(c):55} {wr:5.1f}% WR, EV={ev:+.2f}R ({n} trades)')

print()
print('=== EXPLORING LOWER THRESHOLDS WITH STRONG FILTERS ===')
for thresh in [0.50, 0.55, 0.60]:
    print(f'\n--- RF >= {thresh} ---')
    for combo_name, combo_filters in [
        ('trend_up + vol_expand', ['trend_up', 'vol_expand']),
        ('up3 + vol_expand', ['up3', 'vol_expand']),
        ('trend_up + ny_overlap', ['trend_up', 'ny_overlap']),
        ('big_body + trend_up', ['big_body', 'trend_up']),
        ('lower_rej + trend_up', ['lower_rej', 'trend_up']),
    ]:
        mask = (test['rf_prob'] >= thresh)
        for f in combo_filters: mask = mask & filters[f](test)
        sub = test[mask]
        if len(sub) >= 10:
            wr = sub['win'].mean()
            ev = calc_ev(wr)
            print(f'  {combo_name:30} {wr*100:5.1f}% WR, EV={ev:+.2f}R ({len(sub)} trades)')

print()
print('=== SINGLE FILTERS (RF >= 0.70) ===')
rf70 = test[test['rf_prob'] >= 0.70]
print(f'{"RF>=0.70 only":20} {rf70["win"].mean()*100:5.1f}% ({len(rf70)} trades)')
for name, filt in filters.items():
    sub = test[(test['rf_prob'] >= 0.70) & filt(test)]
    if len(sub) >= 20:
        print(f'{name:20} {sub["win"].mean()*100:5.1f}% ({len(sub)} trades)')

print()
print('=== BEST 2-FILTER COMBOS (RF >= 0.70, min 20 trades) ===')
results = []
for c in combinations(filters.keys(), 2):
    mask = (test['rf_prob'] >= 0.70)
    for f in c: mask = mask & filters[f](test)
    sub = test[mask]
    if len(sub) >= 20:
        wr = sub['win'].mean()
        results.append((c, wr*100, len(sub)))
results.sort(key=lambda x: -x[1])
for c, wr, n in results[:15]:
    print(f'{" + ".join(c):40} {wr:5.1f}% ({n} trades)')

print()
print('=== BEST 3-FILTER COMBOS (RF >= 0.70, min 15 trades) ===')
results3 = []
for c in combinations(filters.keys(), 3):
    mask = (test['rf_prob'] >= 0.70)
    for f in c: mask = mask & filters[f](test)
    sub = test[mask]
    if len(sub) >= 15:
        wr = sub['win'].mean()
        results3.append((c, wr*100, len(sub)))
results3.sort(key=lambda x: -x[1])
for c, wr, n in results3[:15]:
    print(f'{" + ".join(c):55} {wr:5.1f}% ({n} trades)')

print()
print('=== HIGHER THRESHOLDS ===')
for thresh in [0.75, 0.80, 0.85]:
    sub = test[test['rf_prob'] >= thresh]
    if len(sub) >= 10:
        print(f'RF>={thresh}: {sub["win"].mean()*100:.1f}% WR ({len(sub)} trades)')

print()
print('=== RF THRESHOLDS WITH NY_OVERLAP ===')
for thresh in [0.65, 0.70, 0.72, 0.74, 0.75]:
    sub = test[(test['rf_prob'] >= thresh) & (test['session'] == 'NY_OVERLAP')]
    if len(sub) >= 20:
        print(f'RF>={thresh} + ny_overlap: {sub["win"].mean()*100:.1f}% WR ({len(sub)} trades)')

print()
print('=== BEST COMBOS AT RF >= 0.72 (balanced threshold) ===')
results72 = []
for c in combinations(filters.keys(), 2):
    mask = (test['rf_prob'] >= 0.72)
    for f in c: mask = mask & filters[f](test)
    sub = test[mask]
    if len(sub) >= 30:
        wr = sub['win'].mean()
        results72.append((c, wr*100, len(sub)))
results72.sort(key=lambda x: -x[1])
for c, wr, n in results72[:10]:
    print(f'{" + ".join(c):40} {wr:5.1f}% ({n} trades)')

print()
print('=== EXPLORING RULE 4 VARIANTS (up3 + vol_expand) ===')
for thresh in [0.65, 0.68, 0.70, 0.72]:
    sub = test[(test['rf_prob'] >= thresh) & (test['up_count_3'] == 3) & (test['vol_expansion'] == 1)]
    if len(sub) >= 10:
        print(f'RF>={thresh} + up3 + vol_expand: {sub["win"].mean()*100:.1f}% WR ({len(sub)} trades)')
    sub2 = test[(test['rf_prob'] >= thresh) & (test['up_count_3'] == 3) & (test['vol_expansion'] == 1) & (test['session'] == 'NY_OVERLAP')]
    if len(sub2) >= 10:
        print(f'  + ny_overlap: {sub2["win"].mean()*100:.1f}% WR ({len(sub2)} trades)')
