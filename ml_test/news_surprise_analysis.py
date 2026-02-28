"""
news_surprise_analysis.py

Deep dive: Can we predict gold's DIRECTION from the surprise direction?

Analysis:
  1. Per-indicator: correlation between surprise_z and gold reaction
  2. Sign agreement: does positive surprise → gold move in expected direction?
  3. Magnitude scaling: do bigger surprises produce bigger moves?
  4. Conditional analysis: is direction more predictable when surprise is large?
  5. Composite surprise signal: combine top indicators
  6. Simulated directional trades: if we traded the direction, what happens?
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# LOAD
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
events_df = pd.read_csv(os.path.join(DATA_DIR, 'news_events_reactions.csv'))
events_df['release_date'] = pd.to_datetime(events_df['release_date'])
events_df['server_time']  = pd.to_datetime(events_df['server_time'])

print('=' * 90)
print('SURPRISE vs GOLD REACTION — DIRECTIONAL ANALYSIS')
print('=' * 90)
print(f'Total events loaded: {len(events_df)}')

# Expected direction mapping
# 'inverse': strong data (positive surprise) → gold DOWN (negative reaction)
# 'direct':  strong data (positive surprise) → gold UP (positive reaction)
# So for "inverse" indicators, the EXPECTED sign of reaction = -1 * sign(surprise)
# For "direct" indicators, the EXPECTED sign of reaction = sign(surprise)

HORIZONS = [5, 15, 30, 60, 240]

# =============================================================================
# 1. RAW CORRELATION: surprise_z vs gold move (per indicator, per horizon)
# =============================================================================
print('\n' + '=' * 90)
print('1. PEARSON CORRELATION: surprise_z vs gold move')
print('=' * 90)
print('   Negative corr for "inverse" indicators = CORRECT behavior')
print('   (strong data → gold falls)')
print()

header = f'{"Indicator":<30} {"Dir":>7} {"N":>4}'
for h in HORIZONS:
    header += f' {"r_"+str(h)+"m":>8}'
header += f' {"p_30m":>8}'
print(header)
print('─' * 110)

corr_results = []

for (sid, name), grp in events_df.groupby(['series_id', 'indicator']):
    if len(grp) < 15:
        continue
    
    row = {'indicator': name, 'series_id': sid, 'n': len(grp),
           'expected_dir': grp['expected_dir'].iloc[0]}
    
    line = f'{name:<30} {row["expected_dir"]:>7} {len(grp):>4}'
    
    for h in HORIZONS:
        col = f'{h}m_chg'
        valid = grp.dropna(subset=['surprise_z', col])
        if len(valid) < 10:
            row[f'r_{h}m'] = np.nan
            row[f'p_{h}m'] = np.nan
            line += f' {"N/A":>8}'
            continue
        
        r, p = stats.pearsonr(valid['surprise_z'], valid[col])
        row[f'r_{h}m'] = r
        row[f'p_{h}m'] = p
        
        # Color-code: significant correlations
        sig = '*' if p < 0.05 else ' '
        line += f' {r:>+7.3f}{sig}'
    
    # p-value for 30m
    p30 = row.get('p_30m', np.nan)
    line += f' {p30:>8.4f}' if pd.notna(p30) else f' {"N/A":>8}'
    
    print(line)
    corr_results.append(row)

corr_df = pd.DataFrame(corr_results)

# =============================================================================
# 2. SIGN AGREEMENT: does surprise direction predict move direction?
# =============================================================================
print('\n' + '=' * 90)
print('2. SIGN AGREEMENT: surprise direction → gold direction')
print('=' * 90)
print('   "Correct" = gold moved in expected direction given surprise sign')
print('   >50% = directional edge exists')
print()

# Filter to meaningful surprises only (|z| > 0.5)
MIN_Z = 0.5

header = f'{"Indicator":<30} {"Dir":>7} {"N":>4}'
for h in HORIZONS:
    header += f' {str(h)+"m":>8}'
print(header)
print('─' * 90)

sign_results = []

for (sid, name), grp in events_df.groupby(['series_id', 'indicator']):
    meaningful = grp[grp['surprise_z'].abs() > MIN_Z].copy()
    if len(meaningful) < 10:
        continue
    
    expected_dir = grp['expected_dir'].iloc[0]
    row = {'indicator': name, 'n_meaningful': len(meaningful)}
    
    line = f'{name:<30} {expected_dir:>7} {len(meaningful):>4}'
    
    for h in HORIZONS:
        col = f'{h}m_chg'
        valid = meaningful.dropna(subset=[col])
        if len(valid) < 5:
            line += f' {"N/A":>8}'
            continue
        
        if expected_dir == 'inverse':
            # positive surprise → expect negative gold move
            correct = ((valid['surprise_z'] > 0) & (valid[col] < 0)) | \
                      ((valid['surprise_z'] < 0) & (valid[col] > 0))
        else:
            # positive surprise → expect positive gold move
            correct = ((valid['surprise_z'] > 0) & (valid[col] > 0)) | \
                      ((valid['surprise_z'] < 0) & (valid[col] < 0))
        
        pct = correct.mean() * 100
        row[f'sign_{h}m'] = pct
        
        marker = '**' if pct >= 60 else '*' if pct >= 55 else '  '
        line += f' {pct:>5.1f}%{marker}'
    
    print(line)
    sign_results.append(row)

# =============================================================================
# 3. MAGNITUDE SCALING: bigger surprise → bigger move?
# =============================================================================
print('\n' + '=' * 90)
print('3. MAGNITUDE SCALING: |surprise| vs |gold move|')
print('=' * 90)
print('   Do bigger surprises produce proportionally bigger gold moves?')
print()

print(f'{"Indicator":<30} {"N":>4} {"r(|z|,|30m|)":>12} {"p-val":>8} {"Slope":>8} {"Interp":>12}')
print('─' * 90)

for (sid, name), grp in events_df.groupby(['series_id', 'indicator']):
    valid = grp.dropna(subset=['surprise_z', '30m_chg'])
    if len(valid) < 15:
        continue
    
    abs_z = valid['surprise_z'].abs()
    abs_move = valid['30m_chg'].abs()
    
    r, p = stats.pearsonr(abs_z, abs_move)
    slope, intercept, _, _, _ = stats.linregress(abs_z, abs_move)
    
    interp = 'STRONG' if r > 0.3 and p < 0.05 else 'moderate' if r > 0.15 else 'weak'
    sig = '*' if p < 0.05 else ' '
    
    print(f'{name:<30} {len(valid):>4} {r:>+11.3f}{sig} {p:>8.4f} {slope:>+7.2f}$ {interp:>12}')

# =============================================================================
# 4. CONDITIONAL: direction accuracy by surprise bucket
# =============================================================================
print('\n' + '=' * 90)
print('4. DIRECTION ACCURACY BY SURPRISE SIZE (30-min horizon)')
print('=' * 90)
print('   Is direction MORE predictable when surprise is larger?')
print()

# Pool all inverse indicators for power
inverse_events = events_df[events_df['expected_dir'] == 'inverse'].copy()
inverse_events = inverse_events.dropna(subset=['surprise_z', '30m_chg'])

# For inverse: correct = sign(surprise) != sign(gold_move)
inverse_events['correct_dir'] = (
    (inverse_events['surprise_z'] > 0) & (inverse_events['30m_chg'] < 0)
) | (
    (inverse_events['surprise_z'] < 0) & (inverse_events['30m_chg'] > 0)
)

# Bucket by absolute surprise size
bins = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 999]
labels = ['0-0.25σ', '0.25-0.5σ', '0.5-1.0σ', '1.0-1.5σ', '1.5-2.0σ', '>2.0σ']
inverse_events['z_bucket'] = pd.cut(inverse_events['surprise_z'].abs(), bins=bins, labels=labels)

print(f'{"Surprise Bucket":<15} {"N":>5} {"Correct%":>10} {"Avg |Move|":>12} {"Avg Move":>12}')
print('─' * 60)

for bucket in labels:
    subset = inverse_events[inverse_events['z_bucket'] == bucket]
    if len(subset) < 3:
        continue
    
    pct_correct = subset['correct_dir'].mean() * 100
    avg_abs = subset['30m_chg'].abs().mean()
    avg_move = subset['30m_chg'].mean()
    
    marker = ' ←' if pct_correct >= 55 else ''
    print(f'{bucket:<15} {len(subset):>5} {pct_correct:>9.1f}% ${avg_abs:>10.2f} ${avg_move:>+10.2f}{marker}')

# Same for direct (unemployment etc.)
direct_events = events_df[events_df['expected_dir'] == 'direct'].copy()
direct_events = direct_events.dropna(subset=['surprise_z', '30m_chg'])
direct_events['correct_dir'] = (
    (direct_events['surprise_z'] > 0) & (direct_events['30m_chg'] > 0)
) | (
    (direct_events['surprise_z'] < 0) & (direct_events['30m_chg'] < 0)
)

if len(direct_events) > 10:
    print(f'\n  Direct indicators (pooled, N={len(direct_events)}):')
    direct_events['z_bucket'] = pd.cut(direct_events['surprise_z'].abs(), bins=bins, labels=labels)
    for bucket in labels:
        subset = direct_events[direct_events['z_bucket'] == bucket]
        if len(subset) < 3:
            continue
        pct = subset['correct_dir'].mean() * 100
        print(f'  {bucket:<15} {len(subset):>5} {pct:>9.1f}%')

# =============================================================================
# 5. SIMULATED DIRECTIONAL TRADES
# =============================================================================
print('\n' + '=' * 90)
print('5. SIMULATED DIRECTIONAL TRADES')
print('=' * 90)
print('   Rule: if |surprise_z| > threshold, trade the expected direction')
print('   Entry at release, exit at horizon. No TP/SL (pure directional).')
print()

for z_thresh in [0.5, 1.0, 1.5]:
    print(f'\n  ── Surprise threshold: |z| > {z_thresh}σ ──')
    
    tradeable = events_df[events_df['surprise_z'].abs() > z_thresh].copy()
    
    if len(tradeable) < 5:
        print(f'    Too few events ({len(tradeable)})')
        continue
    
    for h in [5, 15, 30, 60]:
        col = f'{h}m_chg'
        valid = tradeable.dropna(subset=[col]).copy()
        if len(valid) < 5:
            continue
        
        # Compute PnL: trade in expected direction
        pnls = []
        for _, ev in valid.iterrows():
            gold_move = ev[col]
            surprise = ev['surprise_z']
            exp_dir = ev['expected_dir']
            
            if exp_dir == 'inverse':
                # Positive surprise → short gold, negative surprise → long gold
                trade_dir = -1 if surprise > 0 else 1
            else:
                # Positive surprise → long gold
                trade_dir = 1 if surprise > 0 else -1
            
            pnl = trade_dir * gold_move
            pnls.append(pnl)
        
        pnls = np.array(pnls)
        total = pnls.sum()
        avg = pnls.mean()
        win_rate = (pnls > 0).mean() * 100
        sharpe = (pnls.mean() / pnls.std() * np.sqrt(len(pnls))) if pnls.std() > 0 else 0
        
        print(f'    {h:>3}m exit | N={len(valid):>4} | '
              f'Total: ${total:>+8.2f} | Avg: ${avg:>+5.2f} | '
              f'Win: {win_rate:>5.1f}% | t-stat: {sharpe:>+5.2f}')

# =============================================================================
# 6. BEST INDIVIDUAL INDICATOR STRATEGIES
# =============================================================================
print('\n' + '=' * 90)
print('6. BEST SINGLE-INDICATOR STRATEGIES (|z| > 0.5, 30-min exit)')
print('=' * 90)
print()

strat_results = []

for (sid, name), grp in events_df.groupby(['series_id', 'indicator']):
    valid = grp[(grp['surprise_z'].abs() > 0.5)].dropna(subset=['30m_chg']).copy()
    if len(valid) < 10:
        continue
    
    pnls = []
    for _, ev in valid.iterrows():
        surprise = ev['surprise_z']
        exp_dir = ev['expected_dir']
        
        if exp_dir == 'inverse':
            trade_dir = -1 if surprise > 0 else 1
        else:
            trade_dir = 1 if surprise > 0 else -1
        
        pnls.append(trade_dir * ev['30m_chg'])
    
    pnls = np.array(pnls)
    total = pnls.sum()
    avg = pnls.mean()
    win_rate = (pnls > 0).mean() * 100
    t_stat = (pnls.mean() / (pnls.std() / np.sqrt(len(pnls)))) if pnls.std() > 0 else 0
    
    strat_results.append({
        'indicator': name,
        'n_trades': len(pnls),
        'total_pnl': total,
        'avg_pnl': avg,
        'win_rate': win_rate,
        't_stat': t_stat,
    })

strat_df = pd.DataFrame(strat_results).sort_values('total_pnl', ascending=False)

print(f'{"Indicator":<30} {"N":>4} {"Total PnL":>10} {"Avg":>8} {"Win%":>7} {"t-stat":>8}')
print('─' * 75)

for _, r in strat_df.iterrows():
    sig = '**' if abs(r['t_stat']) > 2.0 else '*' if abs(r['t_stat']) > 1.5 else '  '
    print(f'{r["indicator"]:<30} {r["n_trades"]:>4} '
          f'${r["total_pnl"]:>+9.2f} ${r["avg_pnl"]:>+6.2f} '
          f'{r["win_rate"]:>6.1f}% {r["t_stat"]:>+7.2f}{sig}')

# =============================================================================
# 7. TIME DECAY: how fast does the directional edge decay?
# =============================================================================
print('\n' + '=' * 90)
print('7. EDGE DECAY: directional accuracy over time (pooled, |z| > 0.5)')
print('=' * 90)

all_meaningful = events_df[events_df['surprise_z'].abs() > 0.5].copy()

print(f'\n{"Horizon":<10} {"N":>5} {"Win%":>7} {"Avg PnL":>10} {"Avg |PnL|":>10} {"Edge Ratio":>10}')
print('─' * 60)

for h in HORIZONS:
    col = f'{h}m_chg'
    valid = all_meaningful.dropna(subset=[col])
    
    pnls = []
    for _, ev in valid.iterrows():
        surprise = ev['surprise_z']
        exp_dir = ev['expected_dir']
        if exp_dir == 'inverse':
            trade_dir = -1 if surprise > 0 else 1
        else:
            trade_dir = 1 if surprise > 0 else -1
        pnls.append(trade_dir * ev[col])
    
    pnls = np.array(pnls)
    win = (pnls > 0).mean() * 100
    avg = pnls.mean()
    avg_abs = np.abs(pnls).mean()
    edge_ratio = avg / avg_abs if avg_abs > 0 else 0  # how much of the move is directional
    
    print(f'{h:>4}m     {len(pnls):>5} {win:>6.1f}% ${avg:>+8.2f}  ${avg_abs:>8.2f}  {edge_ratio:>+9.3f}')

print('\n  Edge ratio = avg_pnl / avg_|pnl|')
print('  Closer to +1.0 = strong directional edge')
print('  Closer to  0.0 = random direction (but vol exists)')

# =============================================================================
# 8. YEARLY BREAKDOWN (pooled |z| > 0.5, 30-min directional)
# =============================================================================
print('\n' + '=' * 90)
print('8. YEARLY CONSISTENCY (pooled |z| > 0.5, 30-min directional trades)')
print('=' * 90)

all_meaningful['year'] = all_meaningful['release_date'].dt.year
all_meaningful_valid = all_meaningful.dropna(subset=['30m_chg'])

print(f'\n{"Year":>6} {"N":>5} {"Total PnL":>10} {"Avg PnL":>9} {"Win%":>7}')
print('─' * 45)

for year, ygrp in all_meaningful_valid.groupby('year'):
    pnls = []
    for _, ev in ygrp.iterrows():
        surprise = ev['surprise_z']
        exp_dir = ev['expected_dir']
        if exp_dir == 'inverse':
            trade_dir = -1 if surprise > 0 else 1
        else:
            trade_dir = 1 if surprise > 0 else -1
        pnls.append(trade_dir * ev['30m_chg'])
    
    pnls = np.array(pnls)
    print(f'{year:>6} {len(pnls):>5} ${pnls.sum():>+9.2f} ${pnls.mean():>+7.2f} '
          f'{(pnls > 0).mean()*100:>6.1f}%')

print('\n' + '=' * 90)
print('DONE')
print('=' * 90)
