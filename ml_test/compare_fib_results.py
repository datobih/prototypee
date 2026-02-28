"""
Compare old vs new fib filtering results
"""
import pandas as pd
import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

# Load new top combo
new_top1 = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_combos',
                                     'top1_Fib_50-79pct__High_OB_Vol.csv'))

print('=' * 90)
print('CORRECTED FIB LOGIC RESULTS')
print('=' * 90)

print('\nNEW Top #1: Fib 50-79% + High OB Vol')
print(f'  Trades:        {len(new_top1)}')
bounces = (new_top1['reaction'] == 'BOUNCE').sum()
print(f'  Bounces:       {bounces} ({bounces/len(new_top1)*100:.1f}%)')
avg_rr = new_top1[new_top1['reaction'] == 'BOUNCE']['trade_rr'].mean()
print(f'  Avg RR (bnc):  {avg_rr:.2f}')

print('\nFib retracement distribution:')
fib_stats = new_top1['fib_retracement_pct'].describe()
print(f'  Min:    {fib_stats["min"]:.1f}%')
print(f'  25th:   {fib_stats["25%"]:.1f}%')
print(f'  Median: {fib_stats["50%"]:.1f}%')
print(f'  75th:   {fib_stats["75%"]:.1f}%')
print(f'  Max:    {fib_stats["max"]:.1f}%')

print('\nbars_to_mitigate distribution:')
btm_stats = new_top1['bars_to_mitigate'].describe()
print(f'  Min:    {btm_stats["min"]:.0f}')
print(f'  25th:   {btm_stats["25%"]:.0f}')
print(f'  Median: {btm_stats["50%"]:.0f}')
print(f'  75th:   {btm_stats["75%"]:.0f}')
print(f'  Max:    {btm_stats["max"]:.0f}')

# Check all OBs
all_obs = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed',
                                    'ob_mitigation_study.csv'))
all_obs['ob_dt'] = pd.to_datetime(all_obs['ob_dt'])

print('\n' + '=' * 90)
print('ALL OBs — FIB RETRACEMENT AVAILABILITY')
print('=' * 90)

touched = all_obs[all_obs['touched'] == True]
has_fib = touched['fib_retracement_pct'].notna()
print(f'\nTotal touched OBs:         {len(touched)}')
print(f'  With valid fib (>=5 bars): {has_fib.sum()} ({has_fib.sum()/len(touched)*100:.1f}%)')
print(f'  Degenerate (<5 bars):      {(~has_fib).sum()} ({(~has_fib).sum()/len(touched)*100:.1f}%)')

# Fib distribution for valid cases
valid_fib = touched[has_fib]['fib_retracement_pct']
print(f'\nValid fib distribution (N={len(valid_fib)}):')
print(f'  Median: {valid_fib.median():.1f}%')
print(f'  Mean:   {valid_fib.mean():.1f}%')

# Count by fib bucket
for lo, hi, label in [(0, 38, '<38% (shallow)'), (38, 62, '38-62% (golden)'),
                       (50, 79, '50-79% (deep)'), (79, 100, '79-100% (very deep)'),
                       (100, 999, '>100% (broke structure)')]:
    n = ((valid_fib >= lo) & (valid_fib < hi)).sum()
    pct = n / len(valid_fib) * 100
    print(f'  {label:<30} {n:>4} ({pct:>5.1f}%)')

print('\n' + '=' * 90)
print('COMPARISON: OLD vs NEW')
print('=' * 90)

print('\nOLD (degenerate fib 38-62% + FVG):')
print('  Trades:        20')
print('  Bounces:       11 (55.0%)')
print('  Avg RR:        3.4')
print('  Degenerate:    12/20 (60%)')

print(f'\nNEW (valid fib 50-79% + High OB Vol):')
print(f'  Trades:        {len(new_top1)}')
print(f'  Bounces:       {bounces} ({bounces/len(new_top1)*100:.1f}%)')
print(f'  Avg RR:        {avg_rr:.2f}')
print(f'  All valid:     {len(new_top1)}/{len(new_top1)} (100%)')

print('\n' + '=' * 90)
print('TOP 3 COMBOS (NEW)')
print('=' * 90)

for i in [1, 2, 3]:
    fname = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_combos'))
             if f.startswith(f'top{i}_')][0]
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_combos', fname))
    b = (df['reaction'] == 'BOUNCE').sum()
    rr = df[df['reaction'] == 'BOUNCE']['trade_rr'].mean() if b > 0 else 0
    label = fname.replace(f'top{i}_', '').replace('.csv', '').replace('_', ' ')
    print(f'\n  #{i}: {label}')
    print(f'       N={len(df):>3}  Bounce={b}/{len(df)} ({b/len(df)*100:.1f}%)  Avg RR={rr:.1f}')

print('\n' + '=' * 90)
