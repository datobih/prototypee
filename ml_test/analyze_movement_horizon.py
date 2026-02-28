import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', 
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['range'] = df['High'] - df['Low']

print('='*60)
print('XAUUSD 1-min Movement Analysis')
print('='*60)

print('\nSingle 1-min bar stats:')
print(f'  Median range: ${df["range"].median():.2f}')
print(f'  Mean range:   ${df["range"].mean():.2f}')
print(f'  75th pct:     ${df["range"].quantile(0.75):.2f}')
print(f'  90th pct:     ${df["range"].quantile(0.90):.2f}')
print(f'  95th pct:     ${df["range"].quantile(0.95):.2f}')

highs = df['High'].values
lows = df['Low'].values

print('\n' + '='*60)
print('Forward-looking range by horizon')
print('='*60)

for horizon in [5, 10, 15, 20, 30, 45, 60]:
    hw = sliding_window_view(highs[1:], horizon)
    lw = sliding_window_view(lows[1:], horizon)
    fwd_range = hw.max(axis=1) - lw.min(axis=1)
    
    print(f'\n{horizon}-bar horizon ({horizon} minutes):')
    print(f'  Median range: ${np.median(fwd_range):.2f}')
    print(f'  Mean range:   ${np.mean(fwd_range):.2f}')
    print(f'  75th pct:     ${np.quantile(fwd_range, 0.75):.2f}')
    print(f'  90th pct:     ${np.quantile(fwd_range, 0.90):.2f}')
    
    print(f'  % >= $5.0:    {(fwd_range >= 5.0).mean()*100:.1f}%')
    print(f'  % >= $3.0:    {(fwd_range >= 3.0).mean()*100:.1f}%')
    print(f'  % >= $2.0:    {(fwd_range >= 2.0).mean()*100:.1f}%')
    print(f'  % >= $1.0:    {(fwd_range >= 1.0).mean()*100:.1f}%')

print('\n' + '='*60)
print('Recommendation:')
print('='*60)
print('For a movement predictor, you want:')
print('  - High enough threshold to be meaningful (not noise)')
print('  - Short enough horizon to be actionable (not too far ahead)')
print('  - Balanced label distribution (not too rare, not too common)')
print('\nOptimal range: 30-60% positive rate')
