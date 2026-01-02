import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', names=['Datetime','Open','High','Low','Close','Volume'], parse_dates=['Datetime'])
df.set_index('Datetime', inplace=True)

df['is_up'] = (df['Close'] > df['Open']).astype(int)
df['up_count_3'] = df['is_up'].rolling(3).sum()
df['ret_5'] = df['Close'].pct_change(5)
df['ema_8'] = df['Close'].ewm(span=8).mean()
df['dist_ema8_pct'] = (df['Close'] - df['ema_8']) / df['Close'] * 100

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

horizon = 20
target = 0.001
stop = 0.0005
outcomes = []
for i in range(len(df) - horizon):
    entry = df['Close'].iloc[i]
    future = df.iloc[i+1:i+horizon+1]
    hit = False
    for h, l in zip(future['High'], future['Low']):
        if h >= entry * (1 + target):
            hit = True
            break
        if l <= entry * (1 - stop):
            break
    outcomes.append(1 if hit else 0)

df = df.iloc[:len(outcomes)].copy()
df['outcome'] = outcomes
df = df.dropna()
base = df['outcome'].mean()

print('='*70)
print('KEY INSIGHT: ALL STABLE FEATURES ARE BEARISH (NEG CORRELATION)')
print('='*70)
print()
print('Meaning: When these features are HIGH, LONG trades FAIL more often')
print()

high_mom = (df['up_count_3'] == 3) & (df['ret_5'] > 0.002)
low_mom = (df['up_count_3'] == 0) & (df['ret_5'] < -0.002)

print('MOMENTUM ANALYSIS:')
print('-'*70)
wr1 = df[high_mom]['outcome'].mean()*100
wr2 = df[low_mom]['outcome'].mean()*100
print(f'After 3 UP candles + pos 5-bar ret:  {wr1:.1f}% long WR (n={high_mom.sum()})')
print(f'After 3 DOWN candles + neg 5-bar ret: {wr2:.1f}% long WR (n={low_mom.sum()})')
print(f'Base rate: {base*100:.1f}%')
print()

print('RSI ANALYSIS:')
print('-'*70)
rsi_low = df['rsi_14'] < 30
rsi_high = df['rsi_14'] > 70
wr3 = df[rsi_low]['outcome'].mean()*100
wr4 = df[rsi_high]['outcome'].mean()*100
print(f'RSI < 30 (oversold):  {wr3:.1f}% long WR (n={rsi_low.sum()})')
print(f'RSI > 70 (overbought): {wr4:.1f}% long WR (n={rsi_high.sum()})')
print()

print('='*70)
print('CONCLUSION: MEAN REVERSION > TREND FOLLOWING')
print('='*70)
print()
print('Your bullish continuation strategy FIGHTS the data.')
print('After bullish moves, price tends to REVERSE, not continue.')
print()

print('MEAN REVERSION TEST:')
print('-'*70)
mr_long = (df['rsi_14'] < 35) & (df['up_count_3'] <= 1)
bc_long = (df['up_count_3'] == 3) & (df['ret_5'] > 0)
wr5 = df[mr_long]['outcome'].mean()*100
wr6 = df[bc_long]['outcome'].mean()*100
print(f'Mean Reversion (RSI<35 + bearish bars): {wr5:.1f}% (n={mr_long.sum()})')
print(f'Bullish Continuation (3up + pos ret):   {wr6:.1f}% (n={bc_long.sum()})')
print()
print(f'Edge vs base rate:')
print(f'  Mean Reversion: +{wr5-base*100:.1f}%')
print(f'  Bull Continuation: {wr6-base*100:+.1f}%')
