import pandas as pd
import numpy as np

print('='*80)
print('DEEP FEATURE PREDICTIVE POWER ANALYSIS')
print('='*80)

df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', names=['Datetime','Open','High','Low','Close','Volume'], parse_dates=['Datetime'])
df.set_index('Datetime', inplace=True)
print(f'Loaded {len(df)} bars')

df['body'] = df['Close'] - df['Open']
df['range'] = df['High'] - df['Low']
df['abs_body'] = abs(df['body'])
df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)
df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
df['is_up'] = (df['Close'] > df['Open']).astype(int)
df['up_count_3'] = df['is_up'].rolling(3).sum()
df['up_count_5'] = df['is_up'].rolling(5).sum()
df['ret_1'] = df['Close'].pct_change(1)
df['ret_3'] = df['Close'].pct_change(3)
df['ret_5'] = df['Close'].pct_change(5)
df['atr_10'] = df['range'].rolling(10).mean()
df['vol_ratio'] = df['range'] / (df['atr_10'] + 1e-10)
df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
df['ema_8'] = df['Close'].ewm(span=8).mean()
df['ema_21'] = df['Close'].ewm(span=21).mean()
df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int)
df['dist_ema8_pct'] = (df['Close'] - df['ema_8']) / df['Close'] * 100

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

horizon = 20
target = 0.001
stop = 0.0005

print('Labeling outcomes...')
outcomes = []
for i in range(len(df) - horizon):
    entry = df['Close'].iloc[i]
    future = df.iloc[i+1:i+horizon+1]
    hit_target = False
    for h, l in zip(future['High'], future['Low']):
        if h >= entry * (1 + target):
            hit_target = True
            break
        if l <= entry * (1 - stop):
            break
    outcomes.append(1 if hit_target else 0)

df = df.iloc[:len(outcomes)].copy()
df['outcome'] = outcomes
df = df.dropna()

base = df['outcome'].mean()
print(f'Base rate: {base*100:.1f}% long wins')

n = len(df)
ps = n // 5

print('\n' + '='*80)
print('FEATURE CORRELATION BY TIME PERIOD')
print('='*80)

features = ['is_up', 'up_count_3', 'up_count_5', 'ret_1', 'ret_3', 'ret_5', 
            'big_body', 'body_pct', 'close_position', 'vol_ratio',
            'trend_align', 'dist_ema8_pct', 'rsi_14']

print(f'\n{chr(34)}Feature{chr(34):<18} P1       P2       P3       P4       P5       Avg      Stable?')
print('-'*82)

stable_features = []
for feat in features:
    corrs = []
    for p in range(5):
        start = p * ps
        end = (p + 1) * ps if p < 4 else n
        period_df = df.iloc[start:end]
        corr = period_df[feat].corr(period_df['outcome'])
        corrs.append(corr)
    
    avg = np.mean(corrs)
    std = np.std(corrs)
    same_sign = all(c > 0 for c in corrs) or all(c < 0 for c in corrs)
    is_stable = same_sign and abs(avg) > 0.01
    
    if is_stable:
        stable_features.append((feat, avg, std))
    
    stable_str = 'YES' if is_stable else 'no'
    print(f'{feat:<18} {corrs[0]:>8.4f} {corrs[1]:>8.4f} {corrs[2]:>8.4f} {corrs[3]:>8.4f} {corrs[4]:>8.4f} {avg:>8.4f} {stable_str:>8}')

print('\n' + '='*80)
print('FEATURES WITH CONSISTENT PREDICTIVE DIRECTION')
print('='*80)
if stable_features:
    for feat, avg, std in sorted(stable_features, key=lambda x: abs(x[1]), reverse=True):
        direction = 'BULLISH' if avg > 0 else 'BEARISH'
        print(f'{feat:<18}: {avg:>7.4f} avg corr, std={std:.4f} -> {direction} signal')
else:
    print('NO features show consistent predictive power across all periods!')

print('\n' + '='*80)
print('PATTERN SUCCESS RATES BY PERIOD')
print('='*80)

patterns = [
    ('3 bullish candles', df['up_count_3'] == 3),
    ('3 bearish candles', df['up_count_3'] == 0),
    ('Big body up', (df['big_body'] == 1) & (df['is_up'] == 1)),
    ('Big body down', (df['big_body'] == 1) & (df['is_up'] == 0)),
    ('Close near high', df['close_position'] > 0.7),
    ('Close near low', df['close_position'] < 0.3),
    ('Uptrend aligned', df['trend_align'] == 1),
    ('RSI > 70', df['rsi_14'] > 70),
    ('RSI < 30', df['rsi_14'] < 30),
]

print(f'\nBase rate: {base*100:.1f}%')
print(f'\nPattern              P1       P2       P3       P4       P5       Stable?')
print('-'*72)

for name, mask in patterns:
    rates = []
    for p in range(5):
        start = p * ps
        end = (p + 1) * ps if p < 4 else n
        period_df = df.iloc[start:end]
        period_mask = mask.iloc[start:end]
        if period_mask.sum() > 10:
            rate = period_df[period_mask]['outcome'].mean() * 100
        else:
            rate = float('nan')
        rates.append(rate)
    
    valid_rates = [r for r in rates if not np.isnan(r)]
    if len(valid_rates) >= 4:
        all_above = all(r > base*100 for r in valid_rates)
        all_below = all(r < base*100 for r in valid_rates)
        stable = 'YES' if (all_above or all_below) else 'no'
    else:
        stable = 'n/a'
    
    print(f'{name:<20} {rates[0]:>7.1f}% {rates[1]:>7.1f}% {rates[2]:>7.1f}% {rates[3]:>7.1f}% {rates[4]:>7.1f}% {stable:>8}')
