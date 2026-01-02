import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    df = df.copy()
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)
    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)
    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()
    # is_up = bullish candle (close > open)
    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))
    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']
    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)
    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)
    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']
    return df.dropna()

def label_outcomes(df, horizon, target, stop):
    outcomes = []
    for i in range(len(df) - horizon):
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]
        long_hit = False
        for h, l in zip(future['High'], future['Low']):
            if h >= entry * (1 + target):
                long_hit = True
                break
            if l <= entry * (1 - stop):
                break
        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry * (1 - target):
                    short_hit = True
                    break
                if h >= entry * (1 + stop):
                    break
        outcomes.append(1 if long_hit else (2 if short_hit else 0))
    return outcomes

print('='*100)
print('XAUUSD 1-MINUTE PARAMETER OPTIMIZATION')
print('='*100)

print('Loading and preparing features...')
df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', names=['Datetime','Open','High','Low','Close','Volume'], parse_dates=['Datetime'])
df.set_index('Datetime', inplace=True)
df = create_features(df)
print(f'Data ready: {len(df)} bars')

feature_cols = ['flow_3', 'flow_5', 'flow_10', 'flow_momentum', 'imbalance_3', 'imbalance_5',
    'consistency_3', 'consistency_5', 'vol_ratio', 'vol_expansion', 'vol_contraction',
    'trend_align', 'dist_ema8', 'at_high', 'at_low', 'upper_reject', 'lower_reject',
    'big_body', 'small_body', 'body_pct', 'close_position',
    'combo_flow_trend', 'combo_vol_imbalance', 'combo_consistency_position',
    'combo_body_reject', 'combo_trend_volatility', 'combo_imbalance_momentum',
    'combo_position_consistency', 'combo_vol_flow']

# Higher R:R ratios: 3:1, 4:1, 5:1
# Format: (horizon, target, stop, description)
params = [
    # 3:1 R:R configurations
    (10, 0.0015, 0.0005, '10min, 0.15%/0.05% (3:1)'),
    (15, 0.0015, 0.0005, '15min, 0.15%/0.05% (3:1)'),
    (20, 0.0015, 0.0005, '20min, 0.15%/0.05% (3:1)'),
    (20, 0.003, 0.001, '20min, 0.3%/0.1% (3:1)'),
    (30, 0.003, 0.001, '30min, 0.3%/0.1% (3:1)'),
    (30, 0.0045, 0.0015, '30min, 0.45%/0.15% (3:1)'),
    (45, 0.003, 0.001, '45min, 0.3%/0.1% (3:1)'),
    (60, 0.003, 0.001, '60min, 0.3%/0.1% (3:1)'),
    (60, 0.0045, 0.0015, '60min, 0.45%/0.15% (3:1)'),
    
    # 4:1 R:R configurations
    (15, 0.002, 0.0005, '15min, 0.2%/0.05% (4:1)'),
    (20, 0.002, 0.0005, '20min, 0.2%/0.05% (4:1)'),
    (20, 0.004, 0.001, '20min, 0.4%/0.1% (4:1)'),
    (30, 0.004, 0.001, '30min, 0.4%/0.1% (4:1)'),
    (30, 0.006, 0.0015, '30min, 0.6%/0.15% (4:1)'),
    (45, 0.004, 0.001, '45min, 0.4%/0.1% (4:1)'),
    (60, 0.004, 0.001, '60min, 0.4%/0.1% (4:1)'),
    (60, 0.006, 0.0015, '60min, 0.6%/0.15% (4:1)'),
    
    # 5:1 R:R configurations
    (20, 0.0025, 0.0005, '20min, 0.25%/0.05% (5:1)'),
    (20, 0.005, 0.001, '20min, 0.5%/0.1% (5:1)'),
    (30, 0.005, 0.001, '30min, 0.5%/0.1% (5:1)'),
    (30, 0.0075, 0.0015, '30min, 0.75%/0.15% (5:1)'),
    (45, 0.005, 0.001, '45min, 0.5%/0.1% (5:1)'),
    (60, 0.005, 0.001, '60min, 0.5%/0.1% (5:1)'),
    (60, 0.0075, 0.0015, '60min, 0.75%/0.15% (5:1)'),
    
    # 6:1 R:R configurations (aggressive)
    (30, 0.006, 0.001, '30min, 0.6%/0.1% (6:1)'),
    (45, 0.006, 0.001, '45min, 0.6%/0.1% (6:1)'),
    (60, 0.006, 0.001, '60min, 0.6%/0.1% (6:1)'),
    (60, 0.009, 0.0015, '60min, 0.9%/0.15% (6:1)'),
]

results = []
print(f'\nTesting {len(params)} parameter combinations...')
print('-'*100)

for horizon, target, stop, desc in params:
    outcomes = label_outcomes(df, horizon, target, stop)
    df_labeled = df.iloc[:len(outcomes)].copy()
    df_labeled['outcome'] = outcomes
    
    split = int(len(df_labeled) * 0.6)
    train = df_labeled.iloc[:split]
    test = df_labeled.iloc[split:]
    
    base_rate = (test['outcome'] != 0).sum() / len(test) * 100
    
    if base_rate < 5:
        print(f'{desc:<25} Base: {base_rate:>5.1f}% - SKIPPING (too low)')
        continue
    
    X_train = train[feature_cols].fillna(0)
    y_train = (train['outcome'] != 0).astype(int)
    X_test = test[feature_cols].fillna(0)
    y_test = (test['outcome'] != 0).astype(int)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, rf_probs)
    
    best_thresh = None
    best_rate = 0
    best_count = 0
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = rf_probs >= thresh
        if mask.sum() >= 10:
            rate = (test[mask]['outcome'] != 0).sum() / mask.sum() * 100
            if rate > best_rate:
                best_rate = rate
                best_thresh = thresh
                best_count = mask.sum()
    
    results.append((desc, base_rate, auc, best_thresh, best_rate, best_count, horizon, target, stop))
    print(f'{desc:<25} Base: {base_rate:>5.1f}%  AUC: {auc:.3f}  Best: {best_rate:>5.1f}% @ {best_thresh} ({best_count} trades)')

results.sort(key=lambda x: x[4], reverse=True)

print('\n' + '='*100)
print('TOP 10 CONFIGURATIONS BY WIN RATE')
print('='*100)
print(f'{"Config":<25} {"Base%":<8} {"AUC":<7} {"Best%":<8} {"Thresh":<8} {"Count"}')
print('-'*70)
for desc, base, auc, thresh, rate, count, h, t, s in results[:10]:
    print(f'{desc:<25} {base:>6.1f}  {auc:>6.3f}  {rate:>6.1f}  {thresh:>6.2f}  {count:>6d}')
