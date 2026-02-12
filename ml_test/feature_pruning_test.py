"""
Feature Pruning Analysis for XAUUSD Hedge Strategy
Tests Top-N feature subsets to find optimal pruning level
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings, sys, os
warnings.filterwarnings('ignore')

# Import feature engineering from main script
TARGET = 5.0
STOP = 2.5
HORIZON = 20

print("="*70)
print("FEATURE PRUNING ANALYSIS")
print("="*70)

print("\nLoading data...")
df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', 
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open','High','Low','Close']].copy()
print(f"Loaded {len(df)} bars")

# === Feature engineering (identical to hedging_strategy_strict.py) ===
print("Engineering features...")
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
df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - \
                    ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']
df['high_10'] = df['High'].rolling(10).max()
df['low_10'] = df['Low'].rolling(10).min()
df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)
df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)

# New features
df['flow_15'] = df['directional_flow'].rolling(15).sum()
df['flow_20'] = df['directional_flow'].rolling(20).sum()
df['flow_accel'] = df['flow_3'] - df['flow_3'].shift(3)
df['flow_accel_5'] = df['flow_5'] - df['flow_5'].shift(5)
df['abs_flow_3'] = df['flow_3'].abs()
df['abs_flow_5'] = df['flow_5'].abs()
df['abs_flow_10'] = df['flow_10'].abs()
df['flow_divergence'] = (df['flow_3'] * df['flow_10'] < 0).astype(int)
df['consecutive_up'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumcount() + 1
df['consecutive_up'] = df['consecutive_up'] * df['is_up']
df['consecutive_down'] = (1 - df['is_up']).groupby(((1-df['is_up']) != (1-df['is_up']).shift()).cumsum()).cumcount() + 1
df['consecutive_down'] = df['consecutive_down'] * (1 - df['is_up'])
df['max_consecutive'] = df[['consecutive_up', 'consecutive_down']].max(axis=1)
df['flow_efficiency'] = df['abs_body'] / (df['range'] + 1e-10)
df['flow_eff_3'] = df['flow_efficiency'].rolling(3).mean()
df['flow_eff_5'] = df['flow_efficiency'].rolling(5).mean()
df['atr_roc'] = (df['atr_3'] - df['atr_3'].shift(3)) / (df['atr_3'].shift(3) + 1e-10)
df['vol_breakout'] = df['range'] / (df['atr_20'] + 1e-10)
df['range_min_5'] = df['range'].rolling(5).min()
df['range_max_5'] = df['range'].rolling(5).max()
df['range_squeeze'] = df['range_min_5'] / (df['range_max_5'] + 1e-10)
df['abs_dist_ema8'] = df['dist_ema8'].abs()
df['dist_ema21'] = (df['Close'] - df['ema_21']) / df['Close']
df['abs_dist_ema21'] = df['dist_ema21'].abs()
df['ema_spread'] = (df['ema_8'] - df['ema_21']) / df['Close']
df['abs_ema_spread'] = df['ema_spread'].abs()

df.dropna(inplace=True)

# Combo features
df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']
df['combo_abs_flow_vol'] = df['abs_flow_5'] * df['vol_ratio']
df['combo_eff_flow'] = df['flow_eff_3'] * df['abs_flow_3']
df['combo_consecutive_body'] = df['max_consecutive'] * df['body_pct']
df['combo_atr_roc_accel'] = df['atr_roc'] * df['flow_accel'].abs()
df['combo_squeeze_flow'] = (1 - df['range_squeeze']) * df['abs_flow_3']
df['combo_dist_flow'] = df['abs_dist_ema8'] * df['abs_flow_5']

# Labeling
print(f"Labeling outcomes ({HORIZON} bars, ${TARGET} target, ${STOP} stop)...")
outcomes = []
for i in range(len(df) - HORIZON):
    if i % 50000 == 0:
        print(f"  {i}/{len(df)-HORIZON}...")
    entry = df['Close'].iloc[i]
    future = df.iloc[i+1:i+HORIZON+1]
    long_hit = False
    for h, l in zip(future['High'], future['Low']):
        if h >= entry + TARGET:
            long_hit = True
            break
        if l <= entry - STOP:
            break
    short_hit = False
    if not long_hit:
        for h, l in zip(future['High'], future['Low']):
            if l <= entry - TARGET:
                short_hit = True
                break
            if h >= entry + STOP:
                break
    outcomes.append(1 if long_hit else (2 if short_hit else 0))

df = df.iloc[:len(outcomes)].copy()
df['outcome'] = outcomes

# All 55 features
all_features = [
    'flow_3', 'flow_5', 'flow_10', 'flow_momentum',
    'imbalance_3', 'imbalance_5',
    'consistency_3', 'consistency_5',
    'vol_ratio', 'vol_expansion', 'vol_contraction',
    'trend_align', 'dist_ema8',
    'at_high', 'at_low',
    'upper_reject', 'lower_reject',
    'big_body', 'small_body',
    'body_pct', 'close_position',
    'flow_15', 'flow_20', 'flow_accel', 'flow_accel_5',
    'abs_flow_3', 'abs_flow_5', 'abs_flow_10', 'flow_divergence',
    'max_consecutive', 'flow_efficiency', 'flow_eff_3', 'flow_eff_5',
    'atr_roc', 'vol_breakout', 'range_squeeze',
    'abs_dist_ema8', 'dist_ema21', 'abs_dist_ema21', 'ema_spread', 'abs_ema_spread',
    'combo_flow_trend', 'combo_vol_imbalance', 'combo_consistency_position',
    'combo_body_reject', 'combo_trend_volatility', 'combo_imbalance_momentum',
    'combo_position_consistency', 'combo_vol_flow',
    'combo_abs_flow_vol', 'combo_eff_flow', 'combo_consecutive_body',
    'combo_atr_roc_accel', 'combo_squeeze_flow', 'combo_dist_flow',
]

# Train/test split
split = int(len(df) * 0.6)
train = df.iloc[:split]
test = df.iloc[split:]
y_train = (train['outcome'] != 0).astype(int)
y_test = (test['outcome'] != 0).astype(int)

# =========================================================================
# STEP 1: Train full model to get feature importance ranking
# =========================================================================
print("\nTraining full model (55 features) to rank importance...")
rf_full = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_full.fit(train[all_features].fillna(0), y_train)

importance_ranking = sorted(zip(all_features, rf_full.feature_importances_), 
                           key=lambda x: x[1], reverse=True)

print("\n" + "="*70)
print("FULL FEATURE IMPORTANCE RANKING (all 55)")
print("="*70)
cumulative = 0
for i, (feat, imp) in enumerate(importance_ranking):
    cumulative += imp
    marker = " <<<" if imp >= 0.02 else ""
    print(f"  {i+1:>2}. {feat:<35} {imp:.4f}  (cum: {cumulative:.2%}){marker}")

# =========================================================================
# STEP 2: Test different Top-N configurations
# =========================================================================
print("\n" + "="*70)
print("FEATURE PRUNING: TESTING TOP-N CONFIGURATIONS")
print("="*70)
print("(NetR uses corrected hedge math: WIN=+1R, LOSS=-2R)\n")

results = []
for n in [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 55]:
    top_n_features = [f for f, _ in importance_ranking[:n]]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(train[top_n_features].fillna(0), y_train)
    rf_probs = rf.predict_proba(test[top_n_features].fillna(0))[:, 1]
    
    auc = roc_auc_score(y_test, rf_probs)
    
    row = {'N': n, 'AUC': auc}
    for thresh in [0.65, 0.70, 0.75, 0.80]:
        mask = rf_probs >= thresh
        count = mask.sum()
        if count > 0:
            wins = (test[mask]['outcome'] != 0).sum()
            losses = count - wins
            wr = wins / count * 100
            net_r = wins * 1 + losses * (-2)
            pnl_per_trade = net_r / count
            row[f'WR_{thresh}'] = wr
            row[f'N_{thresh}'] = count
            row[f'R_{thresh}'] = net_r
            row[f'PPT_{thresh}'] = pnl_per_trade
        else:
            row[f'WR_{thresh}'] = 0
            row[f'N_{thresh}'] = 0
            row[f'R_{thresh}'] = 0
            row[f'PPT_{thresh}'] = 0
    results.append(row)

# Print comparison table
header = f"{'Top-N':>6} {'AUC':>6} | {'WR':>6} {'#':>5} {'NetR':>6} {'R/T':>6} | {'WR':>6} {'#':>5} {'NetR':>6} {'R/T':>6} | {'WR':>6} {'#':>5} {'NetR':>6} {'R/T':>6} | {'WR':>6} {'#':>5} {'NetR':>6} {'R/T':>6}"
print(f"{'':>14} {'@0.65':^25} | {'@0.70':^25} | {'@0.75':^25} | {'@0.80':^25}")
print(f"{header}")
print("─" * 130)
for r in results:
    parts = [f"{r['N']:>6} {r['AUC']:>.4f}"]
    for thresh in [0.65, 0.70, 0.75, 0.80]:
        wr = r.get(f'WR_{thresh}', 0)
        cnt = r.get(f'N_{thresh}', 0)
        nr = r.get(f'R_{thresh}', 0)
        ppt = r.get(f'PPT_{thresh}', 0)
        parts.append(f"{wr:>5.1f}% {cnt:>5} {nr:>+5}R {ppt:>+5.2f}")
    print(" | ".join(parts))

# =========================================================================
# STEP 3: Find best configuration
# =========================================================================
print("\n" + "="*70)
print("BEST CONFIGURATIONS BY NET R")
print("="*70)
for thresh in [0.65, 0.70, 0.75, 0.80]:
    valid = [r for r in results if r.get(f'N_{thresh}', 0) >= 10]
    if valid:
        best = max(valid, key=lambda r: r.get(f'R_{thresh}', -999))
        print(f"  @{thresh}: Top-{best['N']:>2} features -> WR={best.get(f'WR_{thresh}',0):>5.1f}%, "
              f"Trades={best.get(f'N_{thresh}',0):>5}, "
              f"NetR={best.get(f'R_{thresh}',0):>+5}R, "
              f"R/trade={best.get(f'PPT_{thresh}',0):>+.3f}")

# Best by R per trade (quality)
print("\n" + "="*70)
print("BEST CONFIGURATIONS BY R/TRADE (QUALITY)")
print("="*70)
for thresh in [0.65, 0.70, 0.75, 0.80]:
    valid = [r for r in results if r.get(f'N_{thresh}', 0) >= 10]
    if valid:
        best = max(valid, key=lambda r: r.get(f'PPT_{thresh}', -999))
        print(f"  @{thresh}: Top-{best['N']:>2} features -> WR={best.get(f'WR_{thresh}',0):>5.1f}%, "
              f"Trades={best.get(f'N_{thresh}',0):>5}, "
              f"NetR={best.get(f'R_{thresh}',0):>+5}R, "
              f"R/trade={best.get(f'PPT_{thresh}',0):>+.3f}")

# Show the recommended feature set
print("\n" + "="*70)
best_at_70 = max([r for r in results if r.get('N_0.7', 0) >= 10], key=lambda r: r.get('R_0.7', -999))
best_n = best_at_70['N']
print(f"RECOMMENDED FEATURE SET: Top {best_n} (best NetR at RF >= 0.70)")
print("="*70)
recommended_features = []
for i, (feat, imp) in enumerate(importance_ranking[:best_n]):
    print(f"  {i+1:>2}. {feat:<35} {imp:.4f}")
    recommended_features.append(feat)

# Save recommended feature list
with open('models/recommended_features.txt', 'w') as f:
    f.write('\n'.join(recommended_features))
print(f"\nSaved: models/recommended_features.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
