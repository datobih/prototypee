import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURABLE PARAMETERS - Modify these values as needed
# =============================================================================
TARGET = 3          # Target profit in dollars
STOP = 1.5          # Stop loss in dollars
HORIZON = 30        # Number of bars to look ahead for outcome
RF_THRESHOLD = 0.70 # Random Forest probability threshold for hedge entry
TSFRESH_CACHE = 'data/processed/tsfresh_features_cache.parquet'

def create_features(df, cache_path=TSFRESH_CACHE):
    """
    Load pre-engineered tsfresh features from parquet cache (built by engineer_features.py),
    join to OHLCV, forward-fill stride gaps, and add session/time features.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f'Feature cache not found: {cache_path}\n'
            'Run engineer_features.py first to build the cache.'
        )

    print(f'  Loading tsfresh features from cache: {cache_path}')
    feat_df = pd.read_parquet(cache_path)

    df = df.copy()
    df = df.join(feat_df, how='left')
    df[feat_df.columns.tolist()] = df[feat_df.columns.tolist()].ffill()

    # --- SESSION / TIME (tsfresh can't capture these) ---
    hour = df.index.hour
    df['hour_sin']   = np.sin(2 * np.pi * hour / 24)
    df['hour_cos']   = np.cos(2 * np.pi * hour / 24)
    df['dow_sin']    = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['dow_cos']    = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['is_london']  = ((hour >= 7)  & (hour < 13)).astype(int)
    df['is_ny']      = ((hour >= 13) & (hour < 20)).astype(int)
    df['is_overlap'] = ((hour >= 13) & (hour < 17)).astype(int)

    return df.dropna()


def label_outcomes(df, horizon=15, target=5.0, stop=2.5):
    """Label outcomes using fixed dollar SL/TP (not percentage-based)"""
    df = df.copy()
    outcomes = []
    
    for i in range(len(df) - horizon):
        if i % 20000 == 0:
            print(f'  Labeling {i}/{len(df)-horizon}...')
        
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]
        
        # Check LONG (fixed dollar targets)
        long_hit = False
        for h, l in zip(future['High'], future['Low']):
            if h >= entry + target:  # TP hit
                long_hit = True
                break
            if l <= entry - stop:  # SL hit
                break
        
        # Check SHORT (fixed dollar targets)
        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry - target:  # TP hit
                    short_hit = True
                    break
                if h >= entry + stop:  # SL hit
                    break
        
        outcomes.append(1 if long_hit else (2 if short_hit else 0))
    
    df = df.iloc[:len(outcomes)].copy()
    df['outcome'] = outcomes
    return df

print('='*80)
print('XAUUSD 1MIN FEATURE CORRELATION ANALYSIS')
print('='*80)

print('\nLoading data...')
# New format: Date, Time, Open, High, Low, Close, TickVol, Vol, Spread
df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', 
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open','High','Low','Close','TickVol']].copy()  # Keep OHLCV
print(f'Loaded {len(df)} bars (1-minute timeframe)')

print('\nEngineering features (tsfresh-validated, window=30)...')
df = create_features(df)

print(f'\nLabeling outcomes ({HORIZON} bars, ${TARGET} target, ${STOP} stop)...')
df = label_outcomes(df, HORIZON, TARGET, STOP)

# Split 60:40 with purge gap to prevent label leakage at boundary
split = int(len(df) * 0.6)
purge = HORIZON  # drop HORIZON bars between train/test so labels don't peek
test = df.iloc[split + purge:].copy()

print(f'\nTrain set: {split} bars (60%)')
print(f'Purge gap: {purge} bars (prevents label leakage)')
print(f'Test set: {len(test)} bars (40% minus purge)')
print(f'Base success rate: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

safe_trades = test['outcome'] != 0

SESSION_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                    'is_london', 'is_ny', 'is_overlap']

import pyarrow.parquet as pq
TSFRESH_FEATURES = pq.read_schema(TSFRESH_CACHE).names

# Use only features that actually exist in df (guards against cache mismatches)
all_features = [f for f in TSFRESH_FEATURES if f in df.columns] + SESSION_FEATURES
print(f'Using {len(all_features)} features ({len(all_features) - len(SESSION_FEATURES)} tsfresh + {len(SESSION_FEATURES)} session)')

# ============================================================================
# TRAIN RF MODEL
# ============================================================================
print('\n' + '='*80)
print('TRAINING RANDOM FOREST')
print('='*80)

train = df.iloc[:split].copy()
X_train = train[all_features].fillna(0)
y_train = train['outcome'].astype(int)  # 3-class: 0=no move, 1=long wins, 2=short wins
X_test = test[all_features].fillna(0)
y_test = test['outcome'].astype(int)

print(f'Training on {len(X_train)} samples with {len(all_features)} features')
print(f'Class distribution: 0={int((y_train==0).sum())} ({(y_train==0).mean()*100:.1f}%) '
      f'1={int((y_train==1).sum())} ({(y_train==1).mean()*100:.1f}%) '
      f'2={int((y_train==2).sum())} ({(y_train==2).mean()*100:.1f}%)')

rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=100,
    min_samples_split=200, max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# 3-class probabilities: classes are [0, 1, 2]
rf_proba = rf.predict_proba(X_test)
class_labels = list(rf.classes_)
p_no_move = rf_proba[:, class_labels.index(0)]
p_long = rf_proba[:, class_labels.index(1)]
p_short = rf_proba[:, class_labels.index(2)]
p_move = p_long + p_short
edge = np.abs(p_long - p_short)

# Backward compat: rf_probs = p_move for threshold analysis
rf_probs = p_move
rf_auc = roc_auc_score((y_test != 0).astype(int), p_move)
print(f'AUC-ROC (move vs no-move): {rf_auc:.4f}')
print(f'\nDirectional probabilities on test set:')
print(f'  p_long  mean={p_long.mean():.3f}  max={p_long.max():.3f}')
print(f'  p_short mean={p_short.mean():.3f}  max={p_short.max():.3f}')
print(f'  p_move  mean={p_move.mean():.3f}  max={p_move.max():.3f}')
print(f'  edge    mean={edge.mean():.3f}  max={edge.max():.3f}')

# Feature importances
feat_imp = sorted(zip(all_features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
print(f'\nTop 15 features by importance:')
for feat, imp in feat_imp[:15]:
    print(f'  {feat:<35} {imp:.4f}')

# Threshold analysis
print(f'\nFiltered trading results by probability threshold:')
print(f'{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count"}')
print('-'*60)
for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = test[rf_probs >= threshold]
    if len(filtered) > 0:
        success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
        frequency = len(filtered) / len(test) * 100
        print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d}')

print(f'\n*** RF >= {RF_THRESHOLD} RESULTS ***')
filtered_75 = test[rf_probs >= RF_THRESHOLD]
if len(filtered_75) > 0:
    success_75 = (filtered_75['outcome'] != 0).sum() / len(filtered_75) * 100
    print(f'Success Rate: {success_75:.1f}%')
    print(f'Trade Count: {len(filtered_75)}')
    print(f'Frequency: {len(filtered_75) / len(test) * 100:.2f}%')

# Save model
import pickle
os.makedirs('models', exist_ok=True)
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/feature_names.txt', 'w') as f:
    f.write('\n'.join(all_features))
print('Saved: models/random_forest.pkl, models/feature_names.txt')

test['rf_prob'] = rf_probs
test['p_long'] = p_long
test['p_short'] = p_short
test['p_move'] = p_move
test['edge'] = edge
print(f'\nBaseline: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

test.to_csv('data/processed/XAUUSD1_feature_analysis.csv')
print(f'\nSaved test results: data/processed/XAUUSD1_feature_analysis.csv')


# ============================================================================
# DIRECTIONAL STRATEGY - FORWARD TEST (Multiclass RF + Alignment Gate)
# ============================================================================
print('\n' + '='*80)
print(f'DIRECTIONAL STRATEGY - FORWARD TEST')
print('='*80)
print(f'Logic: 3-class RF -> p_long, p_short, p_move')
print(f'       p_move >= {RF_THRESHOLD}')
print(f'       LONG:  p_long > p_short  (no alignment gate)')
print(f'       SHORT: p_short > p_long  (no alignment gate)')

# Prepare test set with datetime and session info
test_hedge = test.reset_index()
test_hedge['hour'] = test_hedge['Datetime'].dt.hour

def get_session(hour):
    if 0 <= hour < 8:
        return 'ASIAN'
    elif 8 <= hour < 13:
        return 'LONDON'
    elif 13 <= hour < 17:
        return 'NY_OVERLAP'
    elif 17 <= hour < 22:
        return 'NY'
    else:
        return 'LATE'

test_hedge['session'] = test_hedge['hour'].apply(get_session)

# Add directional columns
test_hedge['p_long'] = p_long
test_hedge['p_short'] = p_short
test_hedge['p_move'] = p_move
test_hedge['edge'] = edge

print(f'\nTotal test bars: {len(test_hedge)}')

# Gate 1: p_move >= threshold
mask_pmove = test_hedge['p_move'] >= RF_THRESHOLD
print(f'Pass p_move >= {RF_THRESHOLD}: {mask_pmove.sum()} ({mask_pmove.sum()/len(test_hedge)*100:.2f}%)')

# Direction: RF model decides — side with higher probability wins
mask_long_dir = test_hedge['p_long'] > test_hedge['p_short']
mask_short_dir = test_hedge['p_short'] > test_hedge['p_long']

# Final entry masks — p_move gate + RF direction only
mask_long_entry = mask_pmove & mask_long_dir
mask_short_entry = mask_pmove & mask_short_dir

print(f'LONG entries (all gates): {mask_long_entry.sum()}')
print(f'SHORT entries (all gates): {mask_short_entry.sum()}')
print(f'Total directional entries: {mask_long_entry.sum() + mask_short_entry.sum()}')

# Simulate directional trades
directional_results = []

# LONG entries
for idx in test_hedge[mask_long_entry].index:
    row = test_hedge.loc[idx]
    entry = row['Close']
    
    if row['outcome'] == 1:
        result = 'WIN'
        pnl = TARGET
    elif row['outcome'] == 2:
        result = 'LOSS'
        pnl = -STOP
    else:
        result = 'LOSS'
        pnl = -STOP
    
    directional_results.append({
        'Datetime': row['Datetime'],
        'session': row['session'],
        'p_long': round(row['p_long'], 3),
        'p_short': round(row['p_short'], 3),
        'p_move': round(row['p_move'], 3),
        'edge': round(row['edge'], 3),
        'trade_dir': 'LONG',
        'entry': entry,
        'result': result,
        'pnl': pnl,
        'outcome': int(row['outcome']),
    })

# SHORT entries
for idx in test_hedge[mask_short_entry].index:
    row = test_hedge.loc[idx]
    entry = row['Close']
    
    if row['outcome'] == 2:
        result = 'WIN'
        pnl = TARGET
    elif row['outcome'] == 1:
        result = 'LOSS'
        pnl = -STOP
    else:
        result = 'LOSS'
        pnl = -STOP
    
    directional_results.append({
        'Datetime': row['Datetime'],
        'session': row['session'],
        'p_long': round(row['p_long'], 3),
        'p_short': round(row['p_short'], 3),
        'p_move': round(row['p_move'], 3),
        'edge': round(row['edge'], 3),
        'trade_dir': 'SHORT',
        'entry': entry,
        'result': result,
        'pnl': pnl,
        'outcome': int(row['outcome']),
    })

# Convert to DataFrame and sort by time
dir_df = pd.DataFrame(directional_results)
if len(dir_df) > 0:
    dir_df = dir_df.sort_values('Datetime').reset_index(drop=True)

if len(dir_df) > 0:
    print(f'\nTotal directional setups: {len(dir_df)}')
    
    wins = (dir_df['result'] == 'WIN').sum()
    losses = len(dir_df) - wins
    win_rate = wins / len(dir_df) * 100
    
    total_pnl_dollars = dir_df['pnl'].sum()
    avg_pnl = total_pnl_dollars / len(dir_df)
    
    gross_profit = dir_df[dir_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(dir_df[dir_df['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    breakeven_wr = STOP / (TARGET + STOP) * 100
    
    print(f'\n--- OVERALL PERFORMANCE ---')
    print(f'Total setups: {len(dir_df)}')
    print(f'Wins: {wins}')
    print(f'Losses: {losses}')
    print(f'Win Rate: {win_rate:.1f}% (breakeven: {breakeven_wr:.1f}%)')
    print(f'\nTotal P&L: ${total_pnl_dollars:,.2f}')
    print(f'Avg P&L: ${avg_pnl:.4f} per trade')
    print(f'Profit Factor: {pf:.2f}')
    print(f'RR: {TARGET}:{STOP} = {TARGET/STOP:.1f}:1')
    
    # By direction
    print(f'\n--- BY TRADE DIRECTION ---')
    for d in ['LONG', 'SHORT']:
        sub = dir_df[dir_df['trade_dir'] == d]
        if len(sub) > 0:
            d_wins = (sub['result'] == 'WIN').sum()
            d_wr = d_wins / len(sub) * 100
            d_pnl = sub['pnl'].sum()
            print(f'{d:<6} {len(sub):>6} trades  WR: {d_wr:>5.1f}%  P&L: ${d_pnl:>8,.2f}')
    
    # Session breakdown
    print(f'\n--- SESSION BREAKDOWN ---')
    for sess in ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']:
        sess_data = dir_df[dir_df['session'] == sess]
        if len(sess_data) > 0:
            sess_wins = (sess_data['result'] == 'WIN').sum()
            sess_wr = sess_wins / len(sess_data) * 100
            sess_pnl = sess_data['pnl'].sum()
            print(f'{sess:<12} {len(sess_data):>5} trades  WR: {sess_wr:>5.1f}%  P&L: ${sess_pnl:>8,.2f}')
    
    # Save results
    dir_df['Datetime'] = dir_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    dir_df['entry'] = dir_df['entry'].round(2)
    dir_df.to_csv('data/processed/HEDGE_strategy.csv', index=False)
    print(f'\nSaved: data/processed/HEDGE_strategy.csv')
    
    # Show sample trades
    print('\n--- SAMPLE DIRECTIONAL TRADES (first 20) ---')
    print(dir_df.head(20).to_string(index=False))
else:
    print('\nNo directional setups generated. Try lowering RF_THRESHOLD.')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)