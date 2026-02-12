import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import sys, os

def create_microstructure_features(df):
    df = df.copy()
    
    # Price structure
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)
    
    # Order flow
    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)
    
    # Imbalance (strong directional pressure)
    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()
    
    # Momentum consistency - is_up = bullish candle (close > open)
    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))
    
    # Volatility
    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)
    
    # Trend structure
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']
    
    # Support/Resistance
    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)
    
    # Rejection patterns
    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
    
    # Size patterns
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)
    
    return df.dropna()

def label_outcomes(df, horizon=15, target=0.0015, stop=0.0005):
    df = df.copy()
    outcomes = []
    
    for i in range(len(df) - horizon):
        if i % 20000 == 0:
            print(f'  Labeling {i}/{len(df)-horizon}...')
        
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]
        
        # Check LONG
        long_hit = False
        for h, l in zip(future['High'], future['Low']):
            if h >= entry * (1 + target):
                long_hit = True
                break
            if l <= entry * (1 - stop):
                break
        
        # Check SHORT
        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry * (1 - target):
                    short_hit = True
                    break
                if h >= entry * (1 + stop):
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
df = pd.read_csv('data/raw/US30.csv', sep='\t', 
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open','High','Low','Close']].copy()  # Keep only OHLC
print(f'Loaded {len(df)} bars (1-minute timeframe)')

print('\nEngineering features...')
df_base = create_microstructure_features(df)

# ============================================================================
# TARGET AND STOP LOSS OPTIMIZATION
# ============================================================================
print('\n' + '='*80)
print('TARGET AND STOP LOSS OPTIMIZATION')
print('='*80)

# Define horizons to test (in bars/minutes)
horizons = [15, 20, 30, 40, 60]

# Define target:stop ratios and values to test
test_configs = [
    # Risk:Reward 2:1 (standard)
    {'target': 0.001, 'stop': 0.0005, 'name': '0.10%:0.05% (2:1)'},
    {'target': 0.0015, 'stop': 0.00075, 'name': '0.15%:0.075% (2:1)'},
    {'target': 0.002, 'stop': 0.001, 'name': '0.20%:0.10% (2:1)'},
    {'target': 0.003, 'stop': 0.0015, 'name': '0.30%:0.15% (2:1)'},
    
    # Risk:Reward 3:1
    {'target': 0.0015, 'stop': 0.0005, 'name': '0.15%:0.05% (3:1)'},
    {'target': 0.003, 'stop': 0.001, 'name': '0.30%:0.10% (3:1)'},
    
    # Risk:Reward 1.5:1
    {'target': 0.00075, 'stop': 0.0005, 'name': '0.075%:0.05% (1.5:1)'},
    {'target': 0.0015, 'stop': 0.001, 'name': '0.15%:0.10% (1.5:1)'},
    
    # Tight stops
    {'target': 0.001, 'stop': 0.00025, 'name': '0.10%:0.025% (4:1)'},
    {'target': 0.0005, 'stop': 0.00025, 'name': '0.05%:0.025% (2:1)'},
]

optimization_results = []

for horizon in horizons:
    print(f'\n--- Testing Horizon: {horizon} bars ({horizon} mins) ---')
    
    for config in test_configs:
        target = config['target']
        stop = config['stop']
        name = config['name']
        
        print(f'  Testing {name}...', end='', flush=True)
        
        # Label outcomes for this configuration
        df_test = label_outcomes(df_base.copy(), horizon, target, stop)
        
        # Calculate metrics
        total_bars = len(df_test)
        movement_trades = (df_test['outcome'] != 0).sum()
        success_rate = (movement_trades / total_bars * 100) if total_bars > 0 else 0
        
        # Calculate R:R ratio
        rr_ratio = target / stop if stop > 0 else 0
        
        # Calculate Expected Value per trade
        # EV = (WinRate * R:R) - (LossRate * 1)
        win_rate_pct = success_rate / 100
        loss_rate_pct = 1 - win_rate_pct
        expected_value = (win_rate_pct * rr_ratio) - (loss_rate_pct * 1)
        
        # Calculate trade frequency
        trade_frequency = (movement_trades / total_bars * 100) if total_bars > 0 else 0
        
        # Efficiency score = EV * sqrt(frequency) to balance profitability with opportunity
        efficiency_score = expected_value * np.sqrt(trade_frequency)
        
        optimization_results.append({
            'horizon': horizon,
            'target': target,
            'stop': stop,
            'config_name': name,
            'rr_ratio': rr_ratio,
            'success_rate': success_rate,
            'movement_trades': movement_trades,
            'trade_frequency': trade_frequency,
            'expected_value': expected_value,
            'efficiency_score': efficiency_score
        })
        
        print(f' Success: {success_rate:.1f}%, EV: {expected_value:.3f}R, Efficiency: {efficiency_score:.3f}')

# Convert to DataFrame and sort by efficiency
results_df = pd.DataFrame(optimization_results)
results_df = results_df.sort_values('efficiency_score', ascending=False)

print('\n' + '='*80)
print('OPTIMIZATION RESULTS - RANKED BY EFFICIENCY')
print('='*80)
print(f'\n{"Rank":<5} {"Horizon":<8} {"Target:Stop":<20} {"R:R":<6} {"Success%":<10} {"Trades":<8} {"Freq%":<8} {"EV (R)":<8} {"Efficiency"}')
print('-'*100)

for idx, row in results_df.iterrows():
    rank = results_df.index.get_loc(idx) + 1
    print(f'{rank:<5} {row["horizon"]:<8} {row["config_name"]:<20} {row["rr_ratio"]:<6.1f} {row["success_rate"]:<10.1f} {row["movement_trades"]:<8} {row["trade_frequency"]:<8.2f} {row["expected_value"]:<8.3f} {row["efficiency_score"]:<8.3f}')

# Show top 5 configurations
print('\n' + '='*80)
print('TOP 5 MOST EFFICIENT CONFIGURATIONS')
print('='*80)

for i, (idx, row) in enumerate(results_df.head(5).iterrows(), 1):
    print(f'\n#{i}. {row["config_name"]} with {row["horizon"]} bar horizon')
    print(f'    Target: {row["target"]*100:.3f}%, Stop: {row["stop"]*100:.3f}%')
    print(f'    Risk:Reward Ratio: {row["rr_ratio"]:.2f}:1')
    print(f'    Success Rate: {row["success_rate"]:.2f}%')
    print(f'    Trade Frequency: {row["trade_frequency"]:.2f}% ({row["movement_trades"]} trades)')
    print(f'    Expected Value: {row["expected_value"]:.3f}R per trade')
    print(f'    Efficiency Score: {row["efficiency_score"]:.3f}')

# Use the best configuration for further analysis
best_config = results_df.iloc[0]
print(f'\n' + '='*80)
print(f'USING BEST CONFIGURATION FOR REMAINING ANALYSIS')
print('='*80)
print(f'Configuration: {best_config["config_name"]}')
print(f'Horizon: {best_config["horizon"]} bars')
print(f'Target: {best_config["target"]*100:.3f}%, Stop: {best_config["stop"]*100:.3f}%')

df = label_outcomes(df_base, int(best_config['horizon']), best_config['target'], best_config['stop'])

# Create combination features on full dataset
print('\nCreating combination features...')
df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']

# Split 60:40 for more test data
split = int(len(df) * 0.6)
test = df.iloc[split:].copy()

print(f'\nTrain set: {split} bars (60%)')
print(f'Test set: {len(test)} bars (40%)')
print(f'Base success rate: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

# Analyze correlations
print('\n' + '='*80)
print('FEATURE CORRELATION WITH SUCCESSFUL TRADES')
print('='*80)

safe_trades = test['outcome'] != 0

feature_cols = [
    'flow_3', 'flow_5', 'flow_10', 'flow_momentum',
    'imbalance_3', 'imbalance_5',
    'consistency_3', 'consistency_5',
    'vol_ratio', 'vol_expansion', 'vol_contraction',
    'trend_align', 'dist_ema8',
    'at_high', 'at_low',
    'upper_reject', 'lower_reject',
    'big_body', 'small_body',
    'body_pct', 'close_position'
]

correlations = []
for feat in feature_cols:
    if feat in test.columns:
        corr = test[feat].corr(safe_trades.astype(int))
        safe_mean = test[safe_trades][feat].mean()
        noise_mean = test[~safe_trades][feat].mean()
        diff = abs(safe_mean - noise_mean)
        correlations.append((feat, corr, safe_mean, noise_mean, diff))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f'\n{"Feature":<20} {"Correlation":<12} {"Safe Mean":<12} {"Noise Mean":<12} {"Difference"}')
print('-'*80)
for feat, corr, safe_m, noise_m, diff in correlations[:15]:
    print(f'{feat:<20} {corr:>11.4f} {safe_m:>11.4f} {noise_m:>11.4f} {diff:>11.4f}')

# Create combination features
print('\n' + '='*80)
print('ANALYZING COMBINATION FEATURE CORRELATIONS')
print('='*80)

combo_features = [
    'combo_flow_trend', 'combo_vol_imbalance', 'combo_consistency_position',
    'combo_body_reject', 'combo_trend_volatility', 'combo_imbalance_momentum',
    'combo_position_consistency', 'combo_vol_flow'
]

print(f'\n{"Combination Feature":<35} {"Correlation":<12} {"Safe Mean":<12} {"Noise Mean":<12} {"Difference"}')
print('-'*100)
combo_correlations = []
for feat in combo_features:
    corr = test[feat].corr(safe_trades.astype(int))
    safe_mean = test[safe_trades][feat].mean()
    noise_mean = test[~safe_trades][feat].mean()
    diff = abs(safe_mean - noise_mean)
    combo_correlations.append((feat, corr, safe_mean, noise_mean, diff))
    print(f'{feat:<35} {corr:>11.4f} {safe_mean:>11.4f} {noise_mean:>11.4f} {diff:>11.4f}')

combo_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# ============================================================================
# MULTI-FEATURE MODELS: Find optimal weighted combinations
# ============================================================================
print('\n' + '='*80)
print('TRAINING MODELS TO FIND OPTIMAL FEATURE COMBINATIONS')
print('='*80)

# Prepare train/test data
train = df.iloc[:split].copy()
train_safe = train['outcome'] != 0

# Select all available features
all_features = feature_cols + combo_features
X_train = train[all_features].fillna(0)
y_train = train_safe.astype(int)
X_test = test[all_features].fillna(0)
y_test = safe_trades.astype(int)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'\nTraining on {len(X_train)} samples with {len(all_features)} features')
print(f'Testing on {len(X_test)} samples')

# 1. Logistic Regression (linear combination of all features)
print('\n--- LOGISTIC REGRESSION ---')
print('Finding optimal linear combination of all features...')
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print(f'AUC-ROC: {lr_auc:.4f}')

# Show most important features by coefficient
feature_importance = list(zip(all_features, lr.coef_[0]))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
print(f'\nTop 10 features by coefficient:')
for feat, coef in feature_importance[:10]:
    print(f'  {feat:<35} {coef:>8.4f}')

# Test different probability thresholds
print(f'\nFiltered trading results by probability threshold:')
print(f'{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count"}')
print('-'*60)
for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = test[lr_probs >= threshold]
    if len(filtered) > 0:
        success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
        frequency = len(filtered) / len(test) * 100
        print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d}')

# 2. Random Forest (non-linear combinations)
print('\n--- RANDOM FOREST ---')
print('Finding optimal non-linear feature combinations...')
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print(f'AUC-ROC: {rf_auc:.4f}')

# Show most important features
feature_importance_rf = list(zip(all_features, rf.feature_importances_))
feature_importance_rf.sort(key=lambda x: x[1], reverse=True)
print(f'\nTop 10 features by importance:')
for feat, importance in feature_importance_rf[:10]:
    print(f'  {feat:<35} {importance:>8.4f}')

# Test different probability thresholds
print(f'\nFiltered trading results by probability threshold:')
print(f'{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count"}')
print('-'*60)
for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = test[rf_probs >= threshold]
    if len(filtered) > 0:
        success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
        frequency = len(filtered) / len(test) * 100
        print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d}')

# Highlight RF >= 0.75 specifically
print(f'\n*** RF >= 0.75 RESULTS ***')
filtered_75 = test[rf_probs >= 0.75]
if len(filtered_75) > 0:
    success_75 = (filtered_75['outcome'] != 0).sum() / len(filtered_75) * 100
    print(f'Success Rate: {success_75:.1f}%')
    print(f'Trade Count: {len(filtered_75)}')
    print(f'Frequency: {len(filtered_75) / len(test) * 100:.2f}%')

# Save probabilities to test set for further analysis
test['lr_prob'] = lr_probs
test['rf_prob'] = rf_probs

# Find best combinations
print('\n' + '='*80)
print('TESTING FEATURE COMBINATIONS (FILTERED SETUPS)')
print('='*80)

results = []

# Test combinations
combo1 = test[(test['imbalance_3'].abs() >= 2) & (test['consistency_5'] >= 4)]
if len(combo1) > 0:
    results.append(('Strong imbalance + high consistency', 
                   (combo1['outcome'] != 0).sum() / len(combo1) * 100,
                   len(combo1) / len(test) * 100, len(combo1)))

combo2 = test[(test['flow_momentum'] > 0.0005) & (test['trend_align'] == 1) & (test['vol_expansion'] == 1)]
if len(combo2) > 0:
    results.append(('Flow accel + trend align + vol expansion', 
                   (combo2['outcome'] != 0).sum() / len(combo2) * 100,
                   len(combo2) / len(test) * 100, len(combo2)))

combo3 = test[(test['big_body'] == 1) & (test['consistency_3'] == 3) & (test['close_position'] > 0.7)]
if len(combo3) > 0:
    results.append(('Big body + 3 up + strong close', 
                   (combo3['outcome'] != 0).sum() / len(combo3) * 100,
                   len(combo3) / len(test) * 100, len(combo3)))

# Test combinations using the new combo features
combo4 = test[test['combo_vol_imbalance'] > test['combo_vol_imbalance'].quantile(0.95)]
if len(combo4) > 0:
    results.append(('Top 5% vol*imbalance', 
                   (combo4['outcome'] != 0).sum() / len(combo4) * 100,
                   len(combo4) / len(test) * 100, len(combo4)))

combo5 = test[test['combo_flow_trend'] > test['combo_flow_trend'].quantile(0.95)]
if len(combo5) > 0:
    results.append(('Top 5% flow*trend', 
                   (combo5['outcome'] != 0).sum() / len(combo5) * 100,
                   len(combo5) / len(test) * 100, len(combo5)))

combo6 = test[test['combo_imbalance_momentum'] > test['combo_imbalance_momentum'].quantile(0.90)]
if len(combo6) > 0:
    results.append(('Top 10% imbalance*momentum', 
                   (combo6['outcome'] != 0).sum() / len(combo6) * 100,
                   len(combo6) / len(test) * 100, len(combo6)))

# Add model-based filters
combo7 = test[test['lr_prob'] >= 0.70]
if len(combo7) > 0:
    results.append(('LogReg prob >= 0.70', 
                   (combo7['outcome'] != 0).sum() / len(combo7) * 100,
                   len(combo7) / len(test) * 100, len(combo7)))

combo8 = test[test['rf_prob'] >= 0.70]
if len(combo8) > 0:
    results.append(('RandomForest prob >= 0.70', 
                   (combo8['outcome'] != 0).sum() / len(combo8) * 100,
                   len(combo8) / len(test) * 100, len(combo8)))

combo9 = test[(test['lr_prob'] >= 0.75) & (test['rf_prob'] >= 0.75)]
if len(combo9) > 0:
    results.append(('Both models >= 0.75', 
                   (combo9['outcome'] != 0).sum() / len(combo9) * 100,
                   len(combo9) / len(test) * 100, len(combo9)))

results.sort(key=lambda x: x[1], reverse=True)

print(f'\n{"Setup":<45} {"Success %":<12} {"Frequency %":<12} {"Count"}')
print('-'*80)
for setup, success, freq, count in results:
    if count >= 10:
        print(f'{setup:<45} {success:>11.1f} {freq:>11.2f} {count:>11d}')

print(f'\nBaseline: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

# Save models and scaler for live trading
print('\n' + '='*80)
print('SAVING MODELS FOR LIVE TRADING')
print('='*80)

import pickle
os.makedirs('models', exist_ok=True)

with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
print('Saved: models/logistic_regression.pkl')

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print('Saved: models/random_forest.pkl')

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Saved: models/scaler.pkl')

# Save feature names for reference
with open('models/feature_names.txt', 'w') as f:
    f.write('\n'.join(all_features))
print('Saved: models/feature_names.txt')

# Print manual trading criteria
print('\n' + '='*80)
print('MANUAL TRADING CRITERIA (NO MODEL NEEDED)')
print('='*80)

# Analyze successful trades to find typical feature ranges
high_prob_trades = test[test['rf_prob'] >= 0.85]
successful = high_prob_trades[high_prob_trades['outcome'] != 0]

if len(successful) > 0:
    print(f'\n85%+ Win Rate Setups ({len(high_prob_trades)} total, {len(successful)} successful):')
    print(f'\nFeature ranges for high-probability trades:')
    print('-'*80)
    
    key_features = ['vol_ratio', 'imbalance_3', 'consistency_5', 'big_body', 
                    'vol_expansion', 'flow_momentum', 'trend_align', 'close_position']
    
    for feat in key_features:
        if feat in successful.columns:
            min_val = successful[feat].quantile(0.25)
            max_val = successful[feat].quantile(0.75)
            median = successful[feat].median()
            print(f'{feat:<25} Median: {median:>8.4f}  Range: {min_val:>8.4f} to {max_val:>8.4f}')

test.to_csv('data/processed/XAUUSD1_feature_analysis.csv')
print(f'\nSaved test results: data/processed/XAUUSD1_feature_analysis.csv')

# ============================================================================
# BULLISH CONTINUATION TRADING RULES
# ============================================================================
print('\n' + '='*80)
print('BULLISH CONTINUATION TRADING RULES')
print('='*80)

# Use the test set which already has rf_prob calculated
test_rules = test.reset_index()

# Add session based on hour
test_rules['hour'] = test_rules['Datetime'].dt.hour

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

test_rules['session'] = test_rules['hour'].apply(get_session)

# Add continuation pattern: 3/3 bars up
test_rules['all_up_3'] = (test_rules['up_count_3'] == 3).astype(int)

# Filter for movement trades only (outcome != 0)
movement = test_rules[test_rules['outcome'] != 0].copy()

print(f'Test set bars: {len(test_rules)}')
print(f'Movement trades: {len(movement)}')

# ============================================================================
# RULE 1: RF 0.80-0.90 + 3/3 UP + BIG BODY + CLOSE > 0.7 + Exclude LONDON
# ============================================================================
print('\n' + '-'*80)
print('RULE 1: Bullish Continuation (Exclude LONDON)')
print('-'*80)
print('Conditions:')
print('  - RF Probability: 0.80 - 0.90')
print('  - Last 3 bars: ALL UP (up_count_3 == 3)')
print('  - Current bar: BIG BODY (> 1.5x 10-bar avg)')
print('  - Close position: > 0.7 (close near high)')
print('  - Sessions: ASIAN, NY_OVERLAP, NY (exclude LONDON, LATE)')

rule1_cond = (
    (movement['rf_prob'] >= 0.80) & 
    (movement['rf_prob'] < 0.90) & 
    (movement['all_up_3'] == 1) & 
    (movement['big_body'] == 1) & 
    (movement['close_position'] > 0.7) &
    (movement['session'].isin(['ASIAN', 'NY_OVERLAP', 'NY']))
)

rule1_trades = movement[rule1_cond].copy()
rule1_trades['result'] = rule1_trades['outcome'].map({1: 'WIN', 2: 'LOSS'})
rule1_trades['direction'] = 'LONG'
rule1_trades['entry_price'] = rule1_trades['Close']
rule1_trades['target_price'] = (rule1_trades['Close'] * 1.001).round(2)
rule1_trades['stop_price'] = (rule1_trades['Close'] * 0.9995).round(2)

if len(rule1_trades) > 0:
    wins = (rule1_trades['result'] == 'WIN').sum()
    losses = (rule1_trades['result'] == 'LOSS').sum()
    win_rate = wins / len(rule1_trades) * 100
    
    print(f'\nResults:')
    print(f'  Total trades: {len(rule1_trades)}')
    print(f'  Wins: {wins}')
    print(f'  Losses: {losses}')
    print(f'  Win Rate: {win_rate:.1f}%')
    print(f'  Expected Value: {(win_rate/100 * 2) - ((1 - win_rate/100) * 1):.2f}R per trade')
    
    # Walk-forward validation
    rule1_trades['period'] = pd.cut(range(len(rule1_trades)), bins=5, labels=['P1','P2','P3','P4','P5'])
    print(f'\nWalk-forward validation:')
    passes = 0
    for p in ['P1','P2','P3','P4','P5']:
        pdata = rule1_trades[rule1_trades['period'] == p]
        if len(pdata) > 0:
            pct = (pdata['result'] == 'WIN').sum() / len(pdata) * 100
            status = 'PASS' if pct > 52 else 'FAIL'
            if pct > 52: passes += 1
            print(f'  {p}: {pct:.1f}% (n={len(pdata)}) [{status}]')
    print(f'  Result: {passes}/5 periods pass')
    
    # Save Rule 1 trades
    rule1_output = rule1_trades[['Datetime', 'session', 'direction', 'entry_price', 'target_price', 'stop_price', 'rf_prob', 'result']].copy()
    rule1_output['Datetime'] = rule1_output['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    rule1_output['rf_prob'] = rule1_output['rf_prob'].round(3)
    rule1_output['entry_price'] = rule1_output['entry_price'].round(2)
    rule1_output.to_csv('data/processed/RULE1_trades.csv', index=False)
    print(f'\nSaved: data/processed/RULE1_trades.csv')
else:
    print('\nNo trades found matching Rule 1 conditions.')

# ============================================================================
# RULE 2: Same as Rule 1 but NY_OVERLAP session only
# ============================================================================
print('\n' + '-'*80)
print('RULE 2: Bullish Continuation (NY_OVERLAP Only)')
print('-'*80)
print('Conditions:')
print('  - RF Probability: 0.80 - 0.90')
print('  - Last 3 bars: ALL UP (up_count_3 == 3)')
print('  - Current bar: BIG BODY (> 1.5x 10-bar avg)')
print('  - Close position: > 0.7 (close near high)')
print('  - Sessions: NY_OVERLAP ONLY (13:00-17:00 UTC)')

rule2_cond = (
    (movement['rf_prob'] >= 0.80) & 
    (movement['rf_prob'] < 0.90) & 
    (movement['all_up_3'] == 1) & 
    (movement['big_body'] == 1) & 
    (movement['close_position'] > 0.7) &
    (movement['session'] == 'NY_OVERLAP')
)

rule2_trades = movement[rule2_cond].copy()
rule2_trades['result'] = rule2_trades['outcome'].map({1: 'WIN', 2: 'LOSS'})
rule2_trades['direction'] = 'LONG'
rule2_trades['entry_price'] = rule2_trades['Close']
rule2_trades['target_price'] = (rule2_trades['Close'] * 1.001).round(2)
rule2_trades['stop_price'] = (rule2_trades['Close'] * 0.9995).round(2)

if len(rule2_trades) > 0:
    wins = (rule2_trades['result'] == 'WIN').sum()
    losses = (rule2_trades['result'] == 'LOSS').sum()
    win_rate = wins / len(rule2_trades) * 100
    
    print(f'\nResults:')
    print(f'  Total trades: {len(rule2_trades)}')
    print(f'  Wins: {wins}')
    print(f'  Losses: {losses}')
    print(f'  Win Rate: {win_rate:.1f}%')
    print(f'  Expected Value: {(win_rate/100 * 2) - ((1 - win_rate/100) * 1):.2f}R per trade')
    
    # Save Rule 2 trades
    rule2_output = rule2_trades[['Datetime', 'session', 'direction', 'entry_price', 'target_price', 'stop_price', 'rf_prob', 'result']].copy()
    rule2_output['Datetime'] = rule2_output['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    rule2_output['rf_prob'] = rule2_output['rf_prob'].round(3)
    rule2_output['entry_price'] = rule2_output['entry_price'].round(2)
    rule2_output.to_csv('data/processed/RULE2_trades.csv', index=False)
    print(f'\nSaved: data/processed/RULE2_trades.csv')
else:
    print('\nNo trades found matching Rule 2 conditions.')

# Print sample trades
if len(rule1_trades) > 0:
    print('\n' + '-'*80)
    print('RULE 1 SAMPLE TRADES (first 10)')
    print('-'*80)
    sample = rule1_output.head(10)
    print(sample.to_string(index=False))

if len(rule2_trades) > 0:
    print('\n' + '-'*80)
    print('RULE 2 SAMPLE TRADES (all)')
    print('-'*80)
    print(rule2_output.to_string(index=False))

# ============================================================================
# RULE 3: RF >= 0.70 + Bullish Pattern (Exclude LONDON)
# ============================================================================
print('\n' + '-'*80)
print('RULE 3: RF >= 0.70 + Bullish Pattern (Exclude LONDON)')
print('-'*80)
print('Conditions:')
print('  - RF Probability: >= 0.70')
print('  - Last 3 bars: ALL UP (up_count_3 == 3)')
print('  - Current bar: BIG BODY (> 1.5x 10-bar avg)')
print('  - Close position: > 0.7 (close near high)')
print('  - Sessions: ASIAN, NY_OVERLAP, NY (exclude LONDON, LATE)')

rule3_cond = (
    (movement['rf_prob'] >= 0.70) & 
    (movement['all_up_3'] == 1) & 
    (movement['big_body'] == 1) & 
    (movement['close_position'] > 0.7) &
    (movement['session'].isin(['ASIAN', 'NY_OVERLAP', 'NY']))
)

rule3_trades = movement[rule3_cond].copy()
rule3_trades['result'] = rule3_trades['outcome'].map({1: 'WIN', 2: 'LOSS'})
rule3_trades['direction'] = 'LONG'
rule3_trades['entry_price'] = rule3_trades['Close']
rule3_trades['target_price'] = (rule3_trades['Close'] * 1.001).round(2)
rule3_trades['stop_price'] = (rule3_trades['Close'] * 0.9995).round(2)

if len(rule3_trades) > 0:
    wins = (rule3_trades['result'] == 'WIN').sum()
    losses = (rule3_trades['result'] == 'LOSS').sum()
    win_rate = wins / len(rule3_trades) * 100
    
    print(f'\nResults:')
    print(f'  Total trades: {len(rule3_trades)}')
    print(f'  Wins: {wins}')
    print(f'  Losses: {losses}')
    print(f'  Win Rate: {win_rate:.1f}%')
    print(f'  Expected Value: {(win_rate/100 * 2) - ((1 - win_rate/100) * 1):.2f}R per trade')
    
    # Walk-forward validation
    rule3_trades['period'] = pd.cut(range(len(rule3_trades)), bins=5, labels=['P1','P2','P3','P4','P5'])
    print(f'\nWalk-forward validation:')
    passes = 0
    for p in ['P1','P2','P3','P4','P5']:
        pdata = rule3_trades[rule3_trades['period'] == p]
        if len(pdata) > 0:
            pct = (pdata['result'] == 'WIN').sum() / len(pdata) * 100
            status = 'PASS' if pct > 52 else 'FAIL'
            if pct > 52: passes += 1
            print(f'  {p}: {pct:.1f}% (n={len(pdata)}) [{status}]')
    print(f'  Result: {passes}/5 periods pass')
    
    # Session breakdown
    print(f'\nSession breakdown:')
    for sess in ['ASIAN', 'NY_OVERLAP', 'NY']:
        sess_trades = rule3_trades[rule3_trades['session'] == sess]
        if len(sess_trades) > 0:
            sess_wins = (sess_trades['result'] == 'WIN').sum()
            sess_wr = sess_wins / len(sess_trades) * 100
            print(f'  {sess:<12} {sess_wr:>5.1f}% ({sess_wins}W/{len(sess_trades)-sess_wins}L, n={len(sess_trades)})')
    
    # Save Rule 3 trades
    rule3_output = rule3_trades[['Datetime', 'session', 'direction', 'entry_price', 'target_price', 'stop_price', 'rf_prob', 'result']].copy()
    rule3_output['Datetime'] = rule3_output['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    rule3_output['rf_prob'] = rule3_output['rf_prob'].round(3)
    rule3_output['entry_price'] = rule3_output['entry_price'].round(2)
    rule3_output.to_csv('data/processed/RULE3_trades.csv', index=False)
    print(f'\nSaved: data/processed/RULE3_trades.csv')
    
    # Show sample trades
    print('\n' + '-'*80)
    print('RULE 3 SAMPLE TRADES (first 15)')
    print('-'*80)
    print(rule3_output.head(15).to_string(index=False))
else:
    print('\nNo trades found matching Rule 3 conditions.')

# ============================================================================
# RULE 4: RF >= 0.70 + Bullish Pattern + NY_OVERLAP ONLY
# ============================================================================
print('\n' + '-'*80)
print('RULE 4: RF >= 0.70 + Bullish Pattern (NY_OVERLAP ONLY)')
print('-'*80)
print('Conditions:')
print('  - RF Probability: >= 0.70')
print('  - Last 3 bars: ALL UP (up_count_3 == 3)')
print('  - Current bar: BIG BODY (> 1.5x 10-bar avg)')
print('  - Close position: > 0.7 (close near high)')
print('  - Sessions: NY_OVERLAP ONLY (13:00-17:00 UTC)')

rule4_cond = (
    (movement['rf_prob'] >= 0.70) & 
    (movement['all_up_3'] == 1) & 
    (movement['big_body'] == 1) & 
    (movement['close_position'] > 0.7) &
    (movement['session'] == 'NY_OVERLAP')
)

rule4_trades = movement[rule4_cond].copy()
rule4_trades['result'] = rule4_trades['outcome'].map({1: 'WIN', 2: 'LOSS'})
rule4_trades['direction'] = 'LONG'
rule4_trades['entry_price'] = rule4_trades['Close']
rule4_trades['target_price'] = (rule4_trades['Close'] * 1.001).round(2)
rule4_trades['stop_price'] = (rule4_trades['Close'] * 0.9995).round(2)

if len(rule4_trades) > 0:
    wins = (rule4_trades['result'] == 'WIN').sum()
    losses = (rule4_trades['result'] == 'LOSS').sum()
    win_rate = wins / len(rule4_trades) * 100
    
    print(f'\nResults:')
    print(f'  Total trades: {len(rule4_trades)}')
    print(f'  Wins: {wins}')
    print(f'  Losses: {losses}')
    print(f'  Win Rate: {win_rate:.1f}%')
    print(f'  Expected Value: {(win_rate/100 * 2) - ((1 - win_rate/100) * 1):.2f}R per trade')
    
    # Walk-forward validation
    if len(rule4_trades) >= 5:
        rule4_trades['period'] = pd.cut(range(len(rule4_trades)), bins=5, labels=['P1','P2','P3','P4','P5'])
        print(f'\nWalk-forward validation:')
        passes = 0
        for p in ['P1','P2','P3','P4','P5']:
            pdata = rule4_trades[rule4_trades['period'] == p]
            if len(pdata) > 0:
                pct = (pdata['result'] == 'WIN').sum() / len(pdata) * 100
                status = 'PASS' if pct > 52 else 'FAIL'
                if pct > 52: passes += 1
                print(f'  {p}: {pct:.1f}% (n={len(pdata)}) [{status}]')
        print(f'  Result: {passes}/5 periods pass')
    
    # Save Rule 4 trades
    rule4_output = rule4_trades[['Datetime', 'session', 'direction', 'entry_price', 'target_price', 'stop_price', 'rf_prob', 'result']].copy()
    rule4_output['Datetime'] = rule4_output['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    rule4_output['rf_prob'] = rule4_output['rf_prob'].round(3)
    rule4_output['entry_price'] = rule4_output['entry_price'].round(2)
    rule4_output.to_csv('data/processed/RULE4_trades.csv', index=False)
    print(f'\nSaved: data/processed/RULE4_trades.csv')
    
    # Show all trades
    print('\n' + '-'*80)
    print('RULE 4 ALL TRADES')
    print('-'*80)
    print(rule4_output.to_string(index=False))
else:
    print('\nNo trades found matching Rule 4 conditions.')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)