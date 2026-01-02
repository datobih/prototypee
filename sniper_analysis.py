import pandas as pd
import numpy as np
import pickle

print('Loading data...')
df = pd.read_csv('data/processed/XAUUSD1_feature_analysis.csv')
raw = pd.read_csv('data/raw/XAUUSD1.csv')

# Load model
with open('models/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Add RF predictions
X = df[feature_names].values
X_scaled = scaler.transform(X)
df['rf_prob'] = rf_model.predict_proba(X_scaled)[:, 1]

# Add session
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour
def get_session(hour):
    if 0 <= hour < 8: return 'ASIAN'
    elif 8 <= hour < 13: return 'LONDON'
    elif 13 <= hour < 17: return 'NY_OVERLAP'
    elif 17 <= hour < 22: return 'NY'
    else: return 'LATE'
df['session'] = df['hour'].apply(get_session)

print(f'Total bars: {len(df)}')
print()

# For each signal bar, calculate how far price moves up vs down in next N bars
# This lets us see the actual R-multiple potential

def analyze_rr_potential(subset, label, lookforward=60):
    """Analyze R-multiple potential for a subset of signals"""
    results = []
    
    for idx in subset.index:
        if idx + lookforward >= len(df):
            continue
            
        entry = df.loc[idx, 'Close']
        atr = df.loc[idx, 'atr'] if 'atr' in df.columns else None
        
        # Use ATR-based stop or fixed pip stop
        if atr and atr > 0:
            stop_distance = atr * 1.5  # 1.5 ATR stop
        else:
            stop_distance = 2.0  # Default $2 stop for gold
        
        stop_price = entry - stop_distance
        
        # Look at next N bars
        future = df.loc[idx+1:idx+lookforward]
        
        # Find max favorable excursion (MFE) and max adverse excursion (MAE)
        max_high = future['High'].max() if 'High' in future.columns else future['Close'].max()
        min_low = future['Low'].min() if 'Low' in future.columns else future['Close'].min()
        
        mfe = max_high - entry  # Max profit before stop
        mae = entry - min_low   # Max drawdown before target
        
        # Calculate R-multiples
        r_profit = mfe / stop_distance if stop_distance > 0 else 0
        r_loss = mae / stop_distance if stop_distance > 0 else 0
        
        # Did we hit stop first or target first?
        hit_stop = min_low <= stop_price
        
        # What R could we achieve with trailing or targets?
        results.append({
            'mfe_r': r_profit,
            'mae_r': r_loss,
            'hit_stop': hit_stop,
            'net_r': r_profit - r_loss if not hit_stop else -1
        })
    
    if not results:
        print(f'{label}: No valid signals')
        return
        
    res_df = pd.DataFrame(results)
    
    avg_mfe = res_df['mfe_r'].mean()
    avg_mae = res_df['mae_r'].mean()
    pct_2r = (res_df['mfe_r'] >= 2).mean() * 100
    pct_3r = (res_df['mfe_r'] >= 3).mean() * 100
    stop_rate = res_df['hit_stop'].mean() * 100
    
    print(f'{label}:')
    print(f'  Signals: {len(res_df)}')
    print(f'  Avg MFE: {avg_mfe:.1f}R | Avg MAE: {avg_mae:.1f}R')
    print(f'  Hit 2R: {pct_2r:.0f}% | Hit 3R: {pct_3r:.0f}%')
    print(f'  Stop hit rate: {stop_rate:.0f}%')
    print()

# Analyze different filter combos
print('=== R-MULTIPLE POTENTIAL ANALYSIS (1.5 ATR stop) ===')
print()

# Baseline
analyze_rr_potential(df[df['rf_prob'] >= 0.70], 'RF >= 0.70 (baseline)')

# Best combos from before
analyze_rr_potential(
    df[(df['rf_prob'] >= 0.70) & (df['session'] == 'NY_OVERLAP')],
    'RF >= 0.70 + NY_OVERLAP'
)

analyze_rr_potential(
    df[(df['rf_prob'] >= 0.70) & (df['up_count_3'] == 3) & (df['vol_expansion'] == 1)],
    'RF >= 0.70 + up3 + vol_expand'
)

analyze_rr_potential(
    df[(df['rf_prob'] >= 0.70) & (df['up_count_3'] == 3) & (df['vol_expansion'] == 1) & (df['session'] == 'NY_OVERLAP')],
    'RF >= 0.70 + up3 + vol_expand + NY_OVERLAP'
)

# Try tighter filters for sniper entries
analyze_rr_potential(
    df[(df['rf_prob'] >= 0.75) & (df['session'] == 'NY_OVERLAP')],
    'RF >= 0.75 + NY_OVERLAP'
)

analyze_rr_potential(
    df[(df['rf_prob'] >= 0.70) & (df['trend_align'] == 1) & (df['vol_expansion'] == 1) & (df['session'] == 'NY_OVERLAP')],
    'RF >= 0.70 + trend + vol + NY_OVERLAP'
)

# Look for setups with low MAE (tight stops possible)
analyze_rr_potential(
    df[(df['rf_prob'] >= 0.70) & (df['lower_reject'] == 1) & (df['session'] == 'NY_OVERLAP')],
    'RF >= 0.70 + lower_reject + NY_OVERLAP (tight SL potential)'
)
