"""
Full Dataset Analysis - No Train/Test Split
This script evaluates Rule 4 performance on ALL data (including training data).
NOTE: This is data leakage and will overestimate performance, but is insightful.
"""

import pandas as pd
import numpy as np
import pickle
import os

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

def label_outcomes(df, horizon=20, target=0.001, stop=0.0005):
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
print('FULL DATASET ANALYSIS (NO SPLIT - DATA LEAKAGE WARNING)')
print('='*80)

# Load the FULL raw data
print('\nLoading FULL raw data from XAUUSD1.csv...')
df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', names=['Datetime','Open','High','Low','Close','Volume'], parse_dates=['Datetime'])
df.set_index('Datetime', inplace=True)
print(f'Loaded {len(df)} bars (1-minute timeframe)')

print('\nEngineering features...')
df = create_microstructure_features(df)

print('\nLabeling outcomes (20 bars=20mins, 0.1% target, 0.05% stop)...')
df = label_outcomes(df, 20, 0.001, 0.0005)

# Create combination features
print('\nCreating combination features...')
df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']

df = df.reset_index()
print(f'\nTotal bars after processing: {len(df)}')

# Load the trained model
print('\nLoading Random Forest model...')
with open('models/random_forest.pkl', 'rb') as f:
    rf = pickle.load(f)

# Load feature names
with open('models/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]
print(f'Features: {len(feature_names)}')

# Get RF probabilities for ALL data
print('\nCalculating RF probabilities for ALL bars...')
X = df[feature_names].fillna(0)
rf_probs = rf.predict_proba(X)[:, 1]
df['rf_prob'] = rf_probs

# Add session
df['hour'] = df['Datetime'].dt.hour

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

df['session'] = df['hour'].apply(get_session)

# Add continuation pattern
df['all_up_3'] = (df['up_count_3'] == 3).astype(int)

# Filter for movement trades only (outcome != 0)
movement = df[df['outcome'] != 0].copy()
all_bars = df.copy()

print(f'\nTotal bars: {len(all_bars)}')
print(f'Movement trades (outcome != 0): {len(movement)}')
print(f'Base success rate: {len(movement) / len(all_bars) * 100:.1f}%')

# ============================================================================
# RF PROBABILITY THRESHOLDS ON ALL DATA
# ============================================================================
print('\n' + '='*80)
print('RF PROBABILITY THRESHOLDS (ALL DATA)')
print('='*80)

print(f'\n{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count":<12} {"Wins":<12} {"Losses"}')
print('-'*80)
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = all_bars[all_bars['rf_prob'] >= threshold]
    if len(filtered) > 0:
        wins = (filtered['outcome'] == 1).sum()
        losses = (filtered['outcome'] == 2).sum()
        no_move = (filtered['outcome'] == 0).sum()
        success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
        frequency = len(filtered) / len(all_bars) * 100
        print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d} {wins:>11d} {losses:>11d}')

# ============================================================================
# RULE 4 ON ALL DATA
# ============================================================================
print('\n' + '='*80)
print('RULE 4: RF >= 0.70 + Bullish Pattern (NY_OVERLAP ONLY) - ALL DATA')
print('='*80)
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
    
    # Walk-forward validation by time periods
    if len(rule4_trades) >= 5:
        rule4_trades = rule4_trades.sort_values('Datetime')
        rule4_trades['period'] = pd.cut(range(len(rule4_trades)), bins=5, labels=['P1','P2','P3','P4','P5'])
        print(f'\nWalk-forward validation (chronological):')
        passes = 0
        for p in ['P1','P2','P3','P4','P5']:
            pdata = rule4_trades[rule4_trades['period'] == p]
            if len(pdata) > 0:
                pct = (pdata['result'] == 'WIN').sum() / len(pdata) * 100
                status = 'PASS' if pct > 52 else 'FAIL'
                if pct > 52: passes += 1
                first_date = pdata['Datetime'].iloc[0].strftime('%Y-%m-%d')
                last_date = pdata['Datetime'].iloc[-1].strftime('%Y-%m-%d')
                print(f'  {p}: {pct:>5.1f}% (n={len(pdata)}) [{status}] | {first_date} to {last_date}')
        print(f'  Result: {passes}/5 periods pass')
    
    # Monthly breakdown
    rule4_trades['month'] = rule4_trades['Datetime'].dt.to_period('M')
    print(f'\nMonthly breakdown:')
    print(f'{"Month":<12} {"Win Rate":<12} {"Trades":<10} {"Wins":<8} {"Losses"}')
    print('-'*50)
    for month in rule4_trades['month'].unique():
        month_trades = rule4_trades[rule4_trades['month'] == month]
        m_wins = (month_trades['result'] == 'WIN').sum()
        m_losses = (month_trades['result'] == 'LOSS').sum()
        m_wr = m_wins / len(month_trades) * 100
        print(f'{str(month):<12} {m_wr:>10.1f}% {len(month_trades):>9d} {m_wins:>7d} {m_losses:>7d}')
    
    # Save all Rule 4 trades
    rule4_output = rule4_trades[['Datetime', 'session', 'direction', 'entry_price', 'target_price', 'stop_price', 'rf_prob', 'result']].copy()
    rule4_output['Datetime'] = rule4_output['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    rule4_output['rf_prob'] = rule4_output['rf_prob'].round(3)
    rule4_output['entry_price'] = rule4_output['entry_price'].round(2)
    rule4_output.to_csv('data/processed/RULE4_ALL_DATA_trades.csv', index=False)
    print(f'\nSaved: data/processed/RULE4_ALL_DATA_trades.csv')
    
    # Show all trades
    print('\n' + '-'*80)
    print('RULE 4 ALL TRADES (FULL DATASET)')
    print('-'*80)
    print(rule4_output.to_string(index=False))
else:
    print('\nNo trades found matching Rule 4 conditions.')

# ============================================================================
# RULE 3 ON ALL DATA (Multiple Sessions)
# ============================================================================
print('\n' + '='*80)
print('RULE 3: RF >= 0.70 + Bullish Pattern (Exclude LONDON) - ALL DATA')
print('='*80)

rule3_cond = (
    (movement['rf_prob'] >= 0.70) & 
    (movement['all_up_3'] == 1) & 
    (movement['big_body'] == 1) & 
    (movement['close_position'] > 0.7) &
    (movement['session'].isin(['ASIAN', 'NY_OVERLAP', 'NY']))
)

rule3_trades = movement[rule3_cond].copy()
rule3_trades['result'] = rule3_trades['outcome'].map({1: 'WIN', 2: 'LOSS'})

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
    
    # Session breakdown
    print(f'\nSession breakdown:')
    for sess in ['ASIAN', 'NY_OVERLAP', 'NY']:
        sess_trades = rule3_trades[rule3_trades['session'] == sess]
        if len(sess_trades) > 0:
            sess_wins = (sess_trades['result'] == 'WIN').sum()
            sess_wr = sess_wins / len(sess_trades) * 100
            print(f'  {sess:<12} {sess_wr:>5.1f}% ({sess_wins}W/{len(sess_trades)-sess_wins}L, n={len(sess_trades)})')

# ============================================================================
# PURE RF FILTER (No Pattern) ON ALL DATA
# ============================================================================
print('\n' + '='*80)
print('PURE RF FILTER (NO PATTERN REQUIREMENTS) - ALL DATA')
print('='*80)

for threshold in [0.70, 0.75, 0.80, 0.85, 0.90]:
    filtered = movement[movement['rf_prob'] >= threshold]
    if len(filtered) > 0:
        wins = (filtered['outcome'] == 1).sum()
        losses = (filtered['outcome'] == 2).sum()
        win_rate = wins / len(filtered) * 100
        print(f'\nRF >= {threshold}:')
        print(f'  Trades: {len(filtered)} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%')

# ============================================================================
# ADDITIONAL ANALYSIS: Look at failed trades
# ============================================================================
if len(rule4_trades) > 0:
    print('\n' + '='*80)
    print('ANALYSIS OF LOSING TRADES')
    print('='*80)
    
    losers = rule4_trades[rule4_trades['result'] == 'LOSS']
    winners = rule4_trades[rule4_trades['result'] == 'WIN']
    
    if len(losers) > 0:
        print(f'\nLosing trades: {len(losers)}')
        print(f'\nAverage RF probability:')
        print(f'  Winners: {winners["rf_prob"].mean():.3f}')
        print(f'  Losers:  {losers["rf_prob"].mean():.3f}')
        
        print(f'\nAverage close_position:')
        print(f'  Winners: {winners["close_position"].mean():.3f}')
        print(f'  Losers:  {losers["close_position"].mean():.3f}')
        
        print(f'\nAverage vol_ratio:')
        print(f'  Winners: {winners["vol_ratio"].mean():.3f}')
        print(f'  Losers:  {losers["vol_ratio"].mean():.3f}')
        
        # Check if losers cluster on specific dates
        print(f'\nLosing trade dates:')
        for _, row in losers.iterrows():
            print(f'  {row["Datetime"]} | RF: {row["rf_prob"]:.3f} | Entry: {row["Close"]:.2f}')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print('\n⚠️  WARNING: These results include data leakage (model trained on same data).')
print('    Expect real-world performance to be LOWER than shown here.')
