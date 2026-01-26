import pandas as pd
import numpy as np
import pickle

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
    
    # Imbalance
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
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - \
                        ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
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
    
    # Combination features
    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']
    
    return df.dropna()

def label_outcomes(df, horizon=20, target=0.001, stop=0.0005):
    """Label outcomes for 2:1 R:R"""
    df = df.copy()
    outcomes = []
    
    for i in range(len(df) - horizon):
        if i % 50000 == 0:
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
        
        outcomes.append(1 if long_hit else 0)
    
    df = df.iloc[:len(outcomes)].copy()
    df['outcome'] = outcomes
    return df

print('='*80)
print('XAUUSD 2022 BACKTEST WITH TRAINED MODEL')
print('='*80)

# Load the trained model
print('\nLoading trained model...')
with open('models/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/feature_names.txt', 'r') as f:
    feature_names = f.read().strip().split('\n')
print(f'Model loaded with {len(feature_names)} features')

# Load 2022 data
print('\nLoading 2022 data...')
df = pd.read_csv('data/raw/XAUUSD2022.csv', sep='\t', 
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open','High','Low','Close']].copy()
print(f'Loaded {len(df)} bars from 2022')
print(f'Date range: {df.index[0]} to {df.index[-1]}')

# Engineer features
print('\nEngineering features...')
df = create_microstructure_features(df)

# Label outcomes
print('\nLabeling outcomes (20 bars, 0.1% target, 0.05% stop)...')
df = label_outcomes(df, 20, 0.001, 0.0005)

# Get predictions
print('\nGenerating predictions...')
X = df[feature_names].fillna(0)
df['rf_prob'] = rf_model.predict_proba(X)[:, 1]

# Add session
df = df.reset_index()
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

# Apply Rule 4
print('\nApplying Rule 4 conditions...')
rule4_cond = (
    (df['rf_prob'] >= 0.70) & 
    (df['up_count_3'] == 3) & 
    (df['big_body'] == 1) & 
    (df['close_position'] > 0.7) &
    (df['session'] == 'NY_OVERLAP')
)

trades = df[rule4_cond].copy()
trades['result'] = trades['outcome'].map({1: 'WIN', 0: 'LOSS'})
trades['direction'] = 'LONG'
trades['entry_price'] = trades['Close']
trades['target_price'] = (trades['Close'] * 1.001).round(2)
trades['stop_price'] = (trades['Close'] * 0.9995).round(2)

print('\n' + '='*80)
print('RULE 4 RESULTS ON 2022 DATA')
print('='*80)

if len(trades) > 0:
    wins = (trades['result'] == 'WIN').sum()
    losses = len(trades) - wins
    win_rate = wins / len(trades) * 100
    
    print(f'\nTotal trades: {len(trades)}')
    print(f'Wins: {wins}')
    print(f'Losses: {losses}')
    print(f'Win Rate: {win_rate:.2f}%')
    print(f'Expected Value: {(win_rate/100 * 2) - ((1 - win_rate/100) * 1):.2f}R per trade')
    
    # Monthly breakdown
    trades['month'] = trades['Datetime'].dt.to_period('M')
    print(f'\nMonthly breakdown:')
    print(f'{"Month":<12} {"Trades":<8} {"Wins":<8} {"Losses":<8} {"Win Rate"}')
    print('-'*60)
    for month in trades['month'].unique():
        month_trades = trades[trades['month'] == month]
        m_wins = (month_trades['result'] == 'WIN').sum()
        m_losses = len(month_trades) - m_wins
        m_wr = m_wins / len(month_trades) * 100 if len(month_trades) > 0 else 0
        print(f'{str(month):<12} {len(month_trades):<8} {m_wins:<8} {m_losses:<8} {m_wr:>6.1f}%')
    
    # Save results
    output = trades[['Datetime', 'session', 'direction', 'entry_price', 'target_price', 'stop_price', 'rf_prob', 'result']].copy()
    output['Datetime'] = output['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    output['rf_prob'] = output['rf_prob'].round(3)
    output.to_csv('data/processed/XAUUSD2022_RULE4_trades.csv', index=False)
    print(f'\nSaved: data/processed/XAUUSD2022_RULE4_trades.csv')
    
    # Show first 20 trades
    print('\n' + '='*80)
    print('FIRST 20 TRADES')
    print('='*80)
    print(output.head(20).to_string(index=False))
else:
    print('\nNo trades found matching Rule 4 conditions in 2022 data.')

print('\n' + '='*80)
print('BACKTEST COMPLETE')
print('='*80)
