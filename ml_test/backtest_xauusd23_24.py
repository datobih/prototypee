"""
Backtest script: loads RF model from hedging_strategy_strict.py
and evaluates on XAUUSD23_24.csv (out-of-sample data).
"""
 
import pandas as pd
import numpy as np
import pickle
import os
 
# =============================================================================
# PARAMETERS (must match hedging_strategy_strict.py)
# =============================================================================
TARGET = 3      # Target profit in dollars
STOP = 1.5      # Stop loss in dollars
HORIZON = 30    # Number of bars to look ahead for outcome
RF_THRESHOLD = 0.75
 
MODEL_PATH = 'models/random_forest.pkl'
FEATURES_PATH = 'models/feature_names.txt'
DATA_PATH = 'data/raw/XAUUSD23_24.csv'
 
# =============================================================================
# FEATURE ENGINEERING (copied exactly from hedging_strategy_strict.py)
# =============================================================================
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
 
    # --- DEEPER FLOW FEATURES ---
    df['flow_15'] = df['directional_flow'].rolling(15).sum()
    df['flow_20'] = df['directional_flow'].rolling(20).sum()
    df['flow_accel'] = df['flow_3'] - df['flow_3'].shift(3)
    df['flow_accel_5'] = df['flow_5'] - df['flow_5'].shift(5)
    df['abs_flow_3'] = df['flow_3'].abs()
    df['abs_flow_5'] = df['flow_5'].abs()
    df['abs_flow_10'] = df['flow_10'].abs()
    df['flow_divergence'] = (df['flow_3'] * df['flow_10'] < 0).astype(int)
 
    # --- FLOW QUALITY ---
    df['consecutive_up'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumcount() + 1
    df['consecutive_up'] = df['consecutive_up'] * df['is_up']
    df['consecutive_down'] = (1 - df['is_up']).groupby(((1-df['is_up']) != (1-df['is_up']).shift()).cumsum()).cumcount() + 1
    df['consecutive_down'] = df['consecutive_down'] * (1 - df['is_up'])
    df['max_consecutive'] = df[['consecutive_up', 'consecutive_down']].max(axis=1)
    df['flow_efficiency'] = df['abs_body'] / (df['range'] + 1e-10)
    df['flow_eff_3'] = df['flow_efficiency'].rolling(3).mean()
    df['flow_eff_5'] = df['flow_efficiency'].rolling(5).mean()
 
    # --- VOLATILITY REGIME ---
    df['atr_roc'] = (df['atr_3'] - df['atr_3'].shift(3)) / (df['atr_3'].shift(3) + 1e-10)
    df['vol_breakout'] = df['range'] / (df['atr_20'] + 1e-10)
    df['range_min_5'] = df['range'].rolling(5).min()
    df['range_max_5'] = df['range'].rolling(5).max()
    df['range_squeeze'] = df['range_min_5'] / (df['range_max_5'] + 1e-10)
 
    # --- DISTANCE FEATURES ---
    df['abs_dist_ema8'] = df['dist_ema8'].abs()
    df['dist_ema21'] = (df['Close'] - df['ema_21']) / df['Close']
    df['abs_dist_ema21'] = df['dist_ema21'].abs()
    df['ema_spread'] = (df['ema_8'] - df['ema_21']) / df['Close']
    df['abs_ema_spread'] = df['ema_spread'].abs()
 
    # --- COMBO FEATURES ---
    df['combo_abs_flow_vol'] = df['abs_flow_5'] * df['vol_ratio']
    df['combo_eff_flow'] = df['flow_eff_3'] * df['abs_flow_3']
    df['combo_consecutive_body'] = df['max_consecutive'] * df['body_pct']
    df['combo_atr_roc_accel'] = df['atr_roc'] * df['flow_accel'].abs()
    df['combo_squeeze_flow'] = (1 - df['range_squeeze']) * df['abs_flow_3']
    df['combo_dist_flow'] = df['abs_dist_ema8'] * df['abs_flow_5']
 
    # =====================================================================
    # ROUND 2 FEATURES
    # =====================================================================
 
    df = df.copy()  # defragment before Round 2 features
 
    # --- A. EXTENDED ABSOLUTE FLOW ---
    df['abs_flow_15'] = df['flow_15'].abs()
    df['abs_flow_20'] = df['flow_20'].abs()
 
    flow10_mean = df['abs_flow_10'].rolling(50).mean()
    flow10_std = df['abs_flow_10'].rolling(50).std()
    df['flow_zscore'] = (df['abs_flow_10'] - flow10_mean) / (flow10_std + 1e-10)
 
    abs_per_bar = df['directional_flow'].abs()
    df['flow_net_ratio_5'] = df['abs_flow_5'] / (abs_per_bar.rolling(5).sum() + 1e-10)
    df['flow_net_ratio_10'] = df['abs_flow_10'] / (abs_per_bar.rolling(10).sum() + 1e-10)
 
    df['flow_max_bar_5'] = df['directional_flow'].abs().rolling(5).max()
    df['flow_concentration_5'] = df['flow_max_bar_5'] / (abs_per_bar.rolling(5).sum() + 1e-10)
 
    # --- B. LONGER EMA DISTANCE ---
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['ema_100'] = df['Close'].ewm(span=100).mean()
    df['dist_ema50'] = (df['Close'] - df['ema_50']) / df['Close']
    df['abs_dist_ema50'] = df['dist_ema50'].abs()
    df['dist_ema100'] = (df['Close'] - df['ema_100']) / df['Close']
    df['abs_dist_ema100'] = df['dist_ema100'].abs()
 
    df['ema_spread_21_50'] = (df['ema_21'] - df['ema_50']) / df['Close']
    df['abs_ema_spread_21_50'] = df['ema_spread_21_50'].abs()
    df['ema_spread_50_100'] = (df['ema_50'] - df['ema_100']) / df['Close']
    df['abs_ema_spread_50_100'] = df['ema_spread_50_100'].abs()
 
    roll_mean_20 = df['Close'].rolling(20).mean()
    roll_std_20 = df['Close'].rolling(20).std()
    df['price_zscore_20'] = (df['Close'] - roll_mean_20) / (roll_std_20 + 1e-10)
    df['abs_price_zscore_20'] = df['price_zscore_20'].abs()
 
    roll_mean_50 = df['Close'].rolling(50).mean()
    roll_std_50 = df['Close'].rolling(50).std()
    df['price_zscore_50'] = (df['Close'] - roll_mean_50) / (roll_std_50 + 1e-10)
    df['abs_price_zscore_50'] = df['price_zscore_50'].abs()
 
    df['boll_upper_20'] = roll_mean_20 + 2 * roll_std_20
    df['boll_lower_20'] = roll_mean_20 - 2 * roll_std_20
    df['boll_width_20'] = (df['boll_upper_20'] - df['boll_lower_20']) / df['Close']
    df['boll_pct_20'] = (df['Close'] - df['boll_lower_20']) / (df['boll_upper_20'] - df['boll_lower_20'] + 1e-10)
    df['boll_outside_20'] = ((df['Close'] > df['boll_upper_20']) | (df['Close'] < df['boll_lower_20'])).astype(int)
 
    # --- C. ATR-NORMALIZED FEATURES ---
    df['flow_per_atr_3'] = df['abs_flow_3'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['flow_per_atr_10'] = df['abs_flow_10'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['dist_ema8_per_atr'] = df['abs_dist_ema8'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['dist_ema21_per_atr'] = df['abs_dist_ema21'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['ema_spread_per_atr'] = df['abs_ema_spread'] / (df['atr_10'] / df['Close'] + 1e-10)
 
    df['atr_percentile'] = df['atr_10'].rolling(100).rank(pct=True)
 
    # --- D. TIME FEATURES ---
    hour = df.index.hour
    minute = df.index.minute
    minutes_in_day = hour * 60 + minute
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_of_week'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
 
    # --- E. HIGHER-ORDER INTERACTIONS ---
    df['combo_flow3_dist21'] = df['abs_flow_3'] * df['abs_dist_ema21']
    df['combo_flow10_dist8'] = df['abs_flow_10'] * df['abs_dist_ema8']
    df['combo_flow10_spread'] = df['abs_flow_10'] * df['abs_ema_spread']
    df['combo_flow_atr_pct'] = df['abs_flow_10'] * df['atr_percentile']
    df['combo_flow_zscore_dist'] = df['flow_zscore'] * df['abs_dist_ema8']
    df['combo_triple'] = df['abs_flow_5'] * df['abs_dist_ema8'] * df['vol_ratio']
    df['combo_net_ratio_flow'] = df['flow_net_ratio_10'] * df['abs_flow_10']
    df['combo_concentration_flow'] = df['flow_concentration_5'] * df['abs_flow_5']
 
    # =====================================================================
    # ROUND 3 FEATURES — volatility regime deep-dive
    # =====================================================================
 
    # --- F. ADVANCED VOLATILITY MEASURES ---
    log_hl = (np.log(df['High'] / df['Low']))**2
    log_co = (np.log(df['Close'] / df['Open']))**2
    gk_single = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df['gk_vol_10'] = gk_single.rolling(10).mean()
    df['gk_vol_20'] = gk_single.rolling(20).mean()
    df['gk_vol_50'] = gk_single.rolling(50).mean()
 
    df['parkinson_vol_10'] = (log_hl / (4 * np.log(2))).rolling(10).mean()
    df['parkinson_vol_20'] = (log_hl / (4 * np.log(2))).rolling(20).mean()
 
    roll_mean_50 = df['Close'].rolling(50).mean()
    roll_std_50 = df['Close'].rolling(50).std()
    df['boll_width_50'] = (4 * roll_std_50) / df['Close']
    df['boll_pct_50'] = (df['Close'] - (roll_mean_50 - 2*roll_std_50)) / (4*roll_std_50 + 1e-10)
 
    roll_mean_100 = df['Close'].rolling(100).mean()
    roll_std_100 = df['Close'].rolling(100).std()
    df['boll_width_100'] = (4 * roll_std_100) / df['Close']
 
    df['keltner_width_20'] = (2 * 1.5 * df['atr_20']) / df['Close']
    df['keltner_width_10'] = (2 * 1.5 * df['atr_10']) / df['Close']
 
    df['boll_inside_keltner'] = (df['boll_width_20'] < df['keltner_width_20']).astype(int)
    df['squeeze_intensity'] = df['keltner_width_20'] / (df['boll_width_20'] + 1e-10)
 
    # --- G. VOLATILITY OF VOLATILITY ---
    df['vol_of_vol_20'] = df['boll_width_20'].rolling(20).std()
    df['vol_of_vol_50'] = df['boll_width_20'].rolling(50).std()
 
    df['boll_width_roc'] = (df['boll_width_20'] - df['boll_width_20'].shift(5)) / (df['boll_width_20'].shift(5) + 1e-10)
    df['gk_vol_roc'] = (df['gk_vol_10'] - df['gk_vol_10'].shift(10)) / (df['gk_vol_10'].shift(10) + 1e-10)
 
    df['boll_width_pct_50'] = df['boll_width_20'].rolling(50).rank(pct=True)
    df['boll_width_pct_100'] = df['boll_width_20'].rolling(100).rank(pct=True)
    df['boll_width_pct_200'] = df['boll_width_20'].rolling(200).rank(pct=True)
    df['atr_percentile_200'] = df['atr_10'].rolling(200).rank(pct=True)
 
    bw_mean = df['boll_width_20'].rolling(50).mean()
    bw_std = df['boll_width_20'].rolling(50).std()
    df['vol_zscore'] = (df['boll_width_20'] - bw_mean) / (bw_std + 1e-10)
 
    # --- H. SESSION/TIME FEATURES ---
    df['is_asian'] = ((hour >= 0) & (hour < 7)).astype(int)
    df['is_london'] = ((hour >= 7) & (hour < 13)).astype(int)
    df['is_ny'] = ((hour >= 13) & (hour < 20)).astype(int)
    df['is_overlap'] = ((hour >= 13) & (hour < 17)).astype(int)
 
    session_start = np.where(hour < 7, 0, np.where(hour < 13, 7, 13))
    df['mins_since_session'] = (hour - session_start) * 60 + minute
    df['mins_session_sin'] = np.sin(2 * np.pi * df['mins_since_session'] / 360)
    df['mins_session_cos'] = np.cos(2 * np.pi * df['mins_since_session'] / 360)
 
    # --- I. INTERACTIONS WITH boll_width ---
    df['combo_bw_flow3'] = df['boll_width_20'] * df['abs_flow_3']
    df['combo_bw_flow10'] = df['boll_width_20'] * df['abs_flow_10']
    df['combo_bw_dist8'] = df['boll_width_20'] * df['abs_dist_ema8']
    df['combo_bw_dist100'] = df['boll_width_20'] * df['abs_dist_ema100']
    df['combo_bw_spread'] = df['boll_width_20'] * df['abs_ema_spread']
    df['combo_bw_hour'] = df['boll_width_20'] * df['hour_cos']
    df['combo_bw_momentum'] = df['boll_width_20'] * df['flow_momentum'].abs()
    df['combo_bw_vol_roc'] = df['boll_width_20'] * df['boll_width_roc']
    df['combo_bw_squeeze'] = df['boll_width_20'] * df['squeeze_intensity']
    df['combo_gk_flow'] = df['gk_vol_10'] * df['abs_flow_10']
    df['combo_gk_dist'] = df['gk_vol_10'] * df['abs_dist_ema21']
 
    # Defragment the DataFrame before returning
    df = df.copy()
 
    return df.dropna()
 
 
def label_outcomes(df, horizon=30, target=3.0, stop=1.5):
    """Label outcomes using fixed dollar SL/TP"""
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
            if h >= entry + target:
                long_hit = True
                break
            if l <= entry - stop:
                break
 
        # Check SHORT
        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry - target:
                    short_hit = True
                    break
                if h >= entry + stop:
                    break
 
        outcomes.append(1 if long_hit else (2 if short_hit else 0))
 
    df = df.iloc[:len(outcomes)].copy()
    df['outcome'] = outcomes
    return df
 
 
# =============================================================================
# MAIN
# =============================================================================
print('='*80)
print('BACKTEST: hedging_strategy_strict model on XAUUSD23_24.csv')
print('='*80)
 
# Load model and feature names
print('\nLoading model...')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(FEATURES_PATH, 'r') as f:
    feature_names = f.read().strip().split('\n')
print(f'Model loaded: {len(feature_names)} features')
 
# Load data
print('\nLoading XAUUSD23_24.csv...')
df = pd.read_csv(DATA_PATH, sep='\t',
                 names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('Datetime', inplace=True)
df = df[['Open','High','Low','Close']].copy()
print(f'Loaded {len(df)} bars')
 
# Feature engineering
print('\nEngineering features...')
df = create_microstructure_features(df)
print(f'After features + dropna: {len(df)} bars')
 
# Label outcomes
print(f'\nLabeling outcomes ({HORIZON} bars, ${TARGET} target, ${STOP} stop)...')
df = label_outcomes(df, HORIZON, TARGET, STOP)
 
# Create combo features (same as hedging_strategy_strict.py does outside the function)
df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']
 
# Predict
print('\nRunning RF predictions...')
X = df[feature_names].fillna(0)
rf_probs = model.predict_proba(X)[:, 1]
df['rf_prob'] = rf_probs
 
# Baseline
base_success = (df['outcome'] != 0).sum() / len(df) * 100
print(f'\nBaseline success rate (any move): {base_success:.1f}%')
 
# Threshold sweep
print(f'\n{"Threshold":<12} {"Win %":<12} {"Freq %":<12} {"Count"}')
print('-'*55)
for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    filtered = df[rf_probs >= threshold]
    if len(filtered) > 0:
        wr = (filtered['outcome'] != 0).sum() / len(filtered) * 100
        freq = len(filtered) / len(df) * 100
        print(f'{threshold:<12.2f} {wr:>11.1f} {freq:>11.2f} {len(filtered):>11d}')
 
# =============================================================================
# HEDGING STRATEGY SIMULATION at RF_THRESHOLD
# =============================================================================
print(f'\n{"="*80}')
print(f'HEDGING STRATEGY RESULTS (RF >= {RF_THRESHOLD})')
print(f'{"="*80}')
print(f'Logic: Take BOTH LONG and SHORT when RF >= {RF_THRESHOLD}')
print(f'       Cancel whichever side hits stop first')
print(f'       TARGET=${TARGET}, STOP=${STOP}')
 
hedge_setups = df[df['rf_prob'] >= RF_THRESHOLD].copy()
print(f'\nTotal bars: {len(df)}')
print(f'Hedge setups: {len(hedge_setups)} ({len(hedge_setups)/len(df)*100:.2f}%)')
 
hedge_results = []
for idx, row in hedge_setups.iterrows():
    if row['outcome'] == 1:
        result = 'WIN'
        surviving = 'LONG'
    elif row['outcome'] == 2:
        result = 'WIN'
        surviving = 'SHORT'
    else:
        result = 'LOSS'
        surviving = 'BOTH_STOPPED'
 
    hedge_results.append({
        'Datetime': idx,
        'rf_prob': row['rf_prob'],
        'entry': row['Close'],
        'surviving_side': surviving,
        'result': result,
        'outcome': row['outcome']
    })
 
hedge_df = pd.DataFrame(hedge_results)
 
if len(hedge_df) > 0:
    wins = (hedge_df['result'] == 'WIN').sum()
    losses = len(hedge_df) - wins
    win_rate = wins / len(hedge_df) * 100
 
    # P&L: WIN = +TARGET - STOP = +1R, LOSS = -STOP - STOP = -2R
    total_pnl_r = wins * 1 + losses * (-2)
    avg_pnl_r = total_pnl_r / len(hedge_df)
 
    # Dollar P&L per 0.01 lot
    pnl_per_win = TARGET - STOP   # net gain on win
    pnl_per_loss = -STOP - STOP   # net loss on loss (both sides stopped)
    total_pnl_dollar = wins * pnl_per_win + losses * pnl_per_loss
 
    print(f'\n--- OVERALL PERFORMANCE ---')
    print(f'Total setups:  {len(hedge_df)}')
    print(f'Wins:          {wins}')
    print(f'Losses:        {losses}')
    print(f'Win Rate:      {win_rate:.1f}%')
    print(f'Net P&L:       {total_pnl_r}R')
    print(f'Avg per setup: {avg_pnl_r:.3f}R')
    print(f'Total $P&L:    ${total_pnl_dollar:.2f} (per 0.01 lot)')
 
    # Breakdown by surviving side
    print(f'\n--- SURVIVING SIDE ---')
    for side in ['LONG', 'SHORT', 'BOTH_STOPPED']:
        ct = (hedge_df['surviving_side'] == side).sum()
        if ct > 0:
            print(f'{side:<15} {ct:>6} ({ct/len(hedge_df)*100:>5.1f}%)')
 
    # Session breakdown
    hedge_df['hour'] = pd.to_datetime(hedge_df['Datetime']).dt.hour if not isinstance(hedge_df['Datetime'].iloc[0], pd.Timestamp) else hedge_df['Datetime'].dt.hour
    def get_session(h):
        if 0 <= h < 8: return 'ASIAN'
        elif 8 <= h < 13: return 'LONDON'
        elif 13 <= h < 17: return 'NY_OVERLAP'
        elif 17 <= h < 22: return 'NY'
        else: return 'LATE'
 
    hedge_df['session'] = hedge_df['hour'].apply(get_session)
 
    print(f'\n--- SESSION BREAKDOWN ---')
    for sess in ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']:
        sd = hedge_df[hedge_df['session'] == sess]
        if len(sd) > 0:
            sw = (sd['result'] == 'WIN').sum()
            sl = len(sd) - sw
            swr = sw / len(sd) * 100
            spnl = sw * 1 + sl * (-2)
            print(f'{sess:<12} Setups: {len(sd):<6} WR: {swr:>5.1f}% P&L: {spnl:>6}R')
 
    # Save results
    os.makedirs('data/processed', exist_ok=True)
    out = hedge_df.copy()
    out['Datetime'] = pd.to_datetime(out['Datetime']).dt.strftime('%Y-%m-%d %H:%M')
    out['rf_prob'] = out['rf_prob'].round(3)
    out['entry'] = out['entry'].round(2)
    out.to_csv('data/processed/BACKTEST_23_24.csv', index=False)
    print(f'\nSaved: data/processed/BACKTEST_23_24.csv')
else:
    print('\nNo hedge setups generated.')
 
print(f'\n{"="*80}')
print('BACKTEST COMPLETE')
print('='*80)