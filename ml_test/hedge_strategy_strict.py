import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

# =============================================================================
# CONFIGURABLE PARAMETERS - Modify these values as needed
# =============================================================================
TARGET = 5   # Target profit in dollars (e.g., 5.0 = $5)
STOP =1          # Stop loss in dollars (e.g., 2.5 = $2.5)
HORIZON = 5      # Number of bars to look ahead for outcome
RF_THRESHOLD = 0.70 # Random Forest probability threshold for hedge entry

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
    
    # =====================================================================
    # NEW OPTIMIZED FEATURES (based on RF feature importance analysis)
    # Top RF features: flow_10, dist_ema8, flow_5, flow_momentum, flow_3
    # Strategy: deeper flow signals, flow quality, vol regime, interactions
    # =====================================================================
    
    # --- DEEPER FLOW FEATURES ---
    # Flow over longer windows (RF loves flow)
    df['flow_15'] = df['directional_flow'].rolling(15).sum()
    df['flow_20'] = df['directional_flow'].rolling(20).sum()
    
    # Flow acceleration (rate of change of flow)
    df['flow_accel'] = df['flow_3'] - df['flow_3'].shift(3)
    df['flow_accel_5'] = df['flow_5'] - df['flow_5'].shift(5)
    
    # Absolute flow magnitude (direction doesn't matter for hedge)
    df['abs_flow_3'] = df['flow_3'].abs()
    df['abs_flow_5'] = df['flow_5'].abs()
    df['abs_flow_10'] = df['flow_10'].abs()
    
    # Flow divergence: short-term vs long-term disagreement
    df['flow_divergence'] = (df['flow_3'] * df['flow_10'] < 0).astype(int)
    
    # --- FLOW QUALITY ---
    # Consecutive directional bars (how "clean" is the flow)
    df['consecutive_up'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumcount() + 1
    df['consecutive_up'] = df['consecutive_up'] * df['is_up']
    df['consecutive_down'] = (1 - df['is_up']).groupby(((1-df['is_up']) != (1-df['is_up']).shift()).cumsum()).cumcount() + 1
    df['consecutive_down'] = df['consecutive_down'] * (1 - df['is_up'])
    df['max_consecutive'] = df[['consecutive_up', 'consecutive_down']].max(axis=1)
    
    # Flow efficiency: how much of the range is in one direction
    df['flow_efficiency'] = df['abs_body'] / (df['range'] + 1e-10)
    df['flow_eff_3'] = df['flow_efficiency'].rolling(3).mean()
    df['flow_eff_5'] = df['flow_efficiency'].rolling(5).mean()
    
    # --- VOLATILITY REGIME ---
    # ATR rate of change (expanding or contracting)
    df['atr_roc'] = (df['atr_3'] - df['atr_3'].shift(3)) / (df['atr_3'].shift(3) + 1e-10)
    
    # Volatility breakout: current range vs recent average
    df['vol_breakout'] = df['range'] / (df['atr_20'] + 1e-10)
    
    # Range compression then expansion pattern
    df['range_min_5'] = df['range'].rolling(5).min()
    df['range_max_5'] = df['range'].rolling(5).max()
    df['range_squeeze'] = df['range_min_5'] / (df['range_max_5'] + 1e-10)
    
    # --- DISTANCE FEATURES (dist_ema8 is #2 in RF) ---
    df['abs_dist_ema8'] = df['dist_ema8'].abs()
    df['dist_ema21'] = (df['Close'] - df['ema_21']) / df['Close']
    df['abs_dist_ema21'] = df['dist_ema21'].abs()
    df['ema_spread'] = (df['ema_8'] - df['ema_21']) / df['Close']
    df['abs_ema_spread'] = df['ema_spread'].abs()
    
    # --- NEW COMBO FEATURES ---
    # Flow × volatility expansion (strong move + expanding vol = breakout)
    df['combo_abs_flow_vol'] = df['abs_flow_5'] * df['vol_ratio']
    # Flow efficiency × flow magnitude (clean strong moves)
    df['combo_eff_flow'] = df['flow_eff_3'] * df['abs_flow_3']
    # Consecutive bars × body size (sustained strong candles)
    df['combo_consecutive_body'] = df['max_consecutive'] * df['body_pct']
    # ATR expansion × flow acceleration (vol expanding while flow accelerating)
    df['combo_atr_roc_accel'] = df['atr_roc'] * df['flow_accel'].abs()
    # Squeeze release × flow (compression breaking with direction)
    df['combo_squeeze_flow'] = (1 - df['range_squeeze']) * df['abs_flow_3']
    # Distance from EMA × flow (overextended + strong flow = momentum)
    df['combo_dist_flow'] = df['abs_dist_ema8'] * df['abs_flow_5']
    
    # =====================================================================
    # ROUND 2 FEATURES — based on RF importance analysis
    # RF top themes: flow magnitude (42.5%), EMA distance (33.5%)
    # Strategy: deepen these, add normalized variants, z-scores, time
    # =====================================================================
    
    df = df.copy()  # defragment before Round 2 features
    
    # --- A. EXTENDED ABSOLUTE FLOW ---
    df['abs_flow_15'] = df['flow_15'].abs()
    df['abs_flow_20'] = df['flow_20'].abs()
    
    # Flow z-score: how extreme is current flow vs recent history
    flow10_mean = df['abs_flow_10'].rolling(50).mean()
    flow10_std = df['abs_flow_10'].rolling(50).std()
    df['flow_zscore'] = (df['abs_flow_10'] - flow10_mean) / (flow10_std + 1e-10)
    
    # Flow net ratio: how one-sided is the flow (1=perfectly one-sided)
    abs_per_bar = df['directional_flow'].abs()
    df['flow_net_ratio_5'] = df['abs_flow_5'] / (abs_per_bar.rolling(5).sum() + 1e-10)
    df['flow_net_ratio_10'] = df['abs_flow_10'] / (abs_per_bar.rolling(10).sum() + 1e-10)
    
    # Flow concentration: max single-bar contribution to total flow
    df['flow_max_bar_5'] = df['directional_flow'].abs().rolling(5).max()
    df['flow_concentration_5'] = df['flow_max_bar_5'] / (abs_per_bar.rolling(5).sum() + 1e-10)
    
    # --- B. LONGER EMA DISTANCE ---
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['ema_100'] = df['Close'].ewm(span=100).mean()
    df['dist_ema50'] = (df['Close'] - df['ema_50']) / df['Close']
    df['abs_dist_ema50'] = df['dist_ema50'].abs()
    df['dist_ema100'] = (df['Close'] - df['ema_100']) / df['Close']
    df['abs_dist_ema100'] = df['dist_ema100'].abs()
    
    # EMA spreads at multiple timeframes
    df['ema_spread_21_50'] = (df['ema_21'] - df['ema_50']) / df['Close']
    df['abs_ema_spread_21_50'] = df['ema_spread_21_50'].abs()
    df['ema_spread_50_100'] = (df['ema_50'] - df['ema_100']) / df['Close']
    df['abs_ema_spread_50_100'] = df['ema_spread_50_100'].abs()
    
    # Price z-score: how far from rolling mean in std units
    roll_mean_20 = df['Close'].rolling(20).mean()
    roll_std_20 = df['Close'].rolling(20).std()
    df['price_zscore_20'] = (df['Close'] - roll_mean_20) / (roll_std_20 + 1e-10)
    df['abs_price_zscore_20'] = df['price_zscore_20'].abs()
    
    roll_mean_50 = df['Close'].rolling(50).mean()
    roll_std_50 = df['Close'].rolling(50).std()
    df['price_zscore_50'] = (df['Close'] - roll_mean_50) / (roll_std_50 + 1e-10)
    df['abs_price_zscore_50'] = df['price_zscore_50'].abs()
    
    # Bollinger band position: where is price within the bands
    df['boll_upper_20'] = roll_mean_20 + 2 * roll_std_20
    df['boll_lower_20'] = roll_mean_20 - 2 * roll_std_20
    df['boll_width_20'] = (df['boll_upper_20'] - df['boll_lower_20']) / df['Close']
    df['boll_pct_20'] = (df['Close'] - df['boll_lower_20']) / (df['boll_upper_20'] - df['boll_lower_20'] + 1e-10)
    df['boll_outside_20'] = ((df['Close'] > df['boll_upper_20']) | (df['Close'] < df['boll_lower_20'])).astype(int)
    
    # --- C. ATR-NORMALIZED FEATURES (regime-independent) ---
    df['flow_per_atr_3'] = df['abs_flow_3'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['flow_per_atr_10'] = df['abs_flow_10'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['dist_ema8_per_atr'] = df['abs_dist_ema8'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['dist_ema21_per_atr'] = df['abs_dist_ema21'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['ema_spread_per_atr'] = df['abs_ema_spread'] / (df['atr_10'] / df['Close'] + 1e-10)
    
    # ATR percentile: where does current vol sit in recent history
    df['atr_percentile'] = df['atr_10'].rolling(100).rank(pct=True)
    
    # --- D. TIME FEATURES (cyclical encoding) ---
    hour = df.index.hour
    minute = df.index.minute
    minutes_in_day = hour * 60 + minute
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_of_week'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # --- E. HIGHER-ORDER INTERACTIONS (top features combined) ---
    # Flow × distance at multiple scales
    df['combo_flow3_dist21'] = df['abs_flow_3'] * df['abs_dist_ema21']
    df['combo_flow10_dist8'] = df['abs_flow_10'] * df['abs_dist_ema8']
    df['combo_flow10_spread'] = df['abs_flow_10'] * df['abs_ema_spread']
    # Flow × volatility regime
    df['combo_flow_atr_pct'] = df['abs_flow_10'] * df['atr_percentile']
    df['combo_flow_zscore_dist'] = df['flow_zscore'] * df['abs_dist_ema8']
    # Triple interaction: flow × distance × vol
    df['combo_triple'] = df['abs_flow_5'] * df['abs_dist_ema8'] * df['vol_ratio']
    # Flow quality × magnitude
    df['combo_net_ratio_flow'] = df['flow_net_ratio_10'] * df['abs_flow_10']
    df['combo_concentration_flow'] = df['flow_concentration_5'] * df['abs_flow_5']
    
    # =====================================================================
    # ROUND 3 FEATURES — volatility regime deep-dive
    # boll_width_20 was #1 at 20% importance, hour ~9%
    # Strategy: more vol measures, vol-of-vol, session, boll_width combos
    # =====================================================================
    
    # --- F. ADVANCED VOLATILITY MEASURES ---
    # Garman-Klass volatility (OHLC-based, more efficient estimator)
    log_hl = (np.log(df['High'] / df['Low']))**2
    log_co = (np.log(df['Close'] / df['Open']))**2
    gk_single = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df['gk_vol_10'] = gk_single.rolling(10).mean()
    df['gk_vol_20'] = gk_single.rolling(20).mean()
    df['gk_vol_50'] = gk_single.rolling(50).mean()
    
    # Parkinson volatility (high-low based)
    df['parkinson_vol_10'] = (log_hl / (4 * np.log(2))).rolling(10).mean()
    df['parkinson_vol_20'] = (log_hl / (4 * np.log(2))).rolling(20).mean()
    
    # Bollinger bands at multiple windows
    roll_mean_50 = df['Close'].rolling(50).mean()
    roll_std_50 = df['Close'].rolling(50).std()
    df['boll_width_50'] = (4 * roll_std_50) / df['Close']
    df['boll_pct_50'] = (df['Close'] - (roll_mean_50 - 2*roll_std_50)) / (4*roll_std_50 + 1e-10)
    
    roll_mean_100 = df['Close'].rolling(100).mean()
    roll_std_100 = df['Close'].rolling(100).std()
    df['boll_width_100'] = (4 * roll_std_100) / df['Close']
    
    # Keltner channel width (ATR-based bands)
    df['keltner_width_20'] = (2 * 1.5 * df['atr_20']) / df['Close']
    df['keltner_width_10'] = (2 * 1.5 * df['atr_10']) / df['Close']
    
    # Boll vs Keltner: squeeze detection (Bollinger inside Keltner = low vol squeeze)
    df['boll_inside_keltner'] = (df['boll_width_20'] < df['keltner_width_20']).astype(int)
    df['squeeze_intensity'] = df['keltner_width_20'] / (df['boll_width_20'] + 1e-10)
    
    # --- G. VOLATILITY OF VOLATILITY ---
    # How stable/unstable is the current vol regime
    df['vol_of_vol_20'] = df['boll_width_20'].rolling(20).std()
    df['vol_of_vol_50'] = df['boll_width_20'].rolling(50).std()
    
    # Vol regime change: is volatility expanding or contracting
    df['boll_width_roc'] = (df['boll_width_20'] - df['boll_width_20'].shift(5)) / (df['boll_width_20'].shift(5) + 1e-10)
    df['gk_vol_roc'] = (df['gk_vol_10'] - df['gk_vol_10'].shift(10)) / (df['gk_vol_10'].shift(10) + 1e-10)
    
    # Vol percentile at multiple windows
    df['boll_width_pct_50'] = df['boll_width_20'].rolling(50).rank(pct=True)
    df['boll_width_pct_100'] = df['boll_width_20'].rolling(100).rank(pct=True)
    df['boll_width_pct_200'] = df['boll_width_20'].rolling(200).rank(pct=True)
    df['atr_percentile_200'] = df['atr_10'].rolling(200).rank(pct=True)
    
    # Vol z-score: how extreme is current vol vs recent history
    bw_mean = df['boll_width_20'].rolling(50).mean()
    bw_std = df['boll_width_20'].rolling(50).std()
    df['vol_zscore'] = (df['boll_width_20'] - bw_mean) / (bw_std + 1e-10)
    
    # --- H. SESSION/TIME FEATURES (hour was ~9% importance) ---
    # Binary session indicators
    df['is_asian'] = ((hour >= 0) & (hour < 7)).astype(int)
    df['is_london'] = ((hour >= 7) & (hour < 13)).astype(int)
    df['is_ny'] = ((hour >= 13) & (hour < 20)).astype(int)
    df['is_overlap'] = ((hour >= 13) & (hour < 17)).astype(int)
    
    # Minutes since session open (captures intraday vol patterns)
    session_start = np.where(hour < 7, 0, np.where(hour < 13, 7, 13))
    df['mins_since_session'] = (hour - session_start) * 60 + minute
    df['mins_session_sin'] = np.sin(2 * np.pi * df['mins_since_session'] / 360)
    df['mins_session_cos'] = np.cos(2 * np.pi * df['mins_since_session'] / 360)
    
    # --- I. INTERACTIONS WITH boll_width (dominant feature) ---
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

def label_outcomes(df, horizon=15, target=5.0, stop=2.5):
    """
    Label outcomes using fixed dollar SL/TP.

    For bars where BOTH the TP and SL are touched within the same 1-minute
    candle (double-touch), we use Candle Structure Wick Analysis to resolve:
    
    LONG:
    - Bullish candle (C >= O): Did it wick down before going up?
      If (Open - Low) >= STOP -> SL hit first -> LOSS. Else WIN.
    - Bearish candle (C < O): Price moved down -> SL hit first -> LOSS.
    
    SHORT:
    - Bearish candle (C <= O): Did it wick up before going down?
      If (High - Open) >= STOP -> SL hit first -> LOSS. Else WIN.
    - Bullish candle (C > O): Price moved up -> SL hit first -> LOSS.
    """
    df = df.copy()
    outcomes = []

    for i in range(len(df) - horizon):
        if i % 20000 == 0:
            print(f'  Labeling {i}/{len(df)-horizon}...')

        entry  = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]

        long_hi_tp  = entry + target
        long_lo_sl  = entry - stop
        short_lo_tp = entry - target
        short_hi_sl = entry + stop

        # --- LONG ---
        long_hit = False
        for o, h, l, c in zip(future['Open'], future['High'], future['Low'], future['Close']):
            hit_tp = h >= long_hi_tp
            hit_sl = l <= long_lo_sl

            if hit_tp and hit_sl:
                # Double-touch: use Candle Structure Wick Analysis
                if c >= o:  # Bullish candle
                    if (o - l) >= stop:
                        pass # SL hit on the bottom wick first -> LOSS
                    else:
                        long_hit = True # Went up first, bottom wick wasn't deep enough -> WIN
                else: # Bearish candle (C < O)
                    pass # Price moved down -> SL hit first -> LOSS
                break  # resolved (win or loss) — stop scanning
            elif hit_sl:
                break  # clean SL hit first
            elif hit_tp:
                long_hit = True
                break   # clean TP hit first

        # --- SHORT (only if LONG didn't trigger) ---
        short_hit = False
        if not long_hit:
            for o, h, l, c in zip(future['Open'], future['High'], future['Low'], future['Close']):
                hit_tp = l <= short_lo_tp
                hit_sl = h >= short_hi_sl

                if hit_tp and hit_sl:
                    # Double-touch: use Candle Structure Wick Analysis
                    if c <= o: # Bearish candle
                        if (h - o) >= stop:
                            pass # SL hit on the top wick first -> LOSS
                        else:
                            short_hit = True # Went down first, top wick wasn't high enough -> WIN
                    else: # Bullish candle (C > O)
                        pass # Price moved up -> SL hit first -> LOSS
                    break  # resolved
                elif hit_sl:
                    break  # clean SL hit first
                elif hit_tp:
                    short_hit = True
                    break   # clean TP hit first

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
df = df[['Open','High','Low','Close']].copy()  # Keep only OHLC
print(f'Loaded {len(df)} bars (1-minute timeframe)')

print('\nEngineering features...')
df = create_microstructure_features(df)

print(f'\nLabeling outcomes ({HORIZON} bars, ${TARGET} target, ${STOP} stop)...')
df = label_outcomes(df, HORIZON, TARGET, STOP)

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

# Split 60:40 with purge gap to prevent label leakage at boundary
split = int(len(df) * 0.6)
purge = HORIZON  # drop HORIZON bars between train/test so labels don't peek
test = df.iloc[split + purge:].copy()

print(f'\nTrain set: {split} bars (60%)')
print(f'Purge gap: {purge} bars (prevents label leakage)')
print(f'Test set: {len(test)} bars (40% minus purge)')
print(f'Base success rate: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

safe_trades = test['outcome'] != 0

# All features for RF training
all_features = [
    # Original
    'abs_ema_spread', 'abs_flow_10', 'abs_flow_3', 'abs_flow_5',
    'abs_dist_ema21', 'combo_dist_flow', 'abs_dist_ema8',
    'flow_momentum', 'ema_spread', 'flow_20',
    # Round 2: extended flow, EMA distance, z-scores, time
    'abs_flow_15', 'abs_flow_20', 'flow_zscore',
    'flow_net_ratio_5', 'flow_net_ratio_10', 'flow_concentration_5',
    'abs_dist_ema50', 'abs_dist_ema100',
    'abs_ema_spread_21_50', 'abs_ema_spread_50_100',
    'abs_price_zscore_20', 'abs_price_zscore_50',
    'boll_width_20', 'boll_pct_20', 'boll_outside_20',
    'flow_per_atr_3', 'flow_per_atr_10',
    'dist_ema8_per_atr', 'dist_ema21_per_atr', 'ema_spread_per_atr',
    'atr_percentile',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'combo_flow3_dist21', 'combo_flow10_dist8', 'combo_flow10_spread',
    'combo_flow_atr_pct', 'combo_flow_zscore_dist', 'combo_triple',
    'combo_net_ratio_flow', 'combo_concentration_flow',
    # Round 3: volatility regime
    'gk_vol_10', 'gk_vol_20', 'gk_vol_50',
    'parkinson_vol_10', 'parkinson_vol_20',
    'boll_width_50', 'boll_width_100', 'boll_pct_50',
    'keltner_width_20', 'keltner_width_10',
    'boll_inside_keltner', 'squeeze_intensity',
    'vol_of_vol_20', 'vol_of_vol_50',
    'boll_width_roc', 'gk_vol_roc',
    'boll_width_pct_50', 'boll_width_pct_100', 'boll_width_pct_200',
    'atr_percentile_200', 'vol_zscore',
    'is_asian', 'is_london', 'is_ny', 'is_overlap',
    'mins_since_session', 'mins_session_sin', 'mins_session_cos',
    'combo_bw_flow3', 'combo_bw_flow10', 'combo_bw_dist8',
    'combo_bw_dist100', 'combo_bw_spread', 'combo_bw_hour',
    'combo_bw_momentum', 'combo_bw_vol_roc', 'combo_bw_squeeze',
    'combo_gk_flow', 'combo_gk_dist',
]

# ============================================================================
# TRAIN RF MODEL
# ============================================================================
print('\n' + '='*80)
print('TRAINING RANDOM FOREST')
print('='*80)

train = df.iloc[:split].copy()
X_train = train[all_features].fillna(0)
y_train = (train['outcome'] != 0).astype(int)
X_test = test[all_features].fillna(0)
y_test = safe_trades.astype(int)

print(f'Training on {len(X_train)} samples with {len(all_features)} features')

rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=100,
    min_samples_split=200, max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print(f'AUC-ROC: {rf_auc:.4f}')

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
print(f'\nBaseline: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

# ============================================================================
# HEDGING STRATEGY - FORWARD TEST
# ============================================================================
print('\n' + '='*80)
print(f'HEDGING STRATEGY - FORWARD TEST (RF >= {RF_THRESHOLD})')
print('='*80)
print(f'Logic: Take BOTH LONG and SHORT when RF >= {RF_THRESHOLD}')
print('       Cancel whichever side hits stop loss first')
print('       Keep the surviving trade until target/stop')

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

# Filter for high probability setups (RF >= RF_THRESHOLD)
hedge_setups = test_hedge[test_hedge['rf_prob'] >= RF_THRESHOLD].copy()

print(f'\nTotal test bars: {len(test_hedge)}')
print(f'High probability setups (RF >= {RF_THRESHOLD}): {len(hedge_setups)} ({len(hedge_setups)/len(test_hedge)*100:.2f}%)')

# For each setup, determine which side survives and which gets stopped out
hedge_results = []

for idx, row in hedge_setups.iterrows():
    entry = row['Close']
    
    # LONG trade parameters (fixed dollar values)
    long_target = entry + TARGET     # $TARGET target
    long_stop = entry - STOP         # $STOP stop
    
    # SHORT trade parameters (fixed dollar values)
    short_target = entry - TARGET    # $TARGET target
    short_stop = entry + STOP        # $STOP stop
    
    # Determine outcome based on actual market movement
    # outcome: 1 = LONG wins, 2 = SHORT wins, 0 = no clear direction
    
    if row['outcome'] == 1:
        # Market went up - LONG wins, SHORT stopped
        surviving_side = 'LONG'
        cancelled_side = 'SHORT'
        result = 'WIN'
    elif row['outcome'] == 2:
        # Market went down - SHORT wins, LONG stopped
        surviving_side = 'SHORT'
        cancelled_side = 'LONG'
        result = 'WIN'
    else:
        # No clear direction - both could have been stopped or neither hit target
        surviving_side = 'BOTH_STOPPED'
        cancelled_side = 'BOTH_STOPPED'
        result = 'LOSS'
    
    hedge_results.append({
        'Datetime': row['Datetime'],
        'session': row['session'],
        'rf_prob': row['rf_prob'],
        'entry': entry,
        'surviving_side': surviving_side,
        'cancelled_side': cancelled_side,
        'result': result,
        'outcome': row['outcome']
    })

# Convert to DataFrame
hedge_df = pd.DataFrame(hedge_results)

if len(hedge_df) > 0:
    print(f'\nTotal hedged setups: {len(hedge_df)}')
    
    # Performance metrics
    wins = (hedge_df['result'] == 'WIN').sum()
    losses = len(hedge_df) - wins
    win_rate = wins / len(hedge_df) * 100
    
    print(f'\n--- OVERALL PERFORMANCE ---')
    print(f'Total setups: {len(hedge_df)}')
    print(f'Wins: {wins}')
    print(f'Losses: {losses}')
    print(f'Win Rate: {win_rate:.1f}%')
    
    # Net P&L calculation
    # WIN: +TARGET - STOP = +$5 - $2.5 = +$2.5 = +1R
    # LOSS (both stopped): -STOP - STOP = -$2.5 - $2.5 = -$5.0 = -2R
    
    total_pnl = wins * 1 + losses * (-2)
    avg_pnl = total_pnl / len(hedge_df)
    
    print(f'\nNet P&L: {total_pnl}R')
    print(f'Average per setup: {avg_pnl:.3f}R')
    print(f'Expected Value: {avg_pnl:.3f}R per setup')
    
    # Breakdown by surviving side
    print(f'\n--- SURVIVING SIDE BREAKDOWN ---')
    for side in ['LONG', 'SHORT', 'BOTH_STOPPED']:
        side_trades = hedge_df[hedge_df['surviving_side'] == side]
        if len(side_trades) > 0:
            count = len(side_trades)
            pct = count / len(hedge_df) * 100
            print(f'{side:<15} {count:>6} ({pct:>5.1f}%)')
    
    # Session breakdown
    print(f'\n--- SESSION BREAKDOWN ---')
    for sess in hedge_df['session'].unique():
        sess_data = hedge_df[hedge_df['session'] == sess]
        sess_wins = (sess_data['result'] == 'WIN').sum()
        sess_losses = len(sess_data) - sess_wins
        sess_wr = sess_wins / len(sess_data) * 100 if len(sess_data) > 0 else 0
        sess_pnl = sess_wins * 1 + sess_losses * (-2)
        print(f'{sess:<12} Setups: {len(sess_data):<6} WR: {sess_wr:>5.1f}% P&L: {sess_pnl:>6}R')
    
    # Save results
    hedge_df['Datetime'] = hedge_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    hedge_df['rf_prob'] = hedge_df['rf_prob'].round(3)
    hedge_df['entry'] = hedge_df['entry'].round(2)
    hedge_df.to_csv('data/processed/HEDGE_strategy.csv', index=False)
    print(f'\nSaved: data/processed/HEDGE_strategy.csv')
    
    # Show sample trades
    print('\n--- SAMPLE HEDGE SETUPS (first 20) ---')
    print(hedge_df.head(20).to_string(index=False))
else:
    print('\nNo hedge setups generated.')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)