"""
XAUUSD Live Hedge Trader using MetaTrader 5
Strategy: Take BOTH LONG and SHORT when RF >= 0.75
          Cancel whichever side hits stop loss first
          Let surviving side run to target

Target: $3.0 (fixed dollar)
Stop: $1.5 (fixed dollar)
RR: 2:1 per side, net 1:2 for hedge (breakeven ~66.7% WR)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('live_trader_hedge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MT5 Configuration
SYMBOL = 'XAUUSDb'  # Adjust suffix based on your broker
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK_BARS = 500  # Need enough bars for rolling(200), ema_100, etc.
MAX_HEDGE_PAIRS = 3  # Maximum simultaneous hedge pairs
LOT_SIZE = 0.01  # Fixed lot size (same as live_trader.py)
TARGET_DOLLARS = 3.0  # $3 target (fixed)
STOP_DOLLARS = 1.5  # $1.5 stop (fixed)
MAGIC_NUMBER_LONG = 123456
MAGIC_NUMBER_SHORT = 654321

# Load trained model
try:
    with open('models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')
    logger.info(f"Model loaded successfully with {len(feature_names)} features")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    sys.exit(1)

def connect_mt5(mt5_path=None):
    """Connect to MetaTrader 5"""
    if mt5_path and os.path.exists(mt5_path):
        if not mt5.initialize(path=mt5_path):
            logger.error(f"MT5 initialize() failed from path: {mt5_path}")
            return False
    else:
        if not mt5.initialize():
            logger.error("MT5 initialize() failed")
            return False
    
    logger.info(f"MT5 initialized. Version: {mt5.version()}")
    logger.info(f"Terminal: {mt5.terminal_info()}")
    
    # Check symbol
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        logger.error(f"Symbol {SYMBOL} not found")
        return False
    
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            logger.error(f"Failed to select symbol {SYMBOL}")
            return False
    
    logger.info(f"Symbol {SYMBOL} configured successfully")
    return True

def wait_for_next_minute():
    """Wait until 2 seconds after the next minute (ensures bar is fully closed)"""
    now = datetime.now(timezone.utc)
    # Calculate seconds until next minute + 2 second buffer
    seconds_to_wait = 60 - now.second - (now.microsecond / 1_000_000) + 2
    if seconds_to_wait > 0:
        logger.info(f"Waiting {seconds_to_wait:.1f}s until next candle closes...")
        time.sleep(seconds_to_wait)
    return datetime.now(timezone.utc)

def create_microstructure_features(df):
    """Create all microstructure features for the model
    Must match hedging_strategy_strict.py exactly"""
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
    
    # Momentum consistency
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
    # ROUND 2 FEATURES — extended flow, EMA distance, z-scores, time
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
    
    # --- D. TIME FEATURES (cyclical encoding) ---
    hour = df.index.hour
    minute = df.index.minute
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
    
    df = df.copy()
    return df.dropna()

def get_market_data():
    """Fetch recent market data from MT5"""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK_BARS)
    if rates is None or len(rates) == 0:
        logger.error("Failed to get market data")
        return None
    
    df = pd.DataFrame(rates)
    df['Datetime'] = pd.to_datetime(df['time'], unit='s')
    # Rename to match training column names (capitalized)
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    })
    df.set_index('Datetime', inplace=True)
    
    return df

def calculate_rf_probability():
    """Calculate Random Forest probability for current bar"""
    df = get_market_data()
    if df is None:
        return None, None
    
    # Create features
    df = create_microstructure_features(df)
    
    # Get latest bar features (use feature_names loaded from file)
    # Use iloc[-2] to get the COMPLETED bar, not the forming bar
    latest = df[feature_names].iloc[-2:].head(1).fillna(0)
    
    # Predict using raw features (RF was trained on unscaled data)
    rf_prob = model.predict_proba(latest)[0][1]
    
    current_bar = df.iloc[-2]
    
    return rf_prob, current_bar

def get_hedge_pairs():
    """Get all active hedge pairs (positions with matching magic numbers)"""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    
    hedge_pairs = []
    long_positions = [p for p in positions if p.magic == MAGIC_NUMBER_LONG]
    short_positions = [p for p in positions if p.magic == MAGIC_NUMBER_SHORT]
    
    # Match by comment or timing
    for long_pos in long_positions:
        for short_pos in short_positions:
            # Consider them a pair if opened within 1 minute of each other
            time_diff = abs(long_pos.time - short_pos.time)
            if time_diff <= 60:
                hedge_pairs.append({
                    'long': long_pos,
                    'short': short_pos,
                    'entry_time': min(long_pos.time, short_pos.time)
                })
                break
    
    return hedge_pairs

def get_active_hedge_count():
    """Count active hedge pairs"""
    return len(get_hedge_pairs())

def get_filling_mode():
    """Auto-detect the correct filling mode for the symbol"""
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        logger.error(f"Cannot get symbol info for {SYMBOL} — skipping trade")
        return None
    filling = info.filling_mode
    # filling_mode is a bitmask: bit 0 (1) = FOK, bit 1 (2) = IOC
    if filling & 1:  # FOK supported
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:  # IOC supported
        return mt5.ORDER_FILLING_IOC
    else:
        logger.error(f"Neither FOK nor IOC supported (filling_mode={filling}) — skipping trade")
        return None

def place_hedge_orders(entry_price, volume):
    """Place both LONG and SHORT orders simultaneously"""
    
    filling_mode = get_filling_mode()
    if filling_mode is None:
        return False
    
    # LONG order (fixed dollar SL/TP)
    long_sl = entry_price - STOP_DOLLARS
    long_tp = entry_price + TARGET_DOLLARS
    
    long_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(SYMBOL).ask,
        "sl": long_sl,
        "tp": long_tp,
        "magic": MAGIC_NUMBER_LONG,
        "comment": f"HEDGE_LONG_{int(time.time())}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    
    # SHORT order (fixed dollar SL/TP)
    short_sl = entry_price + STOP_DOLLARS
    short_tp = entry_price - TARGET_DOLLARS
    
    short_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(SYMBOL).bid,
        "sl": short_sl,
        "tp": short_tp,
        "magic": MAGIC_NUMBER_SHORT,
        "comment": f"HEDGE_SHORT_{int(time.time())}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    
    # Send LONG order
    long_result = mt5.order_send(long_request)
    if long_result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"LONG order failed: {long_result.retcode}, {long_result.comment}")
        return False
    
    logger.info(f"LONG order placed: ticket={long_result.order}, price={long_request['price']:.2f}, SL={long_sl:.2f}, TP={long_tp:.2f}")
    
    # Send SHORT order
    short_result = mt5.order_send(short_request)
    if short_result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"SHORT order failed: {short_result.retcode}, {short_result.comment}")
        # Close the LONG order since SHORT failed
        close_position(long_result.order)
        return False
    
    logger.info(f"SHORT order placed: ticket={short_result.order}, price={short_request['price']:.2f}, SL={short_sl:.2f}, TP={short_tp:.2f}")
    logger.info(f"✓ HEDGE PAIR CREATED (RF signal)")
    
    return True

def close_position(ticket):
    """Close a specific position by ticket"""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return True  # Already closed
    
    position = position[0]
    
    filling_mode = get_filling_mode()
    if filling_mode is None:
        return False
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": ticket,
        "price": mt5.symbol_info_tick(SYMBOL).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask,
        "magic": position.magic,
        "comment": "HEDGE_CANCEL",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    
    result = mt5.order_send(close_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Failed to close position {ticket}: {result.retcode}")
        return False
    
    return True

def monitor_hedge_pairs():
    """Monitor hedge pairs and close the losing side if one hits stop"""
    pairs = get_hedge_pairs()
    
    for pair in pairs:
        long_pos = pair['long']
        short_pos = pair['short']
        
        current_price = mt5.symbol_info_tick(SYMBOL).bid
        
        # Check if LONG hit stop loss
        long_stopped = current_price <= long_pos.sl
        
        # Check if SHORT hit stop loss
        short_stopped = current_price >= short_pos.sl
        
        if long_stopped and not short_stopped:
            logger.info(f"LONG position {long_pos.ticket} stopped out at {current_price:.2f}, keeping SHORT {short_pos.ticket}")
            # LONG already closed by broker SL, no action needed
            
        elif short_stopped and not long_stopped:
            logger.info(f"SHORT position {short_pos.ticket} stopped out at {current_price:.2f}, keeping LONG {long_pos.ticket}")
            # SHORT already closed by broker SL, no action needed
            
        elif long_stopped and short_stopped:
            logger.info(f"BOTH positions stopped out at {current_price:.2f} - PAIR LOSS")

def main():
    parser = argparse.ArgumentParser(description='XAUUSD Hedge Live Trader')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: dry run)')
    parser.add_argument('--mt5-path', type=str, help='Path to MT5 terminal.exe')
    args = parser.parse_args()
    
    LIVE_MODE = args.live
    mode_str = "LIVE TRADING" if LIVE_MODE else "DRY RUN"
    
    logger.info("="*80)
    logger.info(f"XAUUSD HEDGE TRADER - {mode_str}")
    logger.info("="*80)
    logger.info(f"Strategy: Take BOTH LONG and SHORT when RF >= 0.75")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Lot size: {LOT_SIZE}")
    logger.info(f"Target: ${TARGET_DOLLARS}, Stop: ${STOP_DOLLARS}")
    logger.info(f"Max hedge pairs: {MAX_HEDGE_PAIRS}")
    logger.info("="*80)
    
    if not connect_mt5(args.mt5_path):
        sys.exit(1)
    
    last_bar_time = None
    
    # Sync with minute boundary on startup
    wait_for_next_minute()
    
    try:
        while True:
            # Get current bar
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
            if rates is None or len(rates) == 0:
                time.sleep(5)
                continue
            
            current_bar_time = rates[0]['time']
            
            # Process on new bar
            if current_bar_time != last_bar_time:
                last_bar_time = current_bar_time
                
                # Monitor existing hedge pairs
                monitor_hedge_pairs()
                
                # Check for new hedge signal
                rf_prob, current_bar = calculate_rf_probability()
                
                if rf_prob is None:
                    logger.warning("Failed to calculate RF probability")
                    wait_for_next_minute()
                    continue
                
                current_time = datetime.now(timezone.utc)
                logger.info(f"[{current_time.strftime('%Y-%m-%d %H:%M')}] Price: {current_bar['Close']:.2f} | RF: {rf_prob:.3f} | Pairs: {get_active_hedge_count()}/{MAX_HEDGE_PAIRS}")
                
                # Check if we can open new hedge pair
                if rf_prob >= 0.75 and get_active_hedge_count() < MAX_HEDGE_PAIRS:
                    logger.info(f"✓ HEDGE SIGNAL: RF={rf_prob:.3f} >= 0.75")
                    
                    if LIVE_MODE:
                        logger.info(f"Placing hedge orders with volume {LOT_SIZE}...")
                        success = place_hedge_orders(current_bar['Close'], LOT_SIZE)
                        if success:
                            logger.info("✓ Hedge pair placed successfully")
                        else:
                            logger.error("✗ Failed to place hedge pair")
                    else:
                        logger.info("[DRY RUN] Would place hedge pair here")
            
            # Wait for next minute boundary (synchronized with candle close)
            wait_for_next_minute()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed")

if __name__ == "__main__":
    main()
