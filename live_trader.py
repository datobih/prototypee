"""
XAUUSD Live Trader using MetaTrader 5
Rule 4: RF >= 0.70 + Bullish Pattern + NY_OVERLAP Session

Conditions:
- RF Probability >= 0.70
- Last 3 bars: ALL UP (up_count_3 == 3)
- Current bar: BIG BODY (> 1.5x 10-bar avg)
- Close position > 0.7 (close near high)
- Session: NY_OVERLAP only (13:00-17:00 UTC)
- Direction: LONG only
- Target: 0.1% (10 pips on gold)
- Stop: 0.05% (5 pips on gold)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime, timezone
import logging
import math

# ============================================================================
# CONFIGURATION
# ============================================================================
SYMBOL = "XAUUSDm"  # Note: Your broker uses 'm' suffix
TIMEFRAME = mt5.TIMEFRAME_M1
LOT_SIZE = 0.01  # Start small - adjust based on your account
TARGET_PCT = 0.001  # 0.1%
STOP_PCT = 0.0005   # 0.05%
RF_THRESHOLD = 0.70
CHECK_INTERVAL = 60  # Check every 60 seconds
MAX_POSITIONS = 3   # Maximum concurrent positions allowed

# MT5 Terminal Path (leave None to use default, or specify path to terminal64.exe)
# Example: r"C:\Program Files\MetaTrader 5\terminal64.exe"

#C:\Program Files\MetaTrader 5\terminal64.exe
#C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe

MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# Session times (UTC)
NY_OVERLAP_START = 13
NY_OVERLAP_END = 17

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE ENGINEERING (same as analyze_features.py)
# ============================================================================
def create_microstructure_features(df):
    """Create all features needed for the model"""
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
    
    # Momentum consistency - is_up = higher close than previous bar
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
    
    return df

# ============================================================================
# MT5 HELPER FUNCTIONS
# ============================================================================
def connect_mt5():
    """Initialize connection to MT5"""
    if MT5_PATH:
        if not mt5.initialize(MT5_PATH):
            logger.error(f"MT5 initialize() failed for path: {MT5_PATH}")
            logger.error(f"Error: {mt5.last_error()}")
            return False
    else:
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed, error: {mt5.last_error()}")
            return False
    
    logger.info(f"MT5 connected: {mt5.terminal_info()}")
    logger.info(f"MT5 version: {mt5.version()}")
    return True

def get_bars(symbol, timeframe, count=100):
    """Get OHLCV bars from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get rates: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'time': 'Datetime',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    df.set_index('Datetime', inplace=True)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def wait_for_next_minute():
    """Wait until 2 seconds after the next minute (ensures bar is fully closed)"""
    now = datetime.now(timezone.utc)
    # Calculate seconds until next minute + 2 second buffer
    seconds_to_wait = 60 - now.second - (now.microsecond / 1_000_000) + 2
    if seconds_to_wait > 0:
        logger.info(f"Waiting {seconds_to_wait:.1f}s until next candle closes...")
        time.sleep(seconds_to_wait)
    return datetime.now(timezone.utc)

def get_current_session():
    """Get current trading session based on UTC hour"""
    utc_hour = datetime.now(timezone.utc).hour
    if NY_OVERLAP_START <= utc_hour < NY_OVERLAP_END:
        return 'NY_OVERLAP'
    elif 0 <= utc_hour < 8:
        return 'ASIAN'
    elif 8 <= utc_hour < 13:
        return 'LONDON'
    elif 17 <= utc_hour < 22:
        return 'NY'
    else:
        return 'LATE'

def get_open_position_count(symbol):
    """Get the number of open positions for a symbol"""
    positions = mt5.positions_get(symbol=symbol)
    return len(positions) if positions is not None else 0

def get_symbol_info(symbol):
    """Get symbol info for proper lot sizing and pricing"""
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"Symbol {symbol} not found")
        return None
    return info

def place_buy_order(symbol, lot_size, stop_loss, take_profit):
    """Place a BUY order with SL and TP"""
    symbol_info = get_symbol_info(symbol)
    if symbol_info is None:
        return None
    
    # Get current ask price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Failed to get tick: {mt5.last_error()}")
        return None
    
    price = tick.ask
    
    # Prepare the request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 20,  # Max slippage in points
        "magic": 234000,  # EA identifier
        "comment": "Rule4_LONG",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send the order
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: {result.retcode} - {result.comment}")
        return None
    
    logger.info(f"Order placed: {result}")
    return result

# ============================================================================
# SIGNAL DETECTION
# ============================================================================
def check_rule4_signal(df, rf_model, feature_names):
    """
    Check if current bar satisfies Rule 4 conditions:
    - RF Probability >= 0.70
    - up_count_3 == 3 (last 3 bars up)
    - big_body == 1
    - close_position > 0.7
    - Session: NY_OVERLAP
    """
    # Get the last COMPLETED bar (df.iloc[-1] is the forming bar, -2 is completed)
    latest = df.iloc[-2]

    # Always calculate RF probability first (use DataFrame to avoid sklearn warning)
    X = pd.DataFrame([latest[feature_names].values], columns=feature_names)
    X = X.fillna(0)
    rf_prob = rf_model.predict_proba(X)[0, 1]
    
    # Check session first (cheapest check)
    session = get_current_session()
    if session != 'NY_OVERLAP':
        return False, f"Wrong session: {session} | rf={rf_prob:.3f}", rf_prob
    
    # Check pattern conditions
    up_count_3 = latest['up_count_3']
    big_body = latest['big_body']
    close_position = latest['close_position']
    
    if up_count_3 != 3:
        return False, f"up3={int(up_count_3)} bb={int(big_body)} cp={close_position:.2f} | rf={rf_prob:.3f}", rf_prob
    
    if big_body != 1:
        return False, f"up3={int(up_count_3)} bb={int(big_body)} cp={close_position:.2f} | rf={rf_prob:.3f}", rf_prob
    
    if close_position <= 0.7:
        return False, f"up3={int(up_count_3)} bb={int(big_body)} cp={close_position:.2f} | rf={rf_prob:.3f}", rf_prob
    
    if rf_prob < RF_THRESHOLD:
        return False, f"up3={int(up_count_3)} bb={int(big_body)} cp={close_position:.2f} | rf={rf_prob:.3f} (need >={RF_THRESHOLD})", rf_prob
    
    # All conditions met!
    return True, f"SIGNAL! up3={int(up_count_3)} bb={int(big_body)} cp={close_position:.2f} | rf={rf_prob:.3f}", rf_prob

# ============================================================================
# MAIN TRADING LOOP
# ============================================================================
def run_live_trader():
    """Main trading loop"""
    logger.info("="*60)
    logger.info("XAUUSD LIVE TRADER - RULE 4")
    logger.info("="*60)
    
    # Connect to MT5
    if not connect_mt5():
        return
    
    # Load the trained model
    logger.info("Loading model...")
    try:
        with open('models/random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/feature_names.txt', 'r') as f:
            feature_names = f.read().strip().split('\n')
        logger.info(f"Model loaded with {len(feature_names)} features")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        mt5.shutdown()
        return
    
    # Verify symbol is available
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"Failed to select {SYMBOL}")
        mt5.shutdown()
        return
    
    logger.info(f"Trading {SYMBOL} on M1 timeframe")
    logger.info(f"Lot size: {LOT_SIZE}")
    logger.info(f"Target: {TARGET_PCT*100}% | Stop: {STOP_PCT*100}%")
    logger.info(f"RF threshold: {RF_THRESHOLD}")
    logger.info(f"Session: NY_OVERLAP only (13:00-17:00 UTC)")
    logger.info("-"*60)
    
    last_bar_time = None
    
    # Sync with minute boundary on startup
    wait_for_next_minute()
    
    try:
        while True:
            # Get current bars
            df = get_bars(SYMBOL, TIMEFRAME, 100)
            if df is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Check if we have a new bar
            current_bar_time = df.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(5)  # No new bar, wait a bit
                continue
            
            last_bar_time = current_bar_time
            
            # Create features
            df = create_microstructure_features(df)
            if len(df) < 30:
                logger.warning("Not enough data for features")
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Check for signal
            signal, reason, rf_prob = check_rule4_signal(df, rf_model, feature_names)
            
            current_price = df['Close'].iloc[-1]
            session = get_current_session()
            
            if signal:
                logger.info(f"{'='*60}")
                logger.info(f"🚨 SIGNAL DETECTED at {current_bar_time}")
                logger.info(f"Price: {current_price:.2f} | RF Prob: {rf_prob:.3f}")
                
                # Check if we can open another position
                open_count = get_open_position_count(SYMBOL)
                if open_count >= MAX_POSITIONS:
                    logger.info(f"Max positions reached ({open_count}/{MAX_POSITIONS}) - skipping")
                else:
                    # Calculate SL and TP
                    stop_loss = round(current_price * (1 - STOP_PCT), 2)
                    take_profit = round(current_price * (1 + TARGET_PCT), 2)
                    
                    logger.info(f"Placing BUY order:")
                    logger.info(f"  Entry: {current_price:.2f}")
                    logger.info(f"  SL: {stop_loss:.2f} ({STOP_PCT*100}%)")
                    logger.info(f"  TP: {take_profit:.2f} ({TARGET_PCT*100}%)")
                    
                    # Place the order
                    result = place_buy_order(SYMBOL, LOT_SIZE, stop_loss, take_profit)
                    
                    if result:
                        logger.info(f"✅ Order executed successfully!")
                    else:
                        logger.error(f"❌ Order failed!")
                
                logger.info(f"{'='*60}")
            else:
                # Log status every minute
                logger.info(f"[{session}] {current_bar_time} | Price: {current_price:.2f} | {reason}")
            
            # Wait for next minute boundary (candle formation)
            wait_for_next_minute()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed")

# ============================================================================
# DRY RUN MODE (for testing without real trades)
# ============================================================================
def run_dry_mode():
    """Run in simulation mode - no real trades"""
    logger.info("="*60)
    logger.info("XAUUSD LIVE TRADER - DRY RUN MODE")
    logger.info("="*60)
    
    # Connect to MT5
    if not connect_mt5():
        return
    
    # Load the trained model
    logger.info("Loading model...")
    try:
        with open('models/random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/feature_names.txt', 'r') as f:
            feature_names = f.read().strip().split('\n')
        logger.info(f"Model loaded with {len(feature_names)} features")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        mt5.shutdown()
        return
    
    # Verify symbol is available
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"Failed to select {SYMBOL}")
        mt5.shutdown()
        return
    
    logger.info(f"Monitoring {SYMBOL} on M1 timeframe (DRY RUN)")
    logger.info("-"*60)
    
    last_bar_time = None
    
    # Sync with minute boundary on startup
    wait_for_next_minute()
    
    try:
        while True:
            df = get_bars(SYMBOL, TIMEFRAME, 100)
            if df is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            current_bar_time = df.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(5)
                continue
            
            last_bar_time = current_bar_time
            
            df = create_microstructure_features(df)
            if len(df) < 30:
                time.sleep(CHECK_INTERVAL)
                continue
            
            signal, reason, rf_prob = check_rule4_signal(df, rf_model, feature_names)
            
            current_price = df['Close'].iloc[-1]
            session = get_current_session()
            
            if signal:
                stop_loss = round(current_price * (1 - STOP_PCT), 2)
                take_profit = round(current_price * (1 + TARGET_PCT), 2)
                
                logger.info(f"{'='*60}")
                logger.info(f"🚨 [DRY RUN] SIGNAL at {current_bar_time}")
                logger.info(f"  Price: {current_price:.2f}")
                logger.info(f"  RF Prob: {rf_prob:.3f}")
                logger.info(f"  SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                logger.info(f"{'='*60}")
            else:
                logger.info(f"[{session}] {current_bar_time} | {current_price:.2f} | {reason}")
            
            # Wait for next minute boundary (candle formation)
            wait_for_next_minute()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--dry":
        run_dry_mode()
    else:
        print("="*60)
        print("XAUUSD LIVE TRADER")
        print("="*60)
        print("\nUsage:")
        print("  python live_trader.py --dry   : Dry run (no real trades)")
        print("  python live_trader.py --live  : LIVE TRADING (real money!)")
        print("\n⚠️  WARNING: --live mode will place REAL trades!")
        print()
        
        if len(sys.argv) > 1 and sys.argv[1] == "--live":
            confirm = input("Type 'YES' to confirm live trading: ")
            if confirm == "YES":
                run_live_trader()
            else:
                print("Aborted.")
        else:
            print("Starting dry run mode...")
            run_dry_mode()
