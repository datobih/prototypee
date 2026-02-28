"""
XAUUSD Live Hedge Trader using MetaTrader 5
Model: hedging_strategy_strict_old.py (movement detection)
Strategy: Take BOTH LONG and SHORT when RF >= 0.70
          Cancel whichever side hits stop loss first
          Let surviving side run to target

Target: $2.5 (fixed dollar)
Stop: $0.5 (fixed dollar)
RF Threshold: 0.70
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

# Setup logging — save log next to this script
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live_trader_hedge.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(_LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MT5 Configuration
SYMBOL = 'XAUUSDz'  # Adjust suffix based on your broker
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK_BARS = 100  # Only need ~30 bars for deepest rolling window
MAX_HEDGE_PAIRS = 3  # Maximum simultaneous hedge pairs
LOT_SIZE = 0.01  # Fixed lot size
TARGET_DOLLARS = 2.5  # $2.5 target (fixed)
STOP_DOLLARS = 0.5  # $0.5 stop (fixed)
RF_THRESHOLD = 0.70  # RF probability threshold for entry
MAGIC_NUMBER_LONG = 123456
MAGIC_NUMBER_SHORT = 654321

# Load trained model from hedging_strategy_strict_old.py
try:
    with open('models/random_forest_old.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names_old.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')
    logger.info(f"Model loaded: random_forest_old.pkl ({len(feature_names)} features)")
    logger.info(f"Features: {feature_names}")
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
    """Create microstructure features matching hedging_strategy_strict_old.py exactly.
    Only builds the features needed for the 10-feature model."""
    df = df.copy()
    
    # Price structure
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)
    
    # Order flow
    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)
    
    # Deeper flow
    df['flow_20'] = df['directional_flow'].rolling(20).sum()
    df['abs_flow_3'] = df['flow_3'].abs()
    df['abs_flow_5'] = df['flow_5'].abs()
    df['abs_flow_10'] = df['flow_10'].abs()
    
    # Trend structure
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']
    
    # Distance features
    df['abs_dist_ema8'] = df['dist_ema8'].abs()
    df['dist_ema21'] = (df['Close'] - df['ema_21']) / df['Close']
    df['abs_dist_ema21'] = df['dist_ema21'].abs()
    df['ema_spread'] = (df['ema_8'] - df['ema_21']) / df['Close']
    df['abs_ema_spread'] = df['ema_spread'].abs()
    
    # Combo features
    df['combo_dist_flow'] = df['abs_dist_ema8'] * df['abs_flow_5']
    
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
    
    # Get latest COMPLETED bar (not the forming bar)
    latest = df[feature_names].iloc[-2:].head(1).fillna(0)
    
    # Predict using raw features (RF in hedging_strategy_strict_old was
    # trained on unscaled X_train — only LogReg used the scaler)
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
    logger.info(f"Model: hedging_strategy_strict_old.py (movement detection)")
    logger.info("="*80)
    logger.info(f"Strategy: Take BOTH LONG and SHORT when RF >= {RF_THRESHOLD}")
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
                if rf_prob >= RF_THRESHOLD and get_active_hedge_count() < MAX_HEDGE_PAIRS:
                    logger.info(f"✓ HEDGE SIGNAL: RF={rf_prob:.3f} >= {RF_THRESHOLD}")
                    
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
