"""
Directional ML Live Trader — MetaTrader 5
==========================================
Uses the trained directional RF model from backtest_directional.py.
Best params from Optuna (200 trials):
  MODE: directional | TGT=$2.0 | SL=$0.50 | HOR=15
  RF: threshold=0.40, depth=20, trees=200
  Spread: $0.30/trade (1x directional)

Signal logic:
  - P(LONG_WIN)  >= threshold → BUY
  - P(SHORT_WIN) >= threshold → SELL
  - If both fire on the same bar, pick the higher probability.

Usage:
  python live_trader_directional.py --dry     # Dry run (no real trades)
  python live_trader_directional.py --live    # LIVE TRADING (real money!)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import time
import logging
import argparse
import sys
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================
SYMBOL         = "XAUUSDb"      # Adjust suffix for your broker
TIMEFRAME      = mt5.TIMEFRAME_M1
LOT_SIZE       = 0.01           # Start small
MAX_POSITIONS  = 3              # Max concurrent directional positions
CHECK_INTERVAL = 60             # Fallback sleep (seconds)

# Best params from Optuna — loaded from model file, but defaults here
TARGET_DOLLARS = 2.00
STOP_DOLLARS   = 0.50
RF_THRESHOLD   = 0.40

MAGIC_LONG     = 235001
MAGIC_SHORT    = 235002

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'output', 'directional_rf_model.pkl')
LOG_PATH   = os.path.join(SCRIPT_DIR, 'live_trader_directional.log')

# MT5 terminal path (None = auto-detect)
MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE ENGINEERING — must match backtest_directional.py exactly
# ============================================================================
def create_microstructure_features(df):
    df = df.copy()

    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)

    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)

    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()

    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))

    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)

    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - \
                         ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']

    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)

    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)

    df['flow_15'] = df['directional_flow'].rolling(15).sum()
    df['flow_20'] = df['directional_flow'].rolling(20).sum()
    df['flow_accel'] = df['flow_3'] - df['flow_3'].shift(3)
    df['flow_accel_5'] = df['flow_5'] - df['flow_5'].shift(5)
    df['abs_flow_3'] = df['flow_3'].abs()
    df['abs_flow_5'] = df['flow_5'].abs()
    df['abs_flow_10'] = df['flow_10'].abs()
    df['flow_divergence'] = (df['flow_3'] * df['flow_10'] < 0).astype(int)

    df['consecutive_up'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumcount() + 1
    df['consecutive_up'] = df['consecutive_up'] * df['is_up']
    df['consecutive_down'] = (1 - df['is_up']).groupby(((1-df['is_up']) != (1-df['is_up']).shift()).cumsum()).cumcount() + 1
    df['consecutive_down'] = df['consecutive_down'] * (1 - df['is_up'])
    df['max_consecutive'] = df[['consecutive_up', 'consecutive_down']].max(axis=1)

    df['flow_efficiency'] = df['abs_body'] / (df['range'] + 1e-10)
    df['flow_eff_3'] = df['flow_efficiency'].rolling(3).mean()
    df['flow_eff_5'] = df['flow_efficiency'].rolling(5).mean()

    df['atr_roc'] = (df['atr_3'] - df['atr_3'].shift(3)) / (df['atr_3'].shift(3) + 1e-10)
    df['vol_breakout'] = df['range'] / (df['atr_20'] + 1e-10)
    df['range_min_5'] = df['range'].rolling(5).min()
    df['range_max_5'] = df['range'].rolling(5).max()
    df['range_squeeze'] = df['range_min_5'] / (df['range_max_5'] + 1e-10)

    df['abs_dist_ema8'] = df['dist_ema8'].abs()
    df['dist_ema21'] = (df['Close'] - df['ema_21']) / df['Close']
    df['abs_dist_ema21'] = df['dist_ema21'].abs()
    df['ema_spread'] = (df['ema_8'] - df['ema_21']) / df['Close']
    df['abs_ema_spread'] = df['ema_spread'].abs()

    df['combo_abs_flow_vol'] = df['abs_flow_5'] * df['vol_ratio']
    df['combo_eff_flow'] = df['flow_eff_3'] * df['abs_flow_3']
    df['combo_consecutive_body'] = df['max_consecutive'] * df['body_pct']
    df['combo_atr_roc_accel'] = df['atr_roc'] * df['flow_accel'].abs()
    df['combo_squeeze_flow'] = (1 - df['range_squeeze']) * df['abs_flow_3']
    df['combo_dist_flow'] = df['abs_dist_ema8'] * df['abs_flow_5']

    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']

    return df.dropna()


# ============================================================================
# MT5 HELPERS
# ============================================================================
def connect_mt5():
    """Initialize connection to MT5."""
    if MT5_PATH and os.path.exists(MT5_PATH):
        if not mt5.initialize(path=MT5_PATH):
            logger.error(f"MT5 initialize() failed for path: {MT5_PATH}")
            return False
    else:
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

    logger.info(f"MT5 connected: version={mt5.version()}")

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        logger.error(f"Symbol {SYMBOL} not found")
        return False
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            logger.error(f"Failed to select symbol {SYMBOL}")
            return False

    logger.info(f"Symbol {SYMBOL} ready  |  point={symbol_info.point}  digits={symbol_info.digits}")
    return True


def get_filling_mode():
    """Auto-detect the correct filling mode for the symbol."""
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return None
    filling = info.filling_mode
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    return None


def get_bars(count=200):
    """Fetch recent M1 bars from MT5."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get rates: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df['Datetime'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close',
        'tick_volume': 'Volume',
    }, inplace=True)
    df.set_index('Datetime', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def wait_for_next_minute():
    """Wait until 2 seconds after the next minute (ensures bar is fully closed)."""
    now = datetime.now(timezone.utc)
    seconds_to_wait = 60 - now.second - (now.microsecond / 1_000_000) + 2
    if seconds_to_wait > 0:
        logger.info(f"Waiting {seconds_to_wait:.1f}s for next candle close...")
        time.sleep(seconds_to_wait)
    return datetime.now(timezone.utc)


def get_open_positions():
    """Return current open positions for SYMBOL with our magic numbers."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    return [p for p in positions if p.magic in (MAGIC_LONG, MAGIC_SHORT)]


def get_current_session():
    """Return current trading session label (UTC)."""
    h = datetime.now(timezone.utc).hour
    if 0 <= h < 7:
        return 'ASIAN'
    elif 7 <= h < 13:
        return 'LONDON'
    elif 13 <= h < 17:
        return 'NY_OVERLAP'
    elif 17 <= h < 22:
        return 'NY'
    return 'LATE'


# ============================================================================
# ORDER EXECUTION
# ============================================================================
def place_order(direction, entry_price, lot_size):
    """
    Place a directional order (BUY or SELL) with fixed-dollar SL/TP.
    direction: 'LONG' or 'SHORT'
    """
    filling = get_filling_mode()
    if filling is None:
        logger.error("Cannot determine filling mode — skipping order")
        return None

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        logger.error(f"Failed to get tick: {mt5.last_error()}")
        return None

    if direction == 'LONG':
        price = tick.ask
        sl = round(price - STOP_DOLLARS, 2)
        tp = round(price + TARGET_DOLLARS, 2)
        order_type = mt5.ORDER_TYPE_BUY
        magic = MAGIC_LONG
        comment = "DIR_LONG"
    else:
        price = tick.bid
        sl = round(price + STOP_DOLLARS, 2)
        tp = round(price - TARGET_DOLLARS, 2)
        order_type = mt5.ORDER_TYPE_SELL
        magic = MAGIC_SHORT
        comment = "DIR_SHORT"

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)
    if result is None:
        logger.error(f"order_send returned None: {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: retcode={result.retcode}  comment={result.comment}")
        return None

    logger.info(f"ORDER FILLED: {direction} @ {price:.2f}  SL={sl:.2f}  TP={tp:.2f}  ticket={result.order}")
    return result


# ============================================================================
# SIGNAL DETECTION
# ============================================================================
def check_signal(df, model, feature_names, classes):
    """
    Check the last COMPLETED bar for a directional signal.
    Returns: (direction, probability, reason_string)
        direction: 'LONG', 'SHORT', or None
    """
    if len(df) < 2:
        return None, 0.0, "Not enough data"

    # Use the last completed bar (iloc[-2] is completed, iloc[-1] is forming)
    latest = df.iloc[-2]
    X = pd.DataFrame([latest[feature_names].values], columns=feature_names).fillna(0)
    proba = model.predict_proba(X)[0]

    long_ci  = np.where(classes == 1)[0]
    short_ci = np.where(classes == 2)[0]

    prob_long  = proba[long_ci[0]]  if len(long_ci)  > 0 else 0.0
    prob_short = proba[short_ci[0]] if len(short_ci) > 0 else 0.0

    long_signal  = prob_long  >= RF_THRESHOLD
    short_signal = prob_short >= RF_THRESHOLD

    reason = f"P(L)={prob_long:.3f}  P(S)={prob_short:.3f}"

    if long_signal and short_signal:
        # Both above threshold — pick the stronger one
        if prob_long >= prob_short:
            return 'LONG', prob_long, f"SIGNAL LONG (both, picked higher) | {reason}"
        else:
            return 'SHORT', prob_short, f"SIGNAL SHORT (both, picked higher) | {reason}"
    elif long_signal:
        return 'LONG', prob_long, f"SIGNAL LONG | {reason}"
    elif short_signal:
        return 'SHORT', prob_short, f"SIGNAL SHORT | {reason}"
    else:
        return None, max(prob_long, prob_short), reason


# ============================================================================
# MAIN LOOP
# ============================================================================
def run(live_mode=False):
    banner = "LIVE TRADING" if live_mode else "DRY RUN"
    logger.info("=" * 72)
    logger.info(f"  DIRECTIONAL ML LIVE TRADER — {banner}")
    logger.info("=" * 72)

    # ── Load model ────────────────────────────────────────────────────
    logger.info(f"Loading model from {MODEL_PATH} ...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            bundle = pickle.load(f)
        model         = bundle['model']
        feature_names = bundle['features']
        params        = bundle['params']
        classes       = model.classes_
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Override globals with saved params
    global TARGET_DOLLARS, STOP_DOLLARS, RF_THRESHOLD
    TARGET_DOLLARS = params.get('target', TARGET_DOLLARS)
    STOP_DOLLARS   = params.get('stop', STOP_DOLLARS)
    RF_THRESHOLD   = params.get('threshold', RF_THRESHOLD)

    logger.info(f"  Model classes: {classes}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Params: TGT=${TARGET_DOLLARS}  SL=${STOP_DOLLARS}  TH={RF_THRESHOLD}")
    logger.info(f"  Symbol: {SYMBOL}  |  Lot: {LOT_SIZE}  |  Max positions: {MAX_POSITIONS}")
    logger.info("-" * 72)

    # ── Connect MT5 ───────────────────────────────────────────────────
    if not connect_mt5():
        return

    last_bar_time = None
    trade_count = 0

    # Sync to next minute boundary
    wait_for_next_minute()

    try:
        while True:
            # Fetch bars (need ~200 for rolling warmup)
            df = get_bars(300)
            if df is None:
                time.sleep(CHECK_INTERVAL)
                continue

            # Only act on NEW completed bars
            current_bar_time = df.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(5)
                continue
            last_bar_time = current_bar_time

            # Engineer features
            df_feat = create_microstructure_features(df)
            if len(df_feat) < 30:
                logger.warning("Not enough bars after feature warmup")
                wait_for_next_minute()
                continue

            # Check signal
            direction, prob, reason = check_signal(df_feat, model, feature_names, classes)

            session = get_current_session()
            price = df['Close'].iloc[-2]
            open_pos = get_open_positions()
            n_open = len(open_pos)

            if direction is not None:
                logger.info(f"{'='*72}")
                logger.info(f"  SIGNAL: {direction} @ {price:.2f}  prob={prob:.3f}  [{session}]")

                if n_open >= MAX_POSITIONS:
                    logger.info(f"  Skipped — max positions reached ({n_open}/{MAX_POSITIONS})")
                elif live_mode:
                    result = place_order(direction, price, LOT_SIZE)
                    if result:
                        trade_count += 1
                        logger.info(f"  Trade #{trade_count} executed")
                    else:
                        logger.error(f"  Order execution failed")
                else:
                    logger.info(f"  [DRY RUN] Would place {direction} @ {price:.2f}  SL={STOP_DOLLARS}  TP={TARGET_DOLLARS}")
                    trade_count += 1

                logger.info(f"{'='*72}")
            else:
                logger.info(
                    f"[{session}] {current_bar_time}  |  {price:.2f}  |  "
                    f"{reason}  |  pos={n_open}/{MAX_POSITIONS}"
                )

            # Wait for next candle close
            wait_for_next_minute()

    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C)...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        mt5.shutdown()
        logger.info(f"MT5 closed. Trades placed this session: {trade_count}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Directional ML Live Trader')
    parser.add_argument('--live', action='store_true', help='Enable LIVE trading (real money!)')
    parser.add_argument('--dry', action='store_true', help='Dry run mode (default)')
    args = parser.parse_args()

    if args.live:
        print("=" * 72)
        print("  WARNING: LIVE TRADING MODE — REAL MONEY!")
        print(f"  Symbol: {SYMBOL}  |  Lot: {LOT_SIZE}")
        print(f"  Target: ${TARGET_DOLLARS}  |  Stop: ${STOP_DOLLARS}")
        print("=" * 72)
        confirm = input("Type 'YES' to confirm: ")
        if confirm.strip() == "YES":
            run(live_mode=True)
        else:
            print("Aborted.")
    else:
        run(live_mode=False)
