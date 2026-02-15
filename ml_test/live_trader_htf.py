"""
Directional ML Live Trader — HTF Model + HP55_session_vol Filter
=================================================================
Uses the trained HTF RF model (66 features: 26 base + 40 HTF).
Applies HP55_session_vol confluence filter:
  - ML probability threshold >= 0.55
  - London/NY session only (hours 7-17 UTC)
  - vol_ratio > 1.0 (above-average volatility)

Backtested performance (walk-forward test):
  68.8% WR | PF=4.78 | 5/5 months profitable | 17/17 weeks profitable

Usage:
  python live_trader_htf.py --dry     # Dry run (no real trades)
  python live_trader_htf.py --live    # LIVE TRADING (real money!)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import time
import logging
import argparse
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================
SYMBOL         = "XAUUSDb"      # Adjust suffix for your broker
TIMEFRAME      = mt5.TIMEFRAME_M1
LOT_SIZE       = 0.01           # Start small
MAX_POSITIONS  = 3              # Max concurrent directional positions

# HP55_session_vol filter params
RF_THRESHOLD   = 0.55           # Raised from 0.40 to 0.55
SESSION_START  = 7              # UTC hour — London open
SESSION_END    = 17             # UTC hour — NY close
MIN_VOL_RATIO  = 1.0            # ATR3/ATR10 must be above this

# Trade params (overridden by model file if present)
TARGET_DOLLARS = 2.00
STOP_DOLLARS   = 0.50

MAGIC_LONG     = 236001
MAGIC_SHORT    = 236002

# Data management
BARS_INITIAL     = 80000        # ~55 days for daily EMA50 warmup
BARS_PER_LOOP    = 1000         # Fresh 1min bars each loop
HTF_REFRESH_MIN  = 15           # Recompute HTF features every N minutes

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'output', 'directional_rf_model_htf.pkl')
LOG_PATH   = os.path.join(SCRIPT_DIR, 'live_trader_htf.log')

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
# FEATURE ENGINEERING — must match backtest_directional_experiment.py exactly
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


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ── Higher Timeframe Features ──────────────────────────────────────────────
def create_htf_features(df):
    """
    Resample 1min bars to 15min, 1H, 4H, Daily.
    Compute trend/momentum/structure on each HTF.
    Merge back to 1min index with shift(1) to avoid look-ahead.
    """
    htf_dfs = {}

    timeframes = {
        '15min': '15min',
        '1h':    '1h',
        '4h':    '4h',
        '1d':    '1D',
    }

    for tf_name, resample_rule in timeframes.items():
        vol_col = 'Volume' if 'Volume' in df.columns else 'TickVol'
        htf = df.resample(resample_rule).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            vol_col: 'sum',
        }).dropna()

        htf[f'{tf_name}_ema8']  = htf['Close'].ewm(span=8).mean()
        htf[f'{tf_name}_ema21'] = htf['Close'].ewm(span=21).mean()
        htf[f'{tf_name}_ema50'] = htf['Close'].ewm(span=50).mean()

        htf[f'{tf_name}_trend'] = (
            ((htf['Close'] > htf[f'{tf_name}_ema8']) &
             (htf[f'{tf_name}_ema8'] > htf[f'{tf_name}_ema21'])).astype(int) -
            ((htf['Close'] < htf[f'{tf_name}_ema8']) &
             (htf[f'{tf_name}_ema8'] < htf[f'{tf_name}_ema21'])).astype(int)
        )

        htf[f'{tf_name}_trend50'] = (
            (htf['Close'] > htf[f'{tf_name}_ema50']).astype(int) -
            (htf['Close'] < htf[f'{tf_name}_ema50']).astype(int)
        )

        htf[f'{tf_name}_dist_ema21'] = (htf['Close'] - htf[f'{tf_name}_ema21']) / htf['Close']
        htf[f'{tf_name}_rsi'] = compute_rsi(htf['Close'], 14)

        htf_body = htf['Close'] - htf['Open']
        htf_flow = htf_body / htf['Close']
        htf[f'{tf_name}_flow_3'] = htf_flow.rolling(3).sum()
        htf[f'{tf_name}_flow_5'] = htf_flow.rolling(5).sum()

        htf_range = htf['High'] - htf['Low']
        htf[f'{tf_name}_close_pos'] = (htf['Close'] - htf['Low']) / (htf_range + 1e-10)

        htf_atr3  = htf_range.rolling(3).mean()
        htf_atr10 = htf_range.rolling(10).mean()
        htf[f'{tf_name}_vol_ratio'] = htf_atr3 / (htf_atr10 + 1e-10)
        htf[f'{tf_name}_vol_exp'] = (htf_range > htf_atr10 * 1.2).astype(int)
        htf[f'{tf_name}_range_pct'] = htf_range / htf['Close']

        feat_cols = [c for c in htf.columns if c.startswith(f'{tf_name}_')]
        htf_feat = htf[feat_cols].shift(1)
        htf_dfs[tf_name] = htf_feat

    result = df.copy()
    for tf_name, htf_feat in htf_dfs.items():
        htf_reindexed = htf_feat.reindex(result.index, method='ffill')
        result = pd.concat([result, htf_reindexed], axis=1)

    return result


BASE_FEATURES = [
    'abs_ema_spread', 'abs_flow_10', 'abs_flow_3', 'abs_flow_5',
    'abs_dist_ema21', 'combo_dist_flow', 'abs_dist_ema8',
    'flow_momentum', 'ema_spread', 'flow_20',
    'flow_3', 'flow_5', 'flow_10', 'trend_align', 'imbalance_3',
    'imbalance_5', 'close_position', 'vol_ratio', 'vol_expansion',
    'consistency_3', 'consistency_5', 'dist_ema8',
    'combo_flow_trend', 'combo_vol_imbalance',
    'combo_imbalance_momentum', 'combo_vol_flow',
]

HTF_FEATURES = []
for _tf in ['15min', '1h', '4h', '1d']:
    HTF_FEATURES.extend([
        f'{_tf}_trend', f'{_tf}_trend50', f'{_tf}_dist_ema21',
        f'{_tf}_rsi', f'{_tf}_flow_3', f'{_tf}_flow_5',
        f'{_tf}_close_pos', f'{_tf}_vol_ratio', f'{_tf}_vol_exp',
        f'{_tf}_range_pct',
    ])

FEATURE_NAMES = BASE_FEATURES + HTF_FEATURES


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


def get_bars(count=1000):
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
        comment = "HTF_LONG"
    else:
        price = tick.bid
        sl = round(price + STOP_DOLLARS, 2)
        tp = round(price - TARGET_DOLLARS, 2)
        order_type = mt5.ORDER_TYPE_SELL
        magic = MAGIC_SHORT
        comment = "HTF_SHORT"

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
# HTF DATA MANAGER — caches HTF features, refreshes periodically
# ============================================================================
class HTFDataManager:
    """
    Manages HTF feature computation with caching.
    - On startup: fetches large history to warm up daily EMA50 etc.
    - Each minute: provides fresh 1min features with cached HTF values.
    - Refreshes HTF every 15 minutes (when new 15min candle closes).
    """

    def __init__(self):
        self.htf_cache = None       # DataFrame with HTF features (indexed by datetime)
        self.last_htf_refresh = 0   # timestamp of last HTF recompute

    def initialize(self):
        """Fetch large history and compute all HTF features."""
        logger.info(f"Fetching {BARS_INITIAL:,} bars for HTF warmup...")
        df = get_bars(BARS_INITIAL)
        if df is None or len(df) < 1000:
            logger.error("Failed to fetch enough history for HTF features")
            return False

        logger.info(f"  Got {len(df):,} bars ({df.index[0]} → {df.index[-1]})")
        logger.info("  Computing 1min features...")
        df_feat = create_microstructure_features(df)
        logger.info("  Computing HTF features (15min/1H/4H/Daily)...")
        df_feat = create_htf_features(df_feat)

        # Cache only the HTF columns
        self.htf_cache = df_feat[HTF_FEATURES].copy()
        self.last_htf_refresh = time.time()

        n_valid = self.htf_cache.dropna().shape[0]
        logger.info(f"  HTF cache ready: {n_valid:,} valid rows")
        return True

    def needs_refresh(self):
        """Check if HTF features need recomputation."""
        return (time.time() - self.last_htf_refresh) > (HTF_REFRESH_MIN * 60)

    def refresh(self):
        """Re-fetch and recompute HTF features."""
        logger.info("Refreshing HTF features...")
        return self.initialize()

    def get_features(self):
        """
        Fetch fresh 1min bars, compute 1min features,
        merge with cached HTF features.
        Returns full-featured DataFrame or None.
        """
        # Refresh HTF if needed
        if self.needs_refresh():
            self.refresh()

        # Fetch recent bars for fresh 1min features
        df = get_bars(BARS_PER_LOOP)
        if df is None or len(df) < 50:
            return None

        df_feat = create_microstructure_features(df)
        if len(df_feat) < 30:
            return None

        # Merge HTF features from cache via index alignment + forward-fill
        if self.htf_cache is not None:
            for col in HTF_FEATURES:
                if col in self.htf_cache.columns:
                    # Reindex cache to match 1min index, forward-fill
                    df_feat[col] = self.htf_cache[col].reindex(df_feat.index, method='ffill')

        return df_feat


# ============================================================================
# SIGNAL DETECTION with HP55_session_vol filter
# ============================================================================
def check_signal(df_feat, model, feature_names, classes):
    """
    Check the last COMPLETED bar for a directional signal.
    Applies HP55_session_vol filter:
      1. ML probability >= 0.55
      2. Hour 7-17 UTC (London/NY)
      3. vol_ratio > 1.0

    Returns: (direction, probability, reason_string)
        direction: 'LONG', 'SHORT', or None
    """
    if len(df_feat) < 2:
        return None, 0.0, "Not enough data"

    # Use the last completed bar (iloc[-2] is completed, iloc[-1] is forming)
    latest = df_feat.iloc[-2]

    # ── Filter 1: Session check (hours 7-17 UTC) ──
    bar_hour = latest.name.hour if hasattr(latest.name, 'hour') else 0
    if bar_hour < SESSION_START or bar_hour > SESSION_END:
        return None, 0.0, f"Outside session (hour={bar_hour}, need {SESSION_START}-{SESSION_END})"

    # ── Filter 2: Volume check (vol_ratio > 1.0) ──
    vol_ratio = latest.get('vol_ratio', 0.0)
    if pd.isna(vol_ratio) or vol_ratio <= MIN_VOL_RATIO:
        return None, 0.0, f"Low vol_ratio={vol_ratio:.2f} (need >{MIN_VOL_RATIO})"

    # ── ML prediction ──
    # Check for NaN in HTF features
    feat_values = latest[feature_names].values
    if np.any(np.isnan(feat_values)):
        n_nan = np.sum(np.isnan(feat_values))
        return None, 0.0, f"NaN in features ({n_nan} missing)"

    X = pd.DataFrame([feat_values], columns=feature_names)
    proba = model.predict_proba(X)[0]

    long_ci  = np.where(classes == 1)[0]
    short_ci = np.where(classes == 2)[0]

    prob_long  = proba[long_ci[0]]  if len(long_ci)  > 0 else 0.0
    prob_short = proba[short_ci[0]] if len(short_ci) > 0 else 0.0

    # ── Filter 3: Probability threshold (0.55) ──
    long_signal  = prob_long  >= RF_THRESHOLD
    short_signal = prob_short >= RF_THRESHOLD

    reason = f"P(L)={prob_long:.3f}  P(S)={prob_short:.3f}  vol={vol_ratio:.2f}  h={bar_hour}"

    if long_signal and short_signal:
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
    logger.info(f"  HTF DIRECTIONAL ML LIVE TRADER — {banner}")
    logger.info(f"  Filter: HP55_session_vol (prob>=0.55, h={SESSION_START}-{SESSION_END}, vol>{MIN_VOL_RATIO})")
    logger.info("=" * 72)

    # ── Load model ────────────────────────────────────────────────────
    logger.info(f"Loading HTF model from {MODEL_PATH} ...")
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

    # Override trade params from model file
    global TARGET_DOLLARS, STOP_DOLLARS
    TARGET_DOLLARS = params.get('target', TARGET_DOLLARS)
    STOP_DOLLARS   = params.get('stop', STOP_DOLLARS)

    # Verify feature alignment
    if set(feature_names) != set(FEATURE_NAMES):
        missing = set(feature_names) - set(FEATURE_NAMES)
        extra   = set(FEATURE_NAMES) - set(feature_names)
        if missing:
            logger.warning(f"Model expects features not in our list: {missing}")
        if extra:
            logger.warning(f"We compute features the model doesn't use: {extra}")

    logger.info(f"  Model classes: {classes}")
    logger.info(f"  Features: {len(feature_names)} ({len(BASE_FEATURES)} base + {len(HTF_FEATURES)} HTF)")
    logger.info(f"  Params: TGT=${TARGET_DOLLARS}  SL=${STOP_DOLLARS}  TH={RF_THRESHOLD}")
    logger.info(f"  Symbol: {SYMBOL}  |  Lot: {LOT_SIZE}  |  Max positions: {MAX_POSITIONS}")
    logger.info("-" * 72)

    # ── Connect MT5 ───────────────────────────────────────────────────
    if not connect_mt5():
        return

    # ── Initialize HTF data manager ──────────────────────────────────
    data_mgr = HTFDataManager()
    if not data_mgr.initialize():
        logger.error("Failed to initialize HTF data — aborting.")
        mt5.shutdown()
        return

    last_bar_time = None
    trade_count = 0

    # Sync to next minute boundary
    wait_for_next_minute()

    try:
        while True:
            # Get features (1min fresh + cached HTF)
            df_feat = data_mgr.get_features()
            if df_feat is None:
                logger.warning("Failed to get features — retrying next minute")
                wait_for_next_minute()
                continue

            # Only act on NEW completed bars
            current_bar_time = df_feat.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(5)
                continue
            last_bar_time = current_bar_time

            # Check signal (with HP55_session_vol filter applied inside)
            direction, prob, reason = check_signal(
                df_feat, model, feature_names, classes
            )

            session = get_current_session()
            price = df_feat['Close'].iloc[-2]
            open_pos = get_open_positions()
            n_open = len(open_pos)

            if direction is not None:
                logger.info(f"{'='*72}")
                logger.info(f"  SIGNAL: {direction} @ {price:.2f}  prob={prob:.3f}  [{session}]")
                logger.info(f"  {reason}")

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
                    logger.info(f"  [DRY RUN] Would place {direction} @ {price:.2f}  "
                               f"SL=${STOP_DOLLARS}  TP=${TARGET_DOLLARS}")
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
    parser = argparse.ArgumentParser(description='HTF Directional ML Live Trader')
    parser.add_argument('--live', action='store_true', help='Enable LIVE trading (real money!)')
    parser.add_argument('--dry', action='store_true', help='Dry run mode (default)')
    args = parser.parse_args()

    if args.live:
        print("=" * 72)
        print("  WARNING: LIVE TRADING MODE — REAL MONEY!")
        print(f"  Model: HTF (66 features)")
        print(f"  Filter: HP55_session_vol (prob>=0.55, h=7-17, vol>1.0)")
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
