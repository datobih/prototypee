"""
XAUUSD Regime Grid Live Trader
================================
Architecture:
  - Every 15-min bar close  : recompute regime (CI, ER, Hurst, AC1, ADX)
  - Every 5-min bar close   : execute grid logic against current regime
  - RANGING regime          : place BUY/SELL LIMIT orders at grid levels
  - TRENDING/UNCERTAIN      : cancel all pending orders, close all open positions

Run modes:
  python grid_live_trader.py            # dry run (no real orders)
  python grid_live_trader.py --live     # live trading

Parameters must match ranging_regime.py and grid_backtest.py exactly.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import argparse
from datetime import datetime, timezone
import sys
import os

# ============================================================================
# CONFIG — keep in sync with ranging_regime.py and grid_backtest.py
# ============================================================================

SYMBOL             = 'XAUUSDz'     # adjust suffix for your broker
LOT_SIZE           = 0.01
MAGIC_NUMBER       = 777777        # all grid orders tagged with this

ATR_PERIOD         = 14            # bars for ATR (matches grid_backtest.py)
ATR_MULTIPLIER     = 0.5           # ATR fraction -> grid spacing
MIN_SPACING        = 1.0           # minimum $ between grid levels
MAX_SPACING        = 15.0          # maximum $ between grid levels
GRID_LEVELS        = 6             # levels each side of center
RANGE_LOOKBACK     = 20            # 15-min bars for grid center
MAX_OPEN_POSITIONS = 6             # max simultaneous filled grid positions

# Regime thresholds — chosen on 2022-2025 data, do not re-tune on live data
RANGING_THRESHOLD  = 0.62
TRENDING_THRESHOLD = 0.40

# Indicator periods (in 15-min bars)
CI_PERIOD  = 14    # 3.5 hours
ER_PERIOD  = 20    # 5 hours
HURST_WIN  = 50    # 12.5 hours  — need 50+ bars of history before trading
AC_WIN     = 30    # 7.5 hours
ADX_PERIOD = 14    # 3.5 hours

WARMUP_BARS = 60   # 15-min bars fetched to ensure indicators are warmed up

# Indicator weights
W_CI    = 0.30
W_ER    = 0.25
W_HURST = 0.20
W_AC    = 0.15
W_ADX   = 0.10

# Session bias (UTC hours)
SESSION_BIAS = {
    'ASIAN':      +0.05,
    'LONDON':     -0.08,
    'NY_OVERLAP': -0.08,
    'NY':         -0.03,
    'LATE':       +0.03,
}

# Emergency SL — wide backstop in case process crashes mid-trade.
# NOT in backtest — purely a live safety net. Set to None to disable.
# Fixed $ distance so it stays predictable regardless of ATR level.
EMERGENCY_SL_DIST = 20.0  # SL = entry +/- this many dollars

# ============================================================================
# LOGGING
# ============================================================================
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_live_trader.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(_LOG_PATH),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ============================================================================
# REGIME INDICATORS — identical to ranging_regime.py
# ============================================================================

def _choppiness_index(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_sum  = tr.rolling(period).sum()
    hh       = high.rolling(period).max()
    ll       = low.rolling(period).min()
    hl_range = (hh - ll).replace(0, np.nan)
    return (100 * np.log10(atr_sum / hl_range) / np.log10(period)).clip(0, 100)


def _efficiency_ratio(close, period=20):
    direction = close.diff(period).abs()
    noise     = close.diff().abs().rolling(period).sum()
    return (direction / noise.replace(0, np.nan)).clip(0, 1)


def _hurst_rs_single(arr):
    if len(arr) < 10:
        return 0.5
    mean   = arr.mean()
    devs   = arr - mean
    cumdev = np.cumsum(devs)
    R = cumdev.max() - cumdev.min()
    S = arr.std(ddof=1)
    if S == 0 or R == 0:
        return 0.5
    return np.log(R / S) / np.log(len(arr))


def _rolling_hurst(close, window=50):
    return close.rolling(window).apply(_hurst_rs_single, raw=True)


def _lag1_autocorr(close, window=30):
    returns = close.pct_change()
    def _ac(arr):
        if len(arr) < 4:
            return 0.0
        r, r_lag = arr[:-1], arr[1:]
        if r.std() == 0 or r_lag.std() == 0:
            return 0.0
        return np.corrcoef(r, r_lag)[0, 1]
    return returns.rolling(window).apply(_ac, raw=True)


def _adx(high, low, close, period=14):
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    dm_plus  = high - prev_high
    dm_minus = prev_low - low
    dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    atr      = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_sum   = (di_plus + di_minus).replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / di_sum
    return dx.ewm(span=period, adjust=False).mean()


def compute_regime(df15):
    """
    Compute ranging regime from a DataFrame of closed 15-min bars.
    Returns (regime_str, ranging_score) using the LAST closed bar.
    df15 must have columns: Open, High, Low, Close, and a DatetimeIndex.
    """
    if len(df15) < WARMUP_BARS:
        return 'UNCERTAIN', 0.5, 4.0   # fallback ATR

    df = df15.copy()

    # Raw indicators
    df['CI']    = _choppiness_index(df['High'], df['Low'], df['Close'], CI_PERIOD)
    df['ER']    = _efficiency_ratio(df['Close'], ER_PERIOD)
    df['Hurst'] = _rolling_hurst(df['Close'], HURST_WIN)
    df['AC1']   = _lag1_autocorr(df['Close'], AC_WIN)
    df['ADX']   = _adx(df['High'], df['Low'], df['Close'], ADX_PERIOD)

    # Normalize to [0,1] ranging scores
    df['ci_score']    = ((df['CI']    - 38.2) / (61.8 - 38.2)).clip(0, 1)
    df['er_score']    = (1.0 - df['ER']).clip(0, 1)
    df['hurst_score'] = ((0.65 - df['Hurst']) / (0.65 - 0.20)).clip(0, 1)
    df['ac_score']    = ((-df['AC1'] + 0.3) / (0.5 + 0.3)).clip(0, 1)
    df['adx_score']   = ((45 - df['ADX']) / (45 - 15)).clip(0, 1)

    # Composite score
    df['score'] = (
        W_CI    * df['ci_score'] +
        W_ER    * df['er_score'] +
        W_HURST * df['hurst_score'] +
        W_AC    * df['ac_score'] +
        W_ADX   * df['adx_score']
    )

    # Session bias on last bar
    hour = df.index[-1].hour
    if   0  <= hour < 7:  session = 'ASIAN'
    elif 7  <= hour < 12: session = 'LONDON'
    elif 12 <= hour < 17: session = 'NY_OVERLAP'
    elif 17 <= hour < 21: session = 'NY'
    else:                 session = 'LATE'

    bias  = SESSION_BIAS.get(session, 0.0)
    score = float(np.clip(df['score'].iloc[-1] + bias, 0, 1))

    if score >= RANGING_THRESHOLD:
        regime = 'RANGING'
    elif score <= TRENDING_THRESHOLD:
        regime = 'TRENDING'
    else:
        regime = 'UNCERTAIN'

    # ATR on the 15-min bars — same EWM formula used in grid_backtest.py.
    # Use iloc[-2] (the bar BEFORE the one that just closed) to match the
    # backtest's shift(1) alignment: at bar-close T, the backtest uses ATR
    # from T-1 for the next set of 5-min bars.
    prev_close  = df['Close'].shift(1)
    tr          = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series  = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    current_atr = float(atr_series.iloc[-2])

    return regime, score, current_atr


# ============================================================================
# MT5 HELPERS
# ============================================================================

def connect_mt5(mt5_path=None):
    if mt5_path and os.path.exists(mt5_path):
        ok = mt5.initialize(path=mt5_path)
    else:
        ok = mt5.initialize()
    if not ok:
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    log.info(f"MT5 connected. Version: {mt5.version()}")
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        log.error(f"Symbol {SYMBOL} not found")
        return False
    if not info.visible:
        mt5.symbol_select(SYMBOL, True)
    log.info(f"Symbol: {SYMBOL}  point={info.point}  digits={info.digits}")
    return True


def get_filling_mode():
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return None
    if info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    elif info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    return None


def get_bars(timeframe, count):
    """Fetch `count` closed bars. Returns DataFrame or None."""
    # count+1 to exclude the currently forming bar
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 1, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time').rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'tick_volume': 'TickVol',
    })
    return df[['Open', 'High', 'Low', 'Close', 'TickVol']]


def get_current_price():
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None, None
    return tick.bid, tick.ask


def get_grid_orders():
    """All pending orders with our magic number."""
    orders = mt5.orders_get(symbol=SYMBOL)
    if orders is None:
        return []
    return [o for o in orders if o.magic == MAGIC_NUMBER]


def get_grid_positions():
    """All open positions with our magic number."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    return [p for p in positions if p.magic == MAGIC_NUMBER]


def active_price_levels():
    """Return set of price levels that already have a pending or open order."""
    levels = set()
    for o in get_grid_orders():
        levels.add(round(o.price_open, 2))
    for p in get_grid_positions():
        levels.add(round(p.price_open, 2))
    return levels


def build_grid_levels(center, spacing, n_levels):
    levels = []
    for i in range(-n_levels, n_levels + 1):
        levels.append(round(center + i * spacing, 2))
    return sorted(levels)


def place_limit_order(side, price, tp, live_mode, grid_spacing=4.0):
    """
    Place a single BUY LIMIT or SELL LIMIT with TP.
    side: 'BUY' or 'SELL'
    """
    filling = get_filling_mode()
    if filling is None:
        log.warning("Cannot determine filling mode — skipping order")
        return False

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT

    sl = 0.0
    if EMERGENCY_SL_DIST is not None:
        if side == 'BUY':
            sl = round(price - EMERGENCY_SL_DIST, 2)
        else:
            sl = round(price + EMERGENCY_SL_DIST, 2)

    request = {
        'action':      mt5.TRADE_ACTION_PENDING,
        'symbol':      SYMBOL,
        'volume':      LOT_SIZE,
        'type':        order_type,
        'price':       price,
        'tp':          tp,
        'sl':          sl,
        'magic':       MAGIC_NUMBER,
        'comment':     f'GRID_{side}',
        'type_time':   mt5.ORDER_TIME_GTC,
        'type_filling': filling,
    }

    if not live_mode:
        log.info(f"  [DRY RUN] Would place {side} LIMIT @ {price:.2f}  TP={tp:.2f}  SL={sl:.2f}")
        return True

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.warning(f"  Order failed ({side} @ {price:.2f}): retcode={result.retcode} {result.comment}")
        return False

    log.info(f"  Placed {side} LIMIT @ {price:.2f}  TP={tp:.2f}  ticket={result.order}")
    return True


def cancel_all_grid_orders(live_mode):
    """Cancel all pending grid orders."""
    orders = get_grid_orders()
    if not orders:
        return
    for o in orders:
        if not live_mode:
            log.info(f"  [DRY RUN] Would cancel order ticket={o.ticket} @ {o.price_open:.2f}")
            continue
        result = mt5.order_send({
            'action': mt5.TRADE_ACTION_REMOVE,
            'order':  o.ticket,
        })
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"  Cancelled order ticket={o.ticket}")
        else:
            log.warning(f"  Cancel failed ticket={o.ticket}: {result.retcode}")


def close_all_grid_positions(live_mode):
    """Market-close all open grid positions."""
    positions = get_grid_positions()
    if not positions:
        return
    filling = get_filling_mode()
    if filling is None:
        log.error("Cannot close positions: filling mode unknown")
        return

    bid, ask = get_current_price()
    for p in positions:
        if p.type == mt5.POSITION_TYPE_BUY:
            close_type  = mt5.ORDER_TYPE_SELL
            close_price = bid
        else:
            close_type  = mt5.ORDER_TYPE_BUY
            close_price = ask

        if not live_mode:
            log.info(f"  [DRY RUN] Would close ticket={p.ticket} {('BUY' if p.type==0 else 'SELL')} @ {close_price:.2f}  PnL~{p.profit:.2f}")
            continue

        result = mt5.order_send({
            'action':      mt5.TRADE_ACTION_DEAL,
            'symbol':      SYMBOL,
            'volume':      p.volume,
            'type':        close_type,
            'position':    p.ticket,
            'price':       close_price,
            'magic':       MAGIC_NUMBER,
            'comment':     'GRID_FORCE_CLOSE',
            'type_time':   mt5.ORDER_TIME_GTC,
            'type_filling': filling,
        })
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"  Force-closed ticket={p.ticket}  PnL={p.profit:.2f}")
        else:
            log.warning(f"  Close failed ticket={p.ticket}: {result.retcode} {result.comment}")


def sync_grid(grid_levels, current_price, live_mode, grid_spacing=4.0):
    """
    Place limit orders at any grid level that doesn't already have one.
    Respects MAX_OPEN_POSITIONS across pending + open combined.
    """
    taken   = active_price_levels()
    n_open  = len(get_grid_positions())
    n_pend  = len(get_grid_orders())
    total   = n_open + n_pend

    placed = 0
    for level in grid_levels:
        if round(level, 2) in {round(t, 2) for t in taken}:
            continue
        if total + placed >= MAX_OPEN_POSITIONS:
            break

        if level < current_price:
            # BUY LIMIT below price: TP one grid up
            tp   = round(level + grid_spacing, 2)
            side = 'BUY'
        elif level > current_price:
            # SELL LIMIT above price: TP one grid down
            tp   = round(level - grid_spacing, 2)
            side = 'SELL'
        else:
            continue  # level == current price, skip

        if place_limit_order(side, level, tp, live_mode, grid_spacing):
            placed += 1

    if placed:
        log.info(f"Grid sync: placed {placed} new limit orders  (open={n_open} pending={n_pend})")


# ============================================================================
# BAR TIMING
# ============================================================================

def wait_for_next_5min():
    """Sleep until 3 seconds after the next 5-min mark."""
    now     = datetime.now(timezone.utc)
    seconds = now.minute * 60 + now.second + now.microsecond / 1e6
    rem     = seconds % 300          # 300 seconds = 5 minutes
    wait    = (300 - rem) + 3        # +3s buffer for bar to fully close
    log.info(f"Sleeping {wait:.1f}s until next 5-min bar...")
    time.sleep(wait)


def is_15min_bar_close():
    """Return True if current UTC minute is a 15-min boundary (0, 15, 30, 45)."""
    return datetime.now(timezone.utc).minute % 15 == 0


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='XAUUSD Regime Grid Trader')
    parser.add_argument('--live',     action='store_true', help='Enable live trading (default: dry run)')
    parser.add_argument('--mt5-path', type=str,            help='Path to MT5 terminal.exe')
    args = parser.parse_args()

    LIVE_MODE = args.live
    mode_str  = 'LIVE TRADING' if LIVE_MODE else 'DRY RUN'

    log.info('=' * 70)
    log.info(f'XAUUSD REGIME GRID TRADER  [{mode_str}]')
    log.info(f'Symbol: {SYMBOL}  Lot: {LOT_SIZE}  Grid: ATR*{ATR_MULTIPLIER} [{MIN_SPACING}-{MAX_SPACING}$] x {GRID_LEVELS} levels')
    log.info(f'Max positions: {MAX_OPEN_POSITIONS}  Magic: {MAGIC_NUMBER}')
    log.info(f'Regime thresholds: RANGING>={RANGING_THRESHOLD}  TRENDING<={TRENDING_THRESHOLD}')
    log.info('=' * 70)

    if not connect_mt5(args.mt5_path):
        sys.exit(1)

    current_regime   = None
    last_5min_time   = None
    grid_levels      = []
    grid_center      = None
    grid_spacing     = (MIN_SPACING + MAX_SPACING) / 2  # fallback until first 15-min close

    # Sync to next 5-min boundary before starting
    wait_for_next_5min()

    try:
        while True:
            now = datetime.now(timezone.utc)

            # ── Detect new 5-min bar ─────────────────────────────────────────
            bars_5m = get_bars(mt5.TIMEFRAME_M5, 3)
            if bars_5m is None:
                log.warning("Could not fetch 5-min bars — retrying in 10s")
                time.sleep(10)
                continue

            current_5min_time = bars_5m.index[-1]
            if current_5min_time == last_5min_time:
                # No new bar yet — wait
                time.sleep(2)
                continue

            last_5min_time = current_5min_time
            bid, ask = get_current_price()
            if bid is None:
                log.warning("Could not get price tick")
                wait_for_next_5min()
                continue

            # ── On 15-min bar close: recompute regime ────────────────────────
            if is_15min_bar_close():
                bars_15m = get_bars(mt5.TIMEFRAME_M15, WARMUP_BARS + 5)
                if bars_15m is not None and len(bars_15m) >= WARMUP_BARS:
                    new_regime, score, current_atr = compute_regime(bars_15m)

                    # Compute grid center from last RANGE_LOOKBACK closed bars
                    lookback = bars_15m.iloc[-RANGE_LOOKBACK:]
                    grid_center = (lookback['High'].max() + lookback['Low'].min()) / 2

                    # Dynamic ATR-based spacing — matches grid_backtest.py exactly
                    grid_spacing = float(np.clip(current_atr * ATR_MULTIPLIER, MIN_SPACING, MAX_SPACING))
                    grid_spacing = round(grid_spacing, 3)

                    grid_levels = build_grid_levels(grid_center, grid_spacing, GRID_LEVELS)

                    log.info(
                        f"[15M CLOSE] Regime={new_regime}  score={score:.3f}  ATR={current_atr:.2f}  "
                        f"spacing={grid_spacing:.3f}  center={grid_center:.2f}  levels={len(grid_levels)}"
                    )

                    # Regime changed
                    if new_regime != current_regime:
                        log.info(f"Regime CHANGED: {current_regime} -> {new_regime}")

                        if current_regime == 'RANGING':
                            # Leaving ranging — flatten everything
                            log.info("Leaving RANGING: cancelling orders + closing positions")
                            cancel_all_grid_orders(LIVE_MODE)
                            close_all_grid_positions(LIVE_MODE)

                        current_regime = new_regime
                else:
                    log.warning(f"Not enough 15-min history ({len(bars_15m) if bars_15m is not None else 0} bars) — staying {current_regime}")

            # ── Log current state ────────────────────────────────────────────
            n_pos  = len(get_grid_positions())
            n_ord  = len(get_grid_orders())
            log.info(
                f"[5M BAR {current_5min_time.strftime('%H:%M')}] "
                f"Price={bid:.2f}  Regime={current_regime}  "
                f"Positions={n_pos}  PendingOrders={n_ord}"
            )

            # ── Grid execution ───────────────────────────────────────────────
            if current_regime == 'RANGING':
                if grid_levels:
                    sync_grid(grid_levels, bid, LIVE_MODE, grid_spacing)
                else:
                    log.warning("RANGING but no grid levels computed yet — waiting for next 15-min close")

            elif current_regime in ('TRENDING', 'UNCERTAIN'):
                # Safety: if any grid orders/positions exist, close them
                # (handles edge case where regime flip was missed)
                if n_pos > 0 or n_ord > 0:
                    log.warning(f"Found {n_pos} positions + {n_ord} orders in {current_regime} regime — flattening")
                    cancel_all_grid_orders(LIVE_MODE)
                    close_all_grid_positions(LIVE_MODE)

            # ── Wait for next 5-min bar ──────────────────────────────────────
            wait_for_next_5min()

    except KeyboardInterrupt:
        log.info("Shutdown requested by user")
        log.info("Cancelling all pending orders and closing all positions...")
        cancel_all_grid_orders(LIVE_MODE)
        close_all_grid_positions(LIVE_MODE)
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        log.info("Emergency flatten on error...")
        try:
            cancel_all_grid_orders(LIVE_MODE)
            close_all_grid_positions(LIVE_MODE)
        except Exception as e2:
            log.error(f"Emergency flatten failed: {e2}")
    finally:
        mt5.shutdown()
        log.info("MT5 disconnected")


if __name__ == '__main__':
    main()
