"""
trendline_strategy.py
=====================
Mechanical Trendline Breakout Strategy

Reproduces the YouTube strategy mechanically:
  - Detect swing highs and lows (pivot points)
  - Fit valid trendlines requiring >= MIN_TOUCHES touch points
  - Enter LONG on break of downtrend line (price closes above it)
  - Enter SHORT on break of uptrend line (price closes below it)
  - Trail the position using the newly forming trendline as a dynamic stop
  - Hard ATR stop as safety net

No VectorBT used here — trendline trailing stops require dynamic
geometry (swing points form bar-by-bar), which can't be pre-vectorised.

Output
------
  - Console: performance report
  - CSV: data/processed/trendline_strategy_trades.csv
"""

import os
import numpy as np
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CSV_PATH       = 'data/raw/XAUUSD15.csv'     # tab-sep MT4/MT5 export (H1 XAUUSD)
DATE_FROM      = '2025-01-01'
DATE_TO        = '2025-12-31'

SWING_LOOKBACK = 10       # bars on each side to confirm a pivot (10 bars = 2.5h on M15)
MIN_TOUCHES    = 3        # minimum touches to validate a trendline (incl. 2 anchors)
TOUCH_TOL_PCT  = 0.0015   # 0.15% tolerance (tighter — on $4000 XAUUSD this is ±$6)
BREAKOUT_BARS  = 2        # consecutive closes beyond the line needed for entry confirmation
MIN_SLOPE_PER_BAR   = 0.05   # min absolute slope ($/bar) to reject flat consolidation
MIN_ANCHOR_SPAN     = 30     # min bars between anchor points (~7.5h on M15)
MIN_TOUCH_GAP       = 8      # min bars between consecutive touches (no clusters)
MAX_PENETRATION_PCT = 0.001  # max penetration allowed (0.1% of line value) before line is invalid

ATR_PERIOD     = 14
ATR_STOP_MULT  = 2.5      # hard stop = entry_price ± ATR * multiplier

INIT_CASH      = 100_000  # USD, starting equity
LOT_VALUE      = 1.0      # value of 1 unit move in $  (adjust per instrument)
TRADE_SIZE     = 1.0      # units per trade (e.g. 1 oz XAUUSD)
COMMISSION     = 0.50     # $ per trade (entry + exit round-trip included separately)

MAX_LOOKBACK_SWINGS = 30  # max historic swing points to scan when fitting trendlines
COOLDOWN_BARS      = 3    # bars to wait after exit before allowing re-entry
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Load MT4/MT5 tab-separated OHLCV CSV into a clean DataFrame.

    Handles both files with a header row (<DATE> <TIME> ...) and without.
    """
    abs_path = os.path.join(os.path.dirname(__file__), '..', path) if not os.path.isabs(path) else path
    if not os.path.exists(abs_path):
        abs_path = path  # fallback: relative to cwd

    col_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']

    # Peek at first line to detect whether a header row exists
    with open(abs_path, 'r') as f:
        first_line = f.readline().strip()
    has_header = first_line.startswith('<') or first_line.upper().startswith('DATE')

    df = pd.read_csv(
        abs_path,
        sep='\t',
        names=col_names,
        skiprows=1 if has_header else 0,
        dtype={'Date': str, 'Time': str}
    )

    # Some MT5 exports use a combined datetime in the first column
    if df['Time'].str.contains(':').any():
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    else:
        df['Datetime'] = pd.to_datetime(df['Date'], format='%Y.%m.%d')

    df = df.set_index('Datetime').drop(columns=['Date', 'Time', 'TickVol', 'Vol', 'Spread'], errors='ignore')
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.loc[date_from:date_to].dropna(subset=['Open', 'High', 'Low', 'Close'])
    print(f"[Data] Loaded {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ATR CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — INCREMENTAL SWING POINT DETECTOR (live-compatible)
# ══════════════════════════════════════════════════════════════════════════════

def try_confirm_swing(current_bar: int, high: np.ndarray, low: np.ndarray,
                      lookback: int = SWING_LOOKBACK):
    """
    Check whether the bar at `current_bar - lookback` can now be confirmed
    as a swing high and/or swing low, using ONLY data available at current_bar.

    Returns (is_swing_high: bool, is_swing_low: bool, candidate_bar: int).
    """
    candidate = current_bar - lookback
    if candidate < lookback:
        return False, False, candidate

    window_start = candidate - lookback
    window_end   = current_bar + 1

    is_sh = high[candidate] >= high[window_start:window_end].max()
    is_sl = low[candidate]  <= low[window_start:window_end].min()

    return is_sh, is_sl, candidate


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — CLEAN TRENDLINE FITTER & VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def _fit_line(x1: int, y1: float, x2: int, y2: float):
    """Return (slope, intercept) for a line through two points."""
    if x2 == x1:
        return 0.0, y1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _line_value(slope: float, intercept: float, x: int) -> float:
    return slope * x + intercept


def _validate_trendline(
    slope: float,
    intercept: float,
    anchor_x1: int,
    anchor_x2: int,
    swing_idx: np.ndarray,
    swing_prices: np.ndarray,
    close: np.ndarray,
    tol_pct: float,
    min_touch_gap: int,
    max_penetration_pct: float,
    direction: str,   # 'up' or 'dn'
):
    """
    Validate a candidate trendline for cleanliness:

    1. Count swing-point touches that are within tolerance% of the line,
       but only if they are >= min_touch_gap bars from the previous touch.
    2. Non-penetration check: between anchor_x1 and anchor_x2, no CLOSE should
       cross significantly through the line on the wrong side.
       - Uptrend (support): close must not drop far below the line
       - Downtrend (resistance): close must not rise far above the line

    Returns (total_touches: int, is_clean: bool).
    """
    # --- Step 1: count spaced-out touches ---
    touch_bars = [anchor_x1, anchor_x2]  # anchors always count
    for idx, price in zip(swing_idx, swing_prices):
        if idx == anchor_x1 or idx == anchor_x2:
            continue
        # Only count touches between anchors
        if idx < anchor_x1 or idx > anchor_x2:
            continue
        line_val = _line_value(slope, intercept, idx)
        if line_val <= 0:
            continue
        if abs(price - line_val) / line_val <= tol_pct:
            # Check minimum gap from all existing touches
            too_close = any(abs(idx - tb) < min_touch_gap for tb in touch_bars)
            if not too_close:
                touch_bars.append(idx)

    total_touches = len(touch_bars)

    # --- Step 2: non-penetration check ---
    # Check all closes between the anchors — price should respect the line
    is_clean = True
    for bar in range(anchor_x1, anchor_x2 + 1):
        line_val = _line_value(slope, intercept, bar)
        if line_val <= 0:
            continue
        if direction == 'up':
            # Uptrend = support line. Close should NOT be far below.
            penetration = (line_val - close[bar]) / line_val
            if penetration > max_penetration_pct:
                is_clean = False
                break
        else:
            # Downtrend = resistance line. Close should NOT be far above.
            penetration = (close[bar] - line_val) / line_val
            if penetration > max_penetration_pct:
                is_clean = False
                break

    return total_touches, is_clean


def get_active_trendlines(
    current_bar: int,
    sh_idx: np.ndarray,
    sl_idx: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tol_pct: float = TOUCH_TOL_PCT,
    min_touches: int = MIN_TOUCHES,
    max_lookback: int = MAX_LOOKBACK_SWINGS,
):
    """
    For a given bar, find the best CLEAN uptrend and downtrend lines from
    confirmed swing points.

    A clean trendline must:
      - Have anchors >= MIN_ANCHOR_SPAN bars apart
      - Have absolute slope >= MIN_SLOPE_PER_BAR
      - Have >= MIN_TOUCHES spaced-out touch points
      - Pass the non-penetration check (price respects the line)

    Returns:
        (up_valid, up_slope, up_intercept, dn_valid, dn_slope, dn_intercept)
    """
    # --- Uptrend line: through recent swing LOWS (support) ---
    mask_sl = sl_idx < current_bar
    recent_sl = sl_idx[mask_sl][-max_lookback:]
    up_valid, up_slope, up_intercept, up_touches = False, 0.0, 0.0, 0

    if len(recent_sl) >= 2:
        for a_idx in range(len(recent_sl) - 1, 0, -1):
            x2, y2 = recent_sl[a_idx], low[recent_sl[a_idx]]
            for b_idx in range(a_idx - 1, -1, -1):
                x1, y1 = recent_sl[b_idx], low[recent_sl[b_idx]]
                if x2 - x1 < MIN_ANCHOR_SPAN:
                    continue
                slope, intercept = _fit_line(x1, y1, x2, y2)
                if slope < MIN_SLOPE_PER_BAR:
                    continue
                touches, clean = _validate_trendline(
                    slope, intercept, x1, x2,
                    recent_sl, low[recent_sl], close,
                    tol_pct, MIN_TOUCH_GAP, MAX_PENETRATION_PCT, 'up'
                )
                if clean and touches >= min_touches and touches > up_touches:
                    up_valid, up_slope, up_intercept, up_touches = True, slope, intercept, touches
            if up_valid:
                break

    # --- Downtrend line: through recent swing HIGHS (resistance) ---
    mask_sh = sh_idx < current_bar
    recent_sh = sh_idx[mask_sh][-max_lookback:]
    dn_valid, dn_slope, dn_intercept, dn_touches = False, 0.0, 0.0, 0

    if len(recent_sh) >= 2:
        for a_idx in range(len(recent_sh) - 1, 0, -1):
            x2, y2 = recent_sh[a_idx], high[recent_sh[a_idx]]
            for b_idx in range(a_idx - 1, -1, -1):
                x1, y1 = recent_sh[b_idx], high[recent_sh[b_idx]]
                if x2 - x1 < MIN_ANCHOR_SPAN:
                    continue
                slope, intercept = _fit_line(x1, y1, x2, y2)
                if slope > -MIN_SLOPE_PER_BAR:
                    continue
                touches, clean = _validate_trendline(
                    slope, intercept, x1, x2,
                    recent_sh, high[recent_sh], close,
                    tol_pct, MIN_TOUCH_GAP, MAX_PENETRATION_PCT, 'dn'
                )
                if clean and touches >= min_touches and touches > dn_touches:
                    dn_valid, dn_slope, dn_intercept, dn_touches = True, slope, intercept, touches
            if dn_valid:
                break

    return up_valid, up_slope, up_intercept, dn_valid, dn_slope, dn_intercept


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — EVENT-DRIVEN BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Bar-by-bar event loop.

    State machine:
      FLAT  → watch for breakout entry
      LONG  → trail via dynamically updated uptrend line; hard ATR stop
      SHORT → trail via dynamically updated downtrend line; hard ATR stop

    Key fixes vs v1:
      - Entry/exit execute at NEXT bar's open (no look-ahead bias)
      - Trailing trendline is re-computed every bar (dynamic, not frozen)
      - Cooldown after exit prevents same-bar re-entry
      - Trendline deduplication prevents re-trading the same breakout
      - BREAKOUT_BARS=2 requires 2 consecutive closes for confirmation

    Returns (trades_df, equity_curve).
    """
    open_arr  = df['Open'].values
    high_arr  = df['High'].values
    low_arr   = df['Low'].values
    close_arr = df['Close'].values
    atr_arr   = compute_atr(df, ATR_PERIOD).values
    dates     = df.index
    n_bars    = len(df)

    # Swing points discovered incrementally (live-compatible)
    sh_list = []   # swing high bar indices
    sl_list = []   # swing low bar indices
    sh_idx  = np.array([], dtype=int)
    sl_idx  = np.array([], dtype=int)

    warmup = SWING_LOOKBACK * 2 + ATR_PERIOD + 5
    print("[Swings] Using incremental live-compatible detection...")

    trades       = []
    equity       = INIT_CASH
    equity_curve = np.full(n_bars, np.nan)

    # Position state
    position    = 0        # 0 = flat, 1 = long, -1 = short
    entry_bar   = 0
    entry_price = 0.0
    hard_stop   = 0.0
    last_exit_bar = -999   # for cooldown enforcement

    # Trendline deduplication: store (slope, intercept) of last traded breakout
    last_traded_line = (None, None)

    # Breakout confirmation state
    confirm_count     = 0
    pending_dir       = 0
    pending_line_key  = (None, None)  # (slope, intercept) of the line being confirmed

    # Pending entry (signal on bar i, execute on bar i+1)
    pending_entry     = 0   # 0 = none, 1 = long, -1 = short
    pending_entry_atr = 0.0
    pending_trail_sl  = 0.0
    pending_trail_ic  = 0.0

    for i in range(warmup, n_bars):
        equity_curve[i] = equity
        price = close_arr[i]
        atr   = atr_arr[i]

        if np.isnan(atr) or atr == 0:
            continue

        # ── INCREMENTAL SWING DETECTION ──────────────────────────────────
        # At bar i, check if bar (i - lookback) is a confirmed swing
        is_sh, is_sl, cand = try_confirm_swing(i, high_arr, low_arr, SWING_LOOKBACK)
        if is_sh:
            sh_list.append(cand)
            sh_idx = np.array(sh_list, dtype=int)
        if is_sl:
            sl_list.append(cand)
            sl_idx = np.array(sl_list, dtype=int)

        # ── EXECUTE PENDING ENTRY from previous bar's signal ──────────────
        if pending_entry != 0 and position == 0:
            entry_bar   = i
            entry_price = open_arr[i]  # fill at this bar's open
            position    = pending_entry
            if position == 1:
                hard_stop = entry_price - ATR_STOP_MULT * pending_entry_atr
            else:
                hard_stop = entry_price + ATR_STOP_MULT * pending_entry_atr
            pending_entry = 0

        # Compute current trendlines
        up_v, up_sl, up_ic, dn_v, dn_sl, dn_ic = get_active_trendlines(
            i, sh_idx, sl_idx, high_arr, low_arr, close_arr
        )

        up_val = _line_value(up_sl, up_ic, i) if up_v else np.nan
        dn_val = _line_value(dn_sl, dn_ic, i) if dn_v else np.nan

        # ── EXIT LOGIC ────────────────────────────────────────────────────
        if position == 1:  # LONG
            # Dynamically update trailing stop from current uptrend line
            if up_v:
                trail_val = _line_value(up_sl, up_ic, i)
            else:
                trail_val = hard_stop  # fallback to hard stop if no uptrend line

            hit_trail    = price < trail_val
            hit_hard     = low_arr[i] <= hard_stop
            new_dn_break = dn_v and price < dn_val

            if hit_trail or hit_hard or new_dn_break:
                # Signal exit — execute at NEXT bar's open
                if i + 1 < n_bars:
                    exit_price = open_arr[i + 1]
                    exit_bar   = i + 1
                else:
                    exit_price = close_arr[i]
                    exit_bar   = i

                reason = 'hard_stop' if hit_hard else ('new_dn_break' if new_dn_break else 'trail')
                pnl    = (exit_price - entry_price) * TRADE_SIZE * LOT_VALUE - COMMISSION
                equity += pnl
                trades.append({
                    'entry_date':  dates[entry_bar],
                    'exit_date':   dates[exit_bar],
                    'direction':   'LONG',
                    'entry_price': round(entry_price, 5),
                    'exit_price':  round(exit_price, 5),
                    'bars_held':   exit_bar - entry_bar,
                    'pnl':         round(pnl, 2),
                    'exit_reason': reason,
                    'equity':      round(equity, 2),
                })
                position      = 0
                last_exit_bar = i
                confirm_count = 0
                pending_entry = 0
                continue  # skip entry logic on exit bar

        elif position == -1:  # SHORT
            # Dynamically update trailing stop from current downtrend line
            if dn_v:
                trail_val = _line_value(dn_sl, dn_ic, i)
            else:
                trail_val = hard_stop

            hit_trail    = price > trail_val
            hit_hard     = high_arr[i] >= hard_stop
            new_up_break = up_v and price > up_val

            if hit_trail or hit_hard or new_up_break:
                if i + 1 < n_bars:
                    exit_price = open_arr[i + 1]
                    exit_bar   = i + 1
                else:
                    exit_price = close_arr[i]
                    exit_bar   = i

                reason = 'hard_stop' if hit_hard else ('new_up_break' if new_up_break else 'trail')
                pnl    = (entry_price - exit_price) * TRADE_SIZE * LOT_VALUE - COMMISSION
                equity += pnl
                trades.append({
                    'entry_date':  dates[entry_bar],
                    'exit_date':   dates[exit_bar],
                    'direction':   'SHORT',
                    'entry_price': round(entry_price, 5),
                    'exit_price':  round(exit_price, 5),
                    'bars_held':   exit_bar - entry_bar,
                    'pnl':         round(pnl, 2),
                    'exit_reason': reason,
                    'equity':      round(equity, 2),
                })
                position      = 0
                last_exit_bar = i
                confirm_count = 0
                pending_entry = 0
                continue  # skip entry logic on exit bar

        # ── ENTRY LOGIC ───────────────────────────────────────────────────
        if position == 0 and pending_entry == 0:
            # Enforce cooldown after last exit
            if i - last_exit_bar < COOLDOWN_BARS:
                continue

            # Break of downtrend line → LONG signal
            if dn_v and price > dn_val:
                line_key = (round(dn_sl, 10), round(dn_ic, 2))
                # Dedup: don't re-trade the same trendline
                if line_key == last_traded_line:
                    pass
                elif confirm_count > 0 and pending_dir == 1 and pending_line_key == line_key:
                    confirm_count += 1
                else:
                    confirm_count    = 1
                    pending_dir      = 1
                    pending_line_key = line_key

                if confirm_count >= BREAKOUT_BARS:
                    pending_entry     = 1
                    pending_entry_atr = atr
                    last_traded_line  = line_key
                    confirm_count     = 0

            # Break of uptrend line → SHORT signal
            elif up_v and price < up_val:
                line_key = (round(up_sl, 10), round(up_ic, 2))
                if line_key == last_traded_line:
                    pass
                elif confirm_count > 0 and pending_dir == -1 and pending_line_key == line_key:
                    confirm_count += 1
                else:
                    confirm_count    = 1
                    pending_dir      = -1
                    pending_line_key = line_key

                if confirm_count >= BREAKOUT_BARS:
                    pending_entry     = -1
                    pending_entry_atr = atr
                    last_traded_line  = line_key
                    confirm_count     = 0
            else:
                # Price back inside → reset confirmation
                confirm_count = 0

    # Close any open position at end of data
    if position != 0:
        exit_price = close_arr[-1]
        if position == 1:
            pnl = (exit_price - entry_price) * TRADE_SIZE * LOT_VALUE - COMMISSION
        else:
            pnl = (entry_price - exit_price) * TRADE_SIZE * LOT_VALUE - COMMISSION
        equity += pnl
        trades.append({
            'entry_date':  dates[entry_bar],
            'exit_date':   dates[-1],
            'direction':   'LONG' if position == 1 else 'SHORT',
            'entry_price': round(entry_price, 5),
            'exit_price':  round(exit_price, 5),
            'bars_held':   n_bars - 1 - entry_bar,
            'pnl':         round(pnl, 2),
            'exit_reason': 'eod',
            'equity':      round(equity, 2),
        })

    trades_df     = pd.DataFrame(trades)
    equity_series = pd.Series(equity_curve, index=df.index)

    return trades_df, equity_series


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — PERFORMANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(trades_df: pd.DataFrame, equity: pd.Series):
    """Print a pretty performance summary to console."""
    if trades_df.empty:
        print("\n[!] No trades generated. Check data, SWING_LOOKBACK, or MIN_TOUCHES settings.")
        return

    pnl     = trades_df['pnl']
    winners = pnl[pnl > 0]
    losers  = pnl[pnl <= 0]

    win_rate  = len(winners) / len(pnl) * 100 if len(pnl) > 0 else 0
    pf_num    = winners.sum()
    pf_den    = abs(losers.sum())
    pf        = round(pf_num / pf_den, 2) if pf_den != 0 else float('inf')
    avg_win   = winners.mean() if len(winners) > 0 else 0
    avg_loss  = losers.mean()  if len(losers)  > 0 else 0
    avg_rr    = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Max drawdown from equity curve
    eq = equity.dropna()
    running_max = eq.cummax()
    dd = eq - running_max
    max_dd_pct = (dd / running_max * 100).min()
    max_dd_abs = dd.min()

    total_pnl   = pnl.sum()
    total_return = total_pnl / INIT_CASH * 100

    longs  = trades_df[trades_df['direction'] == 'LONG']
    shorts = trades_df[trades_df['direction'] == 'SHORT']

    exit_reasons = trades_df['exit_reason'].value_counts()

    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  MECHANICAL TRENDLINE STRATEGY — RESULTS")
    print(f"{sep}")
    print(f"  Total Trades   : {len(pnl)}")
    print(f"  Long / Short   : {len(longs)} / {len(shorts)}")
    print(f"  Win Rate       : {win_rate:.1f}%")
    print(f"  Profit Factor  : {pf}")
    print(f"  Avg Win  ($)   : {avg_win:.2f}")
    print(f"  Avg Loss ($)   : {avg_loss:.2f}")
    print(f"  Avg R:R        : {avg_rr:.2f}")
    print(f"  Total PnL ($)  : {total_pnl:.2f}")
    print(f"  Total Return   : {total_return:.2f}%")
    print(f"  Max Drawdown   : {max_dd_abs:.2f}$ ({max_dd_pct:.2f}%)")
    print(f"  Avg Bars Held  : {trades_df['bars_held'].mean():.1f}")
    print(f"\n  Exit Reasons:")
    for reason, cnt in exit_reasons.items():
        print(f"    {reason:<18}: {cnt}")
    print(sep)

    # Show top 10 trades
    top10 = trades_df.nlargest(10, 'pnl')[['entry_date', 'exit_date', 'direction', 'entry_price', 'exit_price', 'bars_held', 'pnl', 'exit_reason']]
    print("\n  Top 10 Trades by PnL:")
    print(top10.to_string(index=False))
    print()

    # Show worst 10 trades
    bot10 = trades_df.nsmallest(10, 'pnl')[['entry_date', 'exit_date', 'direction', 'entry_price', 'exit_price', 'bars_held', 'pnl', 'exit_reason']]
    print("  Worst 10 Trades by PnL:")
    print(bot10.to_string(index=False))
    print(sep)


def save_output(trades_df: pd.DataFrame):
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'trendline_strategy_trades.csv')
    trades_df.to_csv(out_path, index=False)
    print(f"[Output] Trades saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 55)
    print("  Mechanical Trendline Breakout Strategy")
    print("=" * 55)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = load_data(CSV_PATH, DATE_FROM, DATE_TO)

    # ── 2. Run backtest ──────────────────────────────────────────────────────
    print("[Backtest] Running event-driven simulation...")
    trades_df, equity = run_backtest(df)
    print(f"[Backtest] Completed. {len(trades_df)} trades generated.")

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print_report(trades_df, equity)

    # ── 4. Save ───────────────────────────────────────────────────────────────
    if not trades_df.empty:
        save_output(trades_df)
