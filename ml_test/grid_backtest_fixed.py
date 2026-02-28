import pandas as pd
import numpy as np
import os

# ============================================================================
# GRID BACKTEST — XAUUSD 5-minute
#
# Architecture:
#   - 15-min ranging_regime.py output drives when the grid is ON/OFF
#   - 5-min bars execute the grid trades
#
# Grid logic:
#   1. When 15-min regime = RANGING: grid is ACTIVE
#   2. Define grid range from last RANGE_LOOKBACK 15-min bars (high/low)
#   3. Place virtual buy orders below price, sell orders above price
#      - Buy at each grid level below: TP = entry + GRID_SPACING
#      - Sell at each grid level above: TP = entry - GRID_SPACING
#   4. When 15-min regime flips to TRENDING/UNCERTAIN: close ALL open
#      positions at current price (force close, may be loss)
#   5. Max MAX_OPEN_POSITIONS positions at any time (capital limit)
#
# Key assumption: $1 per point movement (adjust for lot size in live trading)
# ============================================================================

DATA_5M     = 'data/raw/XAUUSD5.csv'
REGIMES_15M = 'data/processed/XAUUSD15_regimes.csv'
OUT_PATH    = 'data/processed/grid_backtest_results.csv'

# Grid parameters — tune these
GRID_SPACING       = 2.0    # $ between grid levels (gold)
GRID_LEVELS        = 6      # number of levels each side of center
RANGE_LOOKBACK     = 20     # 15-min bars to define initial grid range (5 hours)
MAX_OPEN_POSITIONS = 6      # max simultaneous open positions
PROFIT_PER_POINT   = 1.0    # $ per point move (scale to your lot size)

# Out-of-sample window — set to None to run on all data
BACKTEST_START = None   # None = run full dataset
BACKTEST_END   = None   # None = run full dataset


# ============================================================================
# LOAD DATA
# ============================================================================
def load_ohlc(path):
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip('<>') for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df = df.set_index('Datetime').sort_index()
    df = df.rename(columns={
        'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
        'CLOSE': 'Close', 'TICKVOL': 'TickVol'
    })
    return df[['Open', 'High', 'Low', 'Close', 'TickVol']]


def load_regimes(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[['regime', 'ranging_score_biased', 'session', 'High', 'Low']]


# ============================================================================
# ALIGN 5-MIN BAR TO CURRENT 15-MIN REGIME
# Each 5-min bar looks up the 15-min bar that contains it
# ============================================================================
def align_regime(df5, df15):
    """For each 5-min bar, find the regime of the PREVIOUS closed 15-min bar.

    The 15-min bar timestamp is the bar's OPEN time. Its regime is computed
    using data through its CLOSE. So 5-min bars inside that 15-min bar must
    use the PREVIOUS bar's regime (shift(1)) to avoid lookahead bias.
    """
    combined_idx = df5.index.union(df15.index)
    regimes_reindexed  = df15['regime'].shift(1).reindex(combined_idx).ffill()
    scores_reindexed   = df15['ranging_score_biased'].shift(1).reindex(combined_idx).ffill()
    sessions_reindexed = df15['session'].shift(1).reindex(combined_idx).ffill()

    df5['regime']        = regimes_reindexed.reindex(df5.index)
    df5['regime_score']  = scores_reindexed.reindex(df5.index)
    df5['session']       = sessions_reindexed.reindex(df5.index)
    return df5


# ============================================================================
# ROLLING GRID CENTER — uses recent 15-min range
# ============================================================================
def build_grid_levels(center, spacing, n_levels):
    """Return sorted list of grid levels centered around `center`."""
    levels = []
    for i in range(-n_levels, n_levels + 1):
        levels.append(round(center + i * spacing, 3))
    return sorted(levels)


# ============================================================================
# POSITION TRACKER
# ============================================================================
class Position:
    def __init__(self, side, entry_price, tp_price, open_time):
        self.side        = side         # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.tp_price    = tp_price
        self.open_time   = open_time
        self.close_time  = None
        self.close_price = None
        self.pnl         = None
        self.outcome     = None         # 'TP' or 'FORCE_CLOSE'

    def check_tp(self, high, low):
        """Return True if TP was hit this bar."""
        if self.side == 'LONG'  and high >= self.tp_price:
            return True
        if self.side == 'SHORT' and low  <= self.tp_price:
            return True
        return False

    def close(self, price, time, outcome):
        self.close_price = price
        self.close_time  = time
        self.outcome     = outcome
        if self.side == 'LONG':
            self.pnl = (price - self.entry_price) * PROFIT_PER_POINT
        else:
            self.pnl = (self.entry_price - price) * PROFIT_PER_POINT


# ============================================================================
# MAIN BACKTEST
# ============================================================================
print('=' * 70)
print('GRID BACKTEST — XAUUSD 5-minute')
print(f'Grid spacing: ${GRID_SPACING}  |  Levels: {GRID_LEVELS} each side')
print(f'Max open positions: {MAX_OPEN_POSITIONS}')
print('=' * 70)

print(f'\nLoading data...')
df5  = load_ohlc(DATA_5M)
df15 = load_regimes(REGIMES_15M)
print(f'5-min bars:  {len(df5):,}  ({df5.index[0]} to {df5.index[-1]})')
print(f'15-min bars: {len(df15):,} ({df15.index[0]} to {df15.index[-1]})')

# Align regimes (uses full 15-min history for regime context, then filters)
df5 = align_regime(df5, df15)
df5 = df5.dropna(subset=['regime'])

# Apply OOS window AFTER alignment so rolling context is intact
if BACKTEST_START:
    df5 = df5[df5.index >= BACKTEST_START]
if BACKTEST_END:
    df5 = df5[df5.index < BACKTEST_END]
print(f'OOS window:  {df5.index[0]} to {df5.index[-1]}  ({len(df5):,} bars)')

# Compute rolling 15-min grid center (high+low midpoint over RANGE_LOOKBACK bars)
# We use the 15-min data to define the grid center and attach it to 5-min
df15['grid_center'] = (
    df15['High'].rolling(RANGE_LOOKBACK).max() +
    df15['Low'].rolling(RANGE_LOOKBACK).min()
) / 2
# Align to 5-min — shift(1) for same reason as regime: the 15-min bar's
# grid_center includes that bar's own High/Low (closes at bar end), so
# 5-min bars inside that period must use the PREVIOUS bar's grid_center.
combined_idx = df5.index.union(df15.index)
center_reindexed = df15['grid_center'].shift(1).reindex(combined_idx).ffill()
df5['grid_center'] = center_reindexed.reindex(df5.index)
df5 = df5.dropna(subset=['grid_center'])

print(f'Active bars after alignment: {len(df5):,}')

# ── Simulation ──────────────────────────────────────────────────────────────
# Mirrors live trader exactly:
#   - Orders placed as PENDING limit orders after bar close
#   - Fills happen in SUBSEQUENT bars when bar range touches the level
#   - TP handled automatically (same bar as fill if range covers both)
#   - Regime flip: cancel pending orders + force-close open positions

open_positions = []    # list of Position objects (filled)
pending_orders = []    # list of dicts: {'side', 'level', 'tp'}
closed_trades  = []    # list of Position objects (closed)

# active_levels: levels that already have a pending order OR open position
active_levels  = set()

current_regime = None
equity         = 0.0
equity_curve   = []

for dt, bar in df5.iterrows():
    regime      = bar['regime']
    close_price = bar['Close']
    high        = bar['High']
    low         = bar['Low']
    grid_center = bar['grid_center']

    # ── Step 1: Regime change ────────────────────────────────────────────────
    if regime != current_regime:
        if current_regime == 'RANGING':
            # Cancel all pending limit orders
            pending_orders = []
            # Force-close all open positions
            for pos in open_positions:
                pos.close(close_price, dt, 'FORCE_CLOSE')
                equity += pos.pnl
                closed_trades.append(pos)
            open_positions = []
            active_levels  = set()
        current_regime = regime

    # ── Step 2: Skip non-RANGING bars ────────────────────────────────────────
    if regime != 'RANGING':
        equity_curve.append({'Datetime': dt, 'equity': equity, 'regime': regime,
                             'open_pos': 0})
        continue

    # ── Step 3: Fill pending limit orders hit by this bar's range ────────────
    still_pending = []
    for order in pending_orders:
        if order['side'] == 'BUY' and low <= order['level']:
            if high >= order['tp']:
                # Fill and TP in same bar
                pos = Position('LONG', order['level'], order['tp'], dt)
                pos.close(order['tp'], dt, 'TP')
                equity += pos.pnl
                closed_trades.append(pos)
                active_levels.discard(order['level'])
            else:
                pos = Position('LONG', order['level'], order['tp'], dt)
                open_positions.append(pos)
        elif order['side'] == 'SELL' and high >= order['level']:
            if low <= order['tp']:
                # Fill and TP in same bar
                pos = Position('SHORT', order['level'], order['tp'], dt)
                pos.close(order['tp'], dt, 'TP')
                equity += pos.pnl
                closed_trades.append(pos)
                active_levels.discard(order['level'])
            else:
                pos = Position('SHORT', order['level'], order['tp'], dt)
                open_positions.append(pos)
        else:
            still_pending.append(order)
    pending_orders = still_pending

    # ── Step 4: Check TPs on open positions ──────────────────────────────────
    still_open = []
    for pos in open_positions:
        if pos.check_tp(high, low):
            pos.close(pos.tp_price, dt, 'TP')
            equity += pos.pnl
            closed_trades.append(pos)
            active_levels.discard(pos.entry_price)
        else:
            still_open.append(pos)
    open_positions = still_open

    # ── Step 5: Sync grid — mirrors live trader sync_grid() ──────────────────
    # Rebuild levels from current grid_center (updates at each 15-min boundary)
    grid_levels = build_grid_levels(grid_center, GRID_SPACING, GRID_LEVELS)
    # Place pending orders at any level not already covered, up to position cap
    n_total = len(open_positions) + len(pending_orders)
    for level in grid_levels:
        if level in active_levels:
            continue
        if n_total >= MAX_OPEN_POSITIONS:
            break
        if level < close_price:
            pending_orders.append({'side': 'BUY',  'level': level,
                                   'tp': round(level + GRID_SPACING, 3)})
            active_levels.add(level)
            n_total += 1
        elif level > close_price:
            pending_orders.append({'side': 'SELL', 'level': level,
                                   'tp': round(level - GRID_SPACING, 3)})
            active_levels.add(level)
            n_total += 1

    equity_curve.append({
        'Datetime':  dt,
        'equity':    equity,
        'regime':    regime,
        'open_pos':  len(open_positions),
    })

# Cancel remaining pending orders, force-close remaining open positions
pending_orders = []
if open_positions:
    last_bar = df5.iloc[-1]
    for pos in open_positions:
        pos.close(last_bar['Close'], df5.index[-1], 'FORCE_CLOSE')
        equity += pos.pnl
        closed_trades.append(pos)

# ============================================================================
# RESULTS
# ============================================================================
print(f'\n{"=" * 70}')
print(f'BACKTEST RESULTS')
print(f'{"=" * 70}')

trades_df = pd.DataFrame([{
    'open_time':   t.open_time,
    'close_time':  t.close_time,
    'side':        t.side,
    'entry':       t.entry_price,
    'tp':          t.tp_price,
    'close_price': t.close_price,
    'pnl':         t.pnl,
    'outcome':     t.outcome,
} for t in closed_trades])

if len(trades_df) == 0:
    print('No trades executed.')
else:
    tp_trades    = trades_df[trades_df['outcome'] == 'TP']
    force_trades = trades_df[trades_df['outcome'] == 'FORCE_CLOSE']

    total_trades = len(trades_df)
    tp_count     = len(tp_trades)
    force_count  = len(force_trades)
    win_rate     = tp_count / total_trades * 100

    total_pnl    = trades_df['pnl'].sum()
    avg_pnl      = trades_df['pnl'].mean()
    avg_tp_pnl   = tp_trades['pnl'].mean() if len(tp_trades) > 0 else 0
    avg_fc_pnl   = force_trades['pnl'].mean() if len(force_trades) > 0 else 0

    wins  = trades_df[trades_df['pnl'] > 0]
    loses = trades_df[trades_df['pnl'] < 0]
    profit_factor = wins['pnl'].sum() / abs(loses['pnl'].sum()) if len(loses) > 0 else float('inf')

    # Max drawdown from equity curve
    eq_series = pd.Series([e['equity'] for e in equity_curve])
    rolling_max = eq_series.cummax()
    drawdown    = eq_series - rolling_max
    max_dd      = drawdown.min()

    print(f'\n--- OVERALL ---')
    print(f'  Total trades:      {total_trades:>6,}')
    print(f'  TP hits:           {tp_count:>6,}  ({tp_count/total_trades*100:.1f}%)')
    print(f'  Force closes:      {force_count:>6,}  ({force_count/total_trades*100:.1f}%)')
    print(f'  Win rate:          {win_rate:>6.1f}%')
    print(f'  Total P&L:         ${total_pnl:>10.2f}')
    print(f'  Avg P&L/trade:     ${avg_pnl:>10.2f}')
    print(f'  Avg TP P&L:        ${avg_tp_pnl:>10.2f}')
    print(f'  Avg force-close:   ${avg_fc_pnl:>10.2f}')
    print(f'  Profit factor:     {profit_factor:>10.2f}')
    print(f'  Max drawdown:      ${max_dd:>10.2f}')

    print(f'\n--- BY DIRECTION ---')
    for side in ['LONG', 'SHORT']:
        sub = trades_df[trades_df['side'] == side]
        if len(sub) == 0:
            continue
        wr = (sub['pnl'] > 0).mean() * 100
        print(f'  {side:<6}  {len(sub):>5} trades  WR: {wr:>5.1f}%  P&L: ${sub["pnl"].sum():>8.2f}')

    print(f'\n--- TP vs FORCE-CLOSE BREAKDOWN ---')
    print(f'  TP trades:     {len(tp_trades):>5}  Avg P&L: ${avg_tp_pnl:>7.2f}  Total: ${tp_trades["pnl"].sum():>9.2f}')
    print(f'  Force closes:  {len(force_trades):>5}  Avg P&L: ${avg_fc_pnl:>7.2f}  Total: ${force_trades["pnl"].sum():>9.2f}')

    # Monthly breakdown
    trades_df['month'] = trades_df['open_time'].dt.to_period('M')
    monthly = trades_df.groupby('month').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wr=('pnl', lambda x: (x > 0).mean() * 100)
    )
    print(f'\n--- MONTHLY P&L ---')
    print(f'  {"Month":<10} {"Trades":>7} {"P&L":>10} {"WR%":>6}')
    print(f'  {"-"*38}')
    for month, row in monthly.iterrows():
        flag = '  <<' if row['pnl'] < -50 else ''
        print(f'  {str(month):<10} {int(row["trades"]):>7} ${row["pnl"]:>9.2f} {row["wr"]:>5.1f}%{flag}')
    print(f'  {"-"*38}')
    print(f'  {"TOTAL":<10} {len(trades_df):>7} ${total_pnl:>9.2f}')

    print(f'\n--- SAMPLE TRADES (first 15) ---')
    print(trades_df.head(15).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
if len(trades_df) > 0:
    trades_df.to_csv(OUT_PATH, index=False)
    print(f'\nSaved trades: {OUT_PATH}')

equity_df = pd.DataFrame(equity_curve)
equity_df.to_csv('data/processed/grid_equity_curve.csv', index=False)
print(f'Saved equity curve: data/processed/grid_equity_curve.csv')

print('\n' + '=' * 70)
print('GRID BACKTEST COMPLETE')
print('=' * 70)
