"""
XAUUSD Strategy Comparison Backtest
====================================
Tests 4 strategy types on the same 1-minute dataset:
  1. EMA Crossover Trend Following
  2. Bollinger Mean Reversion
  3. Asian Range Breakout
  4. Trend-Filtered Asian Breakout (Hybrid)

Data: XAUUSD 1-minute bars, March 2022 – December 2024
P&L reported in $/point (multiply by lot size for actual $)
"""

import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = 'XAUUSD_backtest.csv'
SPREAD_COST = 0.30          # Conservative avg spread ($)

# Session times (broker time, assumed UTC+2)
ASIAN_START     = 1
ASIAN_END       = 9
ACTIVE_START    = 9         # London open
ACTIVE_END      = 21        # Before rollover
ENTRY_DEADLINE  = 17
FORCE_CLOSE     = 21


# ── Data Loading & Indicators ────────────────────────────────────────────────
def load_and_prepare(path):
    print(f'Loading {path}...')
    df = pd.read_csv(
        path, sep='\t',
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
               'TickVol', 'Vol', 'Spread'],
        skiprows=1
    )
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f'  {len(df):,} bars: {df.index[0]} → {df.index[-1]}')

    # ── Indicators ──
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']

    # EMAs
    df['ema8']  = df['Close'].ewm(span=8).mean()
    df['ema21'] = df['Close'].ewm(span=21).mean()
    df['ema50'] = df['Close'].ewm(span=50).mean()

    # EMA crossover detection
    above = (df['ema8'] > df['ema21']).astype(int)
    df['ema_cross_up']   = (above == 1) & (above.shift(1) == 0)
    df['ema_cross_down'] = (above == 0) & (above.shift(1) == 1)

    # ATR
    df['atr20'] = df['range'].rolling(20).mean()
    df['atr50'] = df['range'].rolling(50).mean()

    # Bollinger Bands (20, 2σ)
    df['sma20'] = df['Close'].rolling(20).mean()
    df['std20'] = df['Close'].rolling(20).std()
    df['boll_upper'] = df['sma20'] + 2 * df['std20']
    df['boll_lower'] = df['sma20'] - 2 * df['std20']

    # Trend slope
    df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(50)) / (df['ema50'].shift(50) + 1e-10)

    df['hour'] = df.index.hour
    df['trade_date'] = df.index.date

    df = df.dropna().copy()
    print(f'  {len(df):,} bars after indicator warmup')
    return df


# ── Generic forward-scan backtester ──────────────────────────────────────────
def backtest_signals(df, entries_mask, directions, entry_prices,
                     sl_prices, tp_prices, max_hold=480, min_gap=1):
    """
    For each entry signal, scan forward for SL/TP hit.
    directions: +1=LONG, -1=SHORT (numpy array aligned with df).
    """
    highs  = df['High'].values
    lows   = df['Low'].values
    closes = df['Close'].values
    idx    = df.index
    n      = len(df)

    entry_indices = np.where(entries_mask)[0]
    trades = []
    last_exit = -min_gap

    for i in entry_indices:
        if i <= last_exit + min_gap - 1:
            continue

        ep = entry_prices[i]
        sl = sl_prices[i]
        tp = tp_prices[i]
        d  = directions[i]

        if np.isnan(ep) or np.isnan(sl) or np.isnan(tp) or d == 0:
            continue

        end = min(i + max_hold, n)
        outcome    = 'TIME_OUT'
        exit_price = closes[end - 1] if end > i else ep
        exit_idx   = end - 1

        for j in range(i + 1, end):
            if d == 1:  # LONG
                if lows[j] <= sl:
                    outcome, exit_price, exit_idx = 'SL', sl, j; break
                if highs[j] >= tp:
                    outcome, exit_price, exit_idx = 'TP', tp, j; break
            else:       # SHORT
                if highs[j] >= sl:
                    outcome, exit_price, exit_idx = 'SL', sl, j; break
                if lows[j] <= tp:
                    outcome, exit_price, exit_idx = 'TP', tp, j; break

        pnl_raw = (exit_price - ep) * d
        pnl_net = pnl_raw - SPREAD_COST

        trades.append({
            'entry_time':  idx[i],
            'exit_time':   idx[exit_idx],
            'direction':   'LONG' if d == 1 else 'SHORT',
            'entry_price': round(ep, 2),
            'exit_price':  round(exit_price, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'outcome':     outcome,
            'pnl_raw':     round(pnl_raw, 2),
            'pnl_net':     round(pnl_net, 2),
            'hold_bars':   exit_idx - i,
        })
        last_exit = exit_idx

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 1 — EMA Crossover Trend Following
# ══════════════════════════════════════════════════════════════════════════════
def strategy_ema_crossover(df, rr=2.0, atr_sl=1.5):
    """
    Entry:  EMA8 crosses EMA21 during active session
    Filter: Close on same side of EMA50 as trade direction
    SL:     1.5 × ATR20       TP: RR × SL
    Hold:   max 480 bars (8h)  Gap: 10 bars between trades
    """
    active = (df['hour'] >= ACTIVE_START) & (df['hour'] < ACTIVE_END)

    long_sig  = df['ema_cross_up']   & active & (df['Close'] > df['ema50'])
    short_sig = df['ema_cross_down'] & active & (df['Close'] < df['ema50'])

    entries   = (long_sig | short_sig).values
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0))

    ep      = df['Close'].values
    sl_dist = atr_sl * df['atr20'].values
    tp_dist = sl_dist * rr

    sl = np.where(direction == 1, ep - sl_dist,
         np.where(direction == -1, ep + sl_dist, np.nan))
    tp = np.where(direction == 1, ep + tp_dist,
         np.where(direction == -1, ep - tp_dist, np.nan))

    return backtest_signals(df, entries, direction, ep, sl, tp,
                            max_hold=480, min_gap=10)


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 2 — Bollinger Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════
def strategy_bollinger_reversion(df, rr=1.5, atr_sl=1.0):
    """
    Entry:  Close crosses outside Bollinger Band in low-trend regime
    SL:     1.0 × ATR20       TP: max(dist to SMA20, RR × SL)
    Hold:   max 240 bars (4h)  Gap: 20 bars between trades
    """
    slope_abs = df['ema50_slope'].abs()
    low_trend = slope_abs < slope_abs.rolling(500, min_periods=100).median()

    prev_inside_lower = df['Close'].shift(1) >= df['boll_lower'].shift(1)
    prev_inside_upper = df['Close'].shift(1) <= df['boll_upper'].shift(1)

    long_sig  = (df['Close'] < df['boll_lower']) & prev_inside_lower & low_trend
    short_sig = (df['Close'] > df['boll_upper']) & prev_inside_upper & low_trend

    entries   = (long_sig | short_sig).values
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0))

    ep      = df['Close'].values
    sl_dist = atr_sl * df['atr20'].values
    mid     = df['sma20'].values
    dist_mid = np.abs(ep - mid)
    tp_dist = np.maximum(dist_mid, sl_dist * rr)

    sl = np.where(direction == 1, ep - sl_dist,
         np.where(direction == -1, ep + sl_dist, np.nan))
    tp = np.where(direction == 1, ep + tp_dist,
         np.where(direction == -1, ep - tp_dist, np.nan))

    return backtest_signals(df, entries, direction, ep, sl, tp,
                            max_hold=240, min_gap=20)


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 3 — Asian Range Breakout
# ══════════════════════════════════════════════════════════════════════════════
def strategy_asian_breakout(df, rr=2.0, buffer=0.50,
                            min_range=3.0, max_range=25.0):
    trading_days = sorted(df['trade_date'].unique())
    trades = []

    for day in trading_days:
        dd = df[df['trade_date'] == day]

        asian = dd[(dd['hour'] >= ASIAN_START) & (dd['hour'] < ASIAN_END)]
        if len(asian) < 60:
            continue

        ah, al = asian['High'].max(), asian['Low'].min()
        ar = ah - al
        if ar < min_range or ar > max_range:
            continue

        ew = dd[(dd['hour'] >= ASIAN_END) & (dd['hour'] < ENTRY_DEADLINE)]
        if len(ew) == 0:
            continue

        ll, sl_l = ah + buffer, al - buffer
        entry_price = direction = entry_time = None

        for ts, bar in ew.iterrows():
            if bar['High'] >= ll:
                direction, entry_price, entry_time = 'LONG', ll, ts; break
            if bar['Low'] <= sl_l:
                direction, entry_price, entry_time = 'SHORT', sl_l, ts; break

        if entry_price is None:
            continue

        sl_p = (al - buffer) if direction == 'LONG' else (ah + buffer)
        sd   = abs(entry_price - sl_p)
        tp_p = (entry_price + sd * rr) if direction == 'LONG' else (entry_price - sd * rr)

        pe = dd[(dd.index >= entry_time) & (dd['hour'] < FORCE_CLOSE)]
        outcome, ep_out, et_out = 'TIME_OUT', pe.iloc[-1]['Close'], pe.index[-1]

        for ts, bar in pe.iterrows():
            if direction == 'LONG':
                if bar['Low'] <= sl_p:
                    outcome, ep_out, et_out = 'SL', sl_p, ts; break
                if bar['High'] >= tp_p:
                    outcome, ep_out, et_out = 'TP', tp_p, ts; break
            else:
                if bar['High'] >= sl_p:
                    outcome, ep_out, et_out = 'SL', sl_p, ts; break
                if bar['Low'] <= tp_p:
                    outcome, ep_out, et_out = 'TP', tp_p, ts; break

        d = 1 if direction == 'LONG' else -1
        pnl_raw = (ep_out - entry_price) * d
        trades.append({
            'entry_time': entry_time, 'exit_time': et_out,
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'exit_price': round(ep_out, 2),
            'sl': round(sl_p, 2), 'tp': round(tp_p, 2),
            'outcome': outcome,
            'pnl_raw': round(pnl_raw, 2),
            'pnl_net': round(pnl_raw - SPREAD_COST, 2),
            'hold_bars': int((et_out - entry_time).total_seconds() / 60),
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 4 — Trend-Filtered Asian Breakout
# ══════════════════════════════════════════════════════════════════════════════
def strategy_trend_filtered_breakout(df, rr=2.0, buffer=0.50,
                                     min_range=3.0, max_range=25.0):
    """Same as Asian Breakout but only trade in EMA50-slope direction."""
    trading_days = sorted(df['trade_date'].unique())
    trades = []

    for day in trading_days:
        dd = df[df['trade_date'] == day]

        asian = dd[(dd['hour'] >= ASIAN_START) & (dd['hour'] < ASIAN_END)]
        if len(asian) < 60:
            continue

        ah, al = asian['High'].max(), asian['Low'].min()
        ar = ah - al
        if ar < min_range or ar > max_range:
            continue

        # Trend at Asian close
        slope = df.loc[asian.index[-1], 'ema50_slope']
        if np.isnan(slope):
            continue
        trend = 'LONG' if slope > 0 else 'SHORT'

        ew = dd[(dd['hour'] >= ASIAN_END) & (dd['hour'] < ENTRY_DEADLINE)]
        if len(ew) == 0:
            continue

        ll, sl_l = ah + buffer, al - buffer
        entry_price = direction = entry_time = None

        for ts, bar in ew.iterrows():
            if trend == 'LONG' and bar['High'] >= ll:
                direction, entry_price, entry_time = 'LONG', ll, ts; break
            if trend == 'SHORT' and bar['Low'] <= sl_l:
                direction, entry_price, entry_time = 'SHORT', sl_l, ts; break

        if entry_price is None:
            continue

        sl_p = (al - buffer) if direction == 'LONG' else (ah + buffer)
        sd   = abs(entry_price - sl_p)
        tp_p = (entry_price + sd * rr) if direction == 'LONG' else (entry_price - sd * rr)

        pe = dd[(dd.index >= entry_time) & (dd['hour'] < FORCE_CLOSE)]
        outcome, ep_out, et_out = 'TIME_OUT', pe.iloc[-1]['Close'], pe.index[-1]

        for ts, bar in pe.iterrows():
            if direction == 'LONG':
                if bar['Low'] <= sl_p:
                    outcome, ep_out, et_out = 'SL', sl_p, ts; break
                if bar['High'] >= tp_p:
                    outcome, ep_out, et_out = 'TP', tp_p, ts; break
            else:
                if bar['High'] >= sl_p:
                    outcome, ep_out, et_out = 'SL', sl_p, ts; break
                if bar['Low'] <= tp_p:
                    outcome, ep_out, et_out = 'TP', tp_p, ts; break

        d = 1 if direction == 'LONG' else -1
        pnl_raw = (ep_out - entry_price) * d
        trades.append({
            'entry_time': entry_time, 'exit_time': et_out,
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'exit_price': round(ep_out, 2),
            'sl': round(sl_p, 2), 'tp': round(tp_p, 2),
            'outcome': outcome,
            'pnl_raw': round(pnl_raw, 2),
            'pnl_net': round(pnl_raw - SPREAD_COST, 2),
            'hold_bars': int((et_out - entry_time).total_seconds() / 60),
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def summarize(results, name):
    if len(results) == 0:
        return {'Strategy': name, 'Trades': 0, 'WinRate%': 0,
                'TotalPnL': 0, 'AvgPnL': 0, 'PF': 0, 'MaxDD': 0,
                'Sharpe': 0, 'AvgHold': 0}

    t  = len(results)
    w  = len(results[results['outcome'] == 'TP'])
    lo = len(results[results['outcome'] == 'SL'])
    to = len(results[results['outcome'] == 'TIME_OUT'])
    wr = w / t * 100
    pnl = results['pnl_net'].sum()
    avg = results['pnl_net'].mean()

    cum = results['pnl_net'].cumsum()
    dd  = (cum - cum.cummax()).min()

    gross_win  = results.loc[results['pnl_net'] > 0, 'pnl_net'].sum()
    gross_loss = abs(results.loc[results['pnl_net'] < 0, 'pnl_net'].sum())
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    sharpe = avg / (results['pnl_net'].std() + 1e-10) * np.sqrt(252)

    return {
        'Strategy':  name,
        'Trades':    t,
        'Wins':      w,
        'Losses':    lo,
        'Timeouts':  to,
        'WinRate%':  round(wr, 1),
        'TotalPnL':  round(pnl, 2),
        'AvgPnL':    round(avg, 2),
        'PF':        pf,
        'MaxDD':     round(dd, 2),
        'Sharpe':    round(sharpe, 2),
        'AvgHold':   int(results['hold_bars'].mean()),
    }


def print_comparison(summaries):
    comp = pd.DataFrame(summaries)
    col_order = ['Strategy', 'Trades', 'Wins', 'Losses', 'Timeouts',
                 'WinRate%', 'TotalPnL', 'AvgPnL', 'PF', 'MaxDD',
                 'Sharpe', 'AvgHold']
    comp = comp[[c for c in col_order if c in comp.columns]]
    print(comp.to_string(index=False))
    return comp


def yearly_breakdown(results, name):
    if len(results) == 0:
        return
    r = results.copy()
    r['year'] = pd.to_datetime(r['entry_time']).dt.year
    print(f'\n  {name}:')
    for yr in sorted(r['year'].unique()):
        s = r[r['year'] == yr]
        wr = len(s[s['outcome'] == 'TP']) / len(s) * 100
        print(f'    {yr}: {len(s):3d} trades  WR={wr:5.1f}%  '
              f'P&L=${s["pnl_net"].sum():>9,.2f}  Avg=${s["pnl_net"].mean():>6.2f}')


def direction_breakdown(results, name):
    if len(results) == 0:
        return
    print(f'\n  {name}:')
    for d in ['LONG', 'SHORT']:
        s = results[results['direction'] == d]
        if len(s) > 0:
            wr = len(s[s['outcome'] == 'TP']) / len(s) * 100
            print(f'    {d:6s}: {len(s):4d} trades  WR={wr:5.1f}%  '
                  f'P&L=${s["pnl_net"].sum():>9,.2f}  Avg=${s["pnl_net"].mean():>6.2f}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 72)
    print('  XAUUSD STRATEGY COMPARISON BACKTEST')
    print('  March 2022 – December 2024  |  1-minute bars')
    print('=' * 72)

    df = load_and_prepare(DATA_PATH)

    strategies = {}

    # ── Strategy 1 ──
    print('\n  [1/4] EMA Crossover Trend Following ...')
    strategies['EMA Crossover'] = strategy_ema_crossover(df)
    print(f'         → {len(strategies["EMA Crossover"])} trades')

    # ── Strategy 2 ──
    print('  [2/4] Bollinger Mean Reversion ...')
    strategies['Bollinger MR'] = strategy_bollinger_reversion(df)
    print(f'         → {len(strategies["Bollinger MR"])} trades')

    # ── Strategy 3 ──
    print('  [3/4] Asian Range Breakout ...')
    strategies['Asian Breakout'] = strategy_asian_breakout(df)
    print(f'         → {len(strategies["Asian Breakout"])} trades')

    # ── Strategy 4 ──
    print('  [4/4] Trend-Filtered Breakout ...')
    strategies['Trend+Breakout'] = strategy_trend_filtered_breakout(df)
    print(f'         → {len(strategies["Trend+Breakout"])} trades')

    # ── Comparison table ──
    print(f'\n{"="*72}')
    print('  STRATEGY COMPARISON')
    print(f'{"="*72}\n')

    summaries = [summarize(r, name) for name, r in strategies.items()]
    comp = print_comparison(summaries)

    # ── By direction ──
    print(f'\n{"="*72}')
    print('  BY DIRECTION')
    print(f'{"="*72}')
    for name, r in strategies.items():
        direction_breakdown(r, name)

    # ── By year ──
    print(f'\n{"="*72}')
    print('  BY YEAR')
    print(f'{"="*72}')
    for name, r in strategies.items():
        yearly_breakdown(r, name)

    # ── Winner ──
    best_idx = comp['TotalPnL'].idxmax()
    best = comp.iloc[best_idx]
    print(f'\n{"="*72}')
    print(f'  🏆 BEST: {best["Strategy"]}')
    print(f'     P&L=${best["TotalPnL"]:,.2f}  |  WR={best["WinRate%"]}%  |  '
          f'PF={best["PF"]}  |  Trades={best["Trades"]}')
    print(f'{"="*72}')

    # ── Save ──
    for name, r in strategies.items():
        if len(r) > 0:
            fn = name.lower().replace(' ', '_').replace('+', '_')
            r.to_csv(f'data/processed/{fn}_results.csv', index=False)
    comp.to_csv('data/processed/strategy_comparison.csv', index=False)
    print(f'\n  All results saved to data/processed/')
