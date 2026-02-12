"""
Asian Range Breakout (ARB) Strategy — XAUUSD Backtest
=====================================================
Strategy:
  1. Mark the High & Low of the Asian session (01:00–09:00 broker time)
  2. After 09:00, wait for price to break above Asian High or below Asian Low
  3. Entry: breakout level ± buffer
  4. SL: opposite side of Asian range ± buffer
  5. TP: entry ± (SL distance × RR ratio)
  6. One trade per day max; first breakout direction wins
  7. Force-close at 21:00 broker time (before daily rollover)

Data: XAUUSD 1-minute bars, broker time assumed UTC+2
"""

import pandas as pd
import numpy as np
from itertools import product

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = 'XAUUSD_backtest.csv'

# Session times (broker time, assumed UTC+2)
ASIAN_START     = 1      # 01:00 broker (23:00 UTC prev day)
ASIAN_END       = 9      # 09:00 broker (07:00 UTC) — London open
ENTRY_DEADLINE  = 17     # 17:00 broker — stop looking for entries
FORCE_CLOSE     = 21     # 21:00 broker — close before rollover gap

# Strategy parameters
RR_RATIO    = 2.0        # Reward:Risk ratio
BUFFER      = 0.50       # $ buffer above/below range for entry/SL
MIN_RANGE   = 3.0        # Skip if Asian range < $3
MAX_RANGE   = 25.0       # Skip if Asian range > $25
MIN_ASIAN_BARS = 60      # Need at least 60 bars (~1 hour) of Asian data


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(path):
    print(f'Loading {path}...')
    df = pd.read_csv(
        path, sep='\t',
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
               'TickVol', 'Vol', 'Spread'],
        skiprows=1
    )
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df['Spread_dollars'] = df['Spread'] * 0.01  # 1 point = $0.01
    df = df[['Open', 'High', 'Low', 'Close', 'Spread_dollars']].copy()
    print(f'Loaded {len(df):,} bars: {df.index[0]} → {df.index[-1]}')
    return df


# ── Core backtest engine ─────────────────────────────────────────────────────
def run_backtest(df, rr_ratio=RR_RATIO, buffer=BUFFER,
                 min_range=MIN_RANGE, max_range=MAX_RANGE,
                 asian_start=ASIAN_START, asian_end=ASIAN_END,
                 entry_deadline=ENTRY_DEADLINE, force_close=FORCE_CLOSE,
                 use_spread=True, verbose=True):
    """Run Asian Range Breakout backtest. Returns DataFrame of trades."""

    df['trade_date'] = df.index.date
    trading_days = sorted(df['trade_date'].unique())

    if verbose:
        print(f'Trading days: {len(trading_days)}')

    trades = []
    skipped = {
        'too_few_bars': 0,
        'range_too_small': 0,
        'range_too_large': 0,
        'no_breakout': 0,
        'no_exit_bars': 0,
    }

    for day in trading_days:
        day_df = df[df['trade_date'] == day]

        # ── 1. Asian session bars ──
        asian = day_df[
            (day_df.index.hour >= asian_start) & (day_df.index.hour < asian_end)
        ]
        if len(asian) < MIN_ASIAN_BARS:
            skipped['too_few_bars'] += 1
            continue

        asian_high  = asian['High'].max()
        asian_low   = asian['Low'].min()
        asian_range = asian_high - asian_low

        # ── 2. Range filter ──
        if asian_range < min_range:
            skipped['range_too_small'] += 1
            continue
        if asian_range > max_range:
            skipped['range_too_large'] += 1
            continue

        # ── 3. Entry window — look for first breakout ──
        entry_window = day_df[
            (day_df.index.hour >= asian_end) & (day_df.index.hour < entry_deadline)
        ]
        if len(entry_window) == 0:
            skipped['no_breakout'] += 1
            continue

        long_level  = asian_high + buffer
        short_level = asian_low  - buffer

        entry_price  = None
        direction    = None
        entry_time   = None
        entry_spread = 0.0

        for ts, bar in entry_window.iterrows():
            hit_long  = bar['High'] >= long_level
            hit_short = bar['Low']  <= short_level

            if hit_long and hit_short:
                # Both hit on same 1m bar (rare) — use open position as tiebreak
                if bar['Open'] <= (asian_high + asian_low) / 2:
                    hit_short = False
                else:
                    hit_long = False

            if hit_long:
                direction    = 'LONG'
                entry_price  = long_level
                entry_spread = bar['Spread_dollars']
                entry_time   = ts
                break
            elif hit_short:
                direction    = 'SHORT'
                entry_price  = short_level
                entry_spread = bar['Spread_dollars']
                entry_time   = ts
                break

        if entry_price is None:
            skipped['no_breakout'] += 1
            continue

        # ── 4. SL & TP ──
        if direction == 'LONG':
            sl_price = asian_low  - buffer
            sl_dist  = entry_price - sl_price
            tp_price = entry_price + sl_dist * rr_ratio
        else:
            sl_price = asian_high + buffer
            sl_dist  = sl_price - entry_price
            tp_price = entry_price - sl_dist * rr_ratio

        # ── 5. Track outcome ──
        post_entry = day_df[
            (day_df.index >= entry_time) & (day_df.index.hour < force_close)
        ]
        if len(post_entry) == 0:
            skipped['no_exit_bars'] += 1
            continue

        outcome    = None
        exit_price = None
        exit_time  = None

        for ts, bar in post_entry.iterrows():
            if direction == 'LONG':
                if bar['Low'] <= sl_price:
                    outcome, exit_price, exit_time = 'SL', sl_price, ts
                    break
                if bar['High'] >= tp_price:
                    outcome, exit_price, exit_time = 'TP', tp_price, ts
                    break
            else:
                if bar['High'] >= sl_price:
                    outcome, exit_price, exit_time = 'SL', sl_price, ts
                    break
                if bar['Low'] <= tp_price:
                    outcome, exit_price, exit_time = 'TP', tp_price, ts
                    break

        if outcome is None:
            outcome    = 'TIME_OUT'
            exit_price = post_entry.iloc[-1]['Close']
            exit_time  = post_entry.index[-1]

        # ── 6. P&L ──
        spread_cost = entry_spread if use_spread else 0.0
        if direction == 'LONG':
            pnl_raw = exit_price - entry_price
        else:
            pnl_raw = entry_price - exit_price
        pnl_net = pnl_raw - spread_cost

        trades.append({
            'date':         day,
            'day_of_week':  pd.Timestamp(day).day_name(),
            'direction':    direction,
            'entry_time':   entry_time,
            'exit_time':    exit_time,
            'entry_price':  round(entry_price, 2),
            'exit_price':   round(exit_price, 2),
            'sl_price':     round(sl_price, 2),
            'tp_price':     round(tp_price, 2),
            'asian_high':   round(asian_high, 2),
            'asian_low':    round(asian_low, 2),
            'asian_range':  round(asian_range, 2),
            'sl_distance':  round(sl_dist, 2),
            'outcome':      outcome,
            'pnl_raw':      round(pnl_raw, 2),
            'spread_cost':  round(spread_cost, 2),
            'pnl_net':      round(pnl_net, 2),
            'duration_min': round((exit_time - entry_time).total_seconds() / 60, 1),
        })

    results = pd.DataFrame(trades)
    if verbose:
        for reason, count in skipped.items():
            if count > 0:
                print(f'  Skipped ({reason}): {count}')
    return results


# ── Reporting ────────────────────────────────────────────────────────────────
def print_results(results, label=''):
    if len(results) == 0:
        print('No trades generated!')
        return

    total    = len(results)
    wins     = results[results['outcome'] == 'TP']
    losses   = results[results['outcome'] == 'SL']
    timeouts = results[results['outcome'] == 'TIME_OUT']
    win_rate = len(wins) / total * 100

    print(f'\n{"="*70}')
    if label:
        print(f'  {label}')
        print(f'{"="*70}')

    print(f'\n  Total trades:     {total}')
    print(f'  Wins (TP):        {len(wins)} ({len(wins)/total*100:.1f}%)')
    print(f'  Losses (SL):      {len(losses)} ({len(losses)/total*100:.1f}%)')
    print(f'  Timeouts:         {len(timeouts)} ({len(timeouts)/total*100:.1f}%)')
    print(f'  WIN RATE:         {win_rate:.1f}%')

    # P&L
    total_pnl = results['pnl_net'].sum()
    avg_win   = wins['pnl_net'].mean() if len(wins) > 0 else 0
    avg_loss  = losses['pnl_net'].mean() if len(losses) > 0 else 0

    print(f'\n  Total P&L:        ${total_pnl:,.2f}')
    print(f'  Avg P&L/trade:    ${results["pnl_net"].mean():,.2f}')
    print(f'  Avg win:          ${avg_win:,.2f}')
    print(f'  Avg loss:         ${avg_loss:,.2f}')
    if len(losses) > 0 and losses['pnl_net'].sum() != 0:
        pf = wins['pnl_net'].sum() / abs(losses['pnl_net'].sum())
        print(f'  Profit factor:    {pf:.2f}')

    # Drawdown
    cumulative  = results['pnl_net'].cumsum()
    running_max = cumulative.cummax()
    max_dd      = (cumulative - running_max).min()
    print(f'  Max drawdown:     ${max_dd:,.2f}')
    print(f'  Peak equity:      ${cumulative.max():,.2f}')

    # By direction
    print(f'\n  {"─"*50}')
    print(f'  BY DIRECTION:')
    for d in ['LONG', 'SHORT']:
        s = results[results['direction'] == d]
        if len(s) > 0:
            wr = len(s[s['outcome'] == 'TP']) / len(s) * 100
            print(f'    {d:6s}: {len(s):3d} trades  WR={wr:5.1f}%  '
                  f'P&L=${s["pnl_net"].sum():>9,.2f}  Avg=${s["pnl_net"].mean():>6.2f}')

    # By day of week
    print(f'\n  {"─"*50}')
    print(f'  BY DAY OF WEEK:')
    for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        s = results[results['day_of_week'] == dow]
        if len(s) > 0:
            wr = len(s[s['outcome'] == 'TP']) / len(s) * 100
            print(f'    {dow:12s}: {len(s):3d} trades  WR={wr:5.1f}%  '
                  f'P&L=${s["pnl_net"].sum():>9,.2f}  Avg=${s["pnl_net"].mean():>6.2f}')

    # By year
    print(f'\n  {"─"*50}')
    print(f'  BY YEAR:')
    results_copy = results.copy()
    results_copy['year'] = pd.to_datetime(results_copy['date']).dt.year
    for yr in sorted(results_copy['year'].unique()):
        s = results_copy[results_copy['year'] == yr]
        wr = len(s[s['outcome'] == 'TP']) / len(s) * 100
        print(f'    {yr}: {len(s):3d} trades  WR={wr:5.1f}%  '
              f'P&L=${s["pnl_net"].sum():>9,.2f}  Avg=${s["pnl_net"].mean():>6.2f}')

    # By month
    print(f'\n  {"─"*50}')
    print(f'  BY MONTH:')
    results_copy['month'] = pd.to_datetime(results_copy['date']).dt.to_period('M')
    monthly = results_copy.groupby('month').agg(
        trades=('pnl_net', 'count'),
        pnl=('pnl_net', 'sum'),
        wins=('outcome', lambda x: (x == 'TP').sum()),
    )
    monthly['wr'] = (monthly['wins'] / monthly['trades'] * 100).round(1)
    for idx, row in monthly.iterrows():
        bar = '█' * int(max(0, row['pnl']) / 5) + '░' * int(max(0, -row['pnl']) / 5)
        print(f'    {str(idx):8s}: {row["trades"]:3.0f} trades  WR={row["wr"]:5.1f}%  '
              f'P&L=${row["pnl"]:>9,.2f}  {bar}')

    # Asian range stats
    print(f'\n  {"─"*50}')
    print(f'  ASIAN RANGE STATS:')
    print(f'    Average:  ${results["asian_range"].mean():.2f}')
    print(f'    Median:   ${results["asian_range"].median():.2f}')
    print(f'    Min:      ${results["asian_range"].min():.2f}')
    print(f'    Max:      ${results["asian_range"].max():.2f}')

    # Duration
    print(f'\n  TRADE DURATION:')
    print(f'    Average:  {results["duration_min"].mean():.0f} min '
          f'({results["duration_min"].mean()/60:.1f} hrs)')
    print(f'    Median:   {results["duration_min"].median():.0f} min')

    # Equity checkpoints
    cum = results['pnl_net'].cumsum()
    print(f'\n  EQUITY CURVE (checkpoints):')
    for pct in [0, 25, 50, 75, 100]:
        i = min(int(len(cum) * pct / 100), len(cum) - 1)
        print(f'    Trade {i+1:>4d}/{total}: ${cum.iloc[i]:>9,.2f}')


# ── Parameter sensitivity sweep ─────────────────────────────────────────────
def parameter_sweep(df):
    print(f'\n{"="*70}')
    print('  PARAMETER SENSITIVITY ANALYSIS')
    print(f'{"="*70}')

    rr_values     = [1.5, 2.0, 2.5, 3.0]
    buffer_values = [0.25, 0.50, 1.00]
    min_rng_vals  = [2.0, 3.0, 5.0]

    rows = []
    for rr, buf, mr in product(rr_values, buffer_values, min_rng_vals):
        res = run_backtest(df, rr_ratio=rr, buffer=buf, min_range=mr,
                           verbose=False)
        if len(res) == 0:
            continue
        total = len(res)
        wins  = len(res[res['outcome'] == 'TP'])
        wr    = wins / total * 100
        pnl   = res['pnl_net'].sum()
        avg   = res['pnl_net'].mean()
        cum   = res['pnl_net'].cumsum()
        dd    = (cum - cum.cummax()).min()
        rows.append({
            'RR': rr, 'Buffer': buf, 'MinRange': mr,
            'Trades': total, 'WinRate': round(wr, 1),
            'TotalPnL': round(pnl, 2), 'AvgPnL': round(avg, 2),
            'MaxDD': round(dd, 2),
        })

    sweep = pd.DataFrame(rows)
    sweep = sweep.sort_values('TotalPnL', ascending=False)

    print(f'\n  Top 10 parameter combinations by Total P&L:\n')
    print(f'  {"RR":>4s}  {"Buf":>5s}  {"MinR":>5s}  {"Trades":>6s}  '
          f'{"WR%":>5s}  {"TotalP&L":>10s}  {"AvgP&L":>8s}  {"MaxDD":>9s}')
    print(f'  {"─"*60}')
    for _, row in sweep.head(10).iterrows():
        print(f'  {row["RR"]:4.1f}  {row["Buffer"]:5.2f}  {row["MinRange"]:5.1f}  '
              f'{row["Trades"]:6.0f}  {row["WinRate"]:5.1f}  '
              f'${row["TotalPnL"]:>9,.2f}  ${row["AvgPnL"]:>7.2f}  '
              f'${row["MaxDD"]:>8,.2f}')

    print(f'\n  Bottom 5 (worst):')
    for _, row in sweep.tail(5).iterrows():
        print(f'  {row["RR"]:4.1f}  {row["Buffer"]:5.2f}  {row["MinRange"]:5.1f}  '
              f'{row["Trades"]:6.0f}  {row["WinRate"]:5.1f}  '
              f'${row["TotalPnL"]:>9,.2f}  ${row["AvgPnL"]:>7.2f}  '
              f'${row["MaxDD"]:>8,.2f}')

    sweep.to_csv('data/processed/arb_parameter_sweep.csv', index=False)
    print(f'\n  Full sweep saved to data/processed/arb_parameter_sweep.csv')
    return sweep


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 70)
    print('  ASIAN RANGE BREAKOUT — XAUUSD BACKTEST')
    print('=' * 70)
    print(f'\n  Config: RR={RR_RATIO}, Buffer=${BUFFER}, '
          f'Range=[${MIN_RANGE}-${MAX_RANGE}]')
    print(f'  Sessions (broker time): Asian={ASIAN_START:02d}:00-{ASIAN_END:02d}:00, '
          f'Entry until {ENTRY_DEADLINE:02d}:00, Close by {FORCE_CLOSE:02d}:00')

    df = load_data(DATA_PATH)

    # ── Run primary backtest ──
    results = run_backtest(df)
    print_results(results, label='PRIMARY BACKTEST RESULTS')

    # Save trade log
    if len(results) > 0:
        results.to_csv('data/processed/arb_backtest_results.csv', index=False)
        print(f'\n  Trade log saved to data/processed/arb_backtest_results.csv')

    # ── Parameter sensitivity ──
    print('\nRunning parameter sweep (36 combinations)...')
    sweep = parameter_sweep(df)

    print(f'\n{"="*70}')
    print('  DONE')
    print(f'{"="*70}')
