"""
XAUUSD Opening Range Breakout (ORB) — Numba + Optuna Optimization
==================================================================
Strategy: Define the high/low of the first N minutes of a session,
          then trade the breakout of that range.

Unlike Asian/London breakout (which uses a fixed session range),
ORB is flexible — it can be applied to ANY session opening:
  - Asian open (01:00 broker)
  - London open (09:00 broker)
  - NY open (15:00 broker)

Params to optimize:
  - session_open_h:  hour the session opens (0–15)
  - orb_minutes:     how many minutes define the opening range (5–120)
  - buffer:          pips above/below range to trigger ($0.05–$2.00)
  - rr:              reward-to-risk ratio (1.0–5.0)
  - min_range:       minimum range size to trade ($0.50–$10)
  - max_range:       maximum range size ($5–$40)
  - deadline_h:      last hour to enter (relative hours after open: 2–10)
  - hold_hours:      max hours to hold (2–14)
  - use_trend:       filter by EMA50 trend direction

Walk-forward: train ≤2023, test 2024.
"""

import numpy as np
import pandas as pd
import numba as nb
import optuna
import warnings
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH      = 'XAUUSD_backtest.csv'
SPREAD_COST    = 0.30
N_TRIALS       = 400
TRAIN_END_YEAR = 2023


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data(path):
    print(f'  Loading {path}...')
    df = pd.read_csv(
        path, sep='\t',
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
               'TickVol', 'Vol', 'Spread'],
        skiprows=1,
    )
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()

    # EMA50 for trend filter
    df['ema50'] = df['Close'].ewm(span=50).mean()
    df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(50)) / (df['ema50'].shift(50) + 1e-10)
    df = df.dropna().copy()

    print(f'  {len(df):,} bars: {df.index[0]} → {df.index[-1]}')
    return df


# ── Precompute day-level data ────────────────────────────────────────────────
def precompute_days(df):
    df = df.copy()
    df['_pos'] = np.arange(len(df))
    df['_date'] = df.index.date

    rows = []
    for date, grp in df.groupby('_date'):
        if len(grp) < 120:
            continue
        rows.append({
            'date':  date,
            'year':  date.year,
            'start': int(grp['_pos'].iloc[0]),
            'end':   int(grp['_pos'].iloc[-1]) + 1,
        })

    days = pd.DataFrame(rows)
    print(f'  {len(days)} trading days '
          f'(train ≤{TRAIN_END_YEAR}: {(days["year"] <= TRAIN_END_YEAR).sum()}, '
          f'test: {(days["year"] > TRAIN_END_YEAR).sum()})')
    return days


# ── Numba-compiled ORB backtest ──────────────────────────────────────────────
@nb.njit(cache=True)
def _fast_orb(highs, lows, closes, hours, minutes, ema_slopes,
              day_starts, day_ends,
              session_open_h, orb_minutes,
              rr, buffer, min_range, max_range,
              deadline_bars_after, hold_bars,
              spread, use_trend_filter):
    """
    Numba JIT Opening Range Breakout.
    For each day:
      1. Find first bar at session_open_h
      2. Compute high/low of first orb_minutes bars
      3. Trade breakout with buffer, SL at opposite side, TP at RR × SL
      4. Exit at SL, TP, or after hold_bars minutes
    """
    n_days = len(day_starts)
    pnl_arr = np.empty(n_days, dtype=np.float64)
    out_arr = np.empty(n_days, dtype=np.int32)
    dir_arr = np.empty(n_days, dtype=np.int32)
    count = 0

    for d in range(n_days):
        s = day_starts[d]
        e = day_ends[d]

        # ── Step 1: find session open bar ──
        open_bar = -1
        for i in range(s, e):
            if hours[i] == session_open_h:
                open_bar = i
                break

        if open_bar == -1:
            continue

        # ── Step 2: compute opening range (first orb_minutes bars) ──
        orb_end = min(open_bar + orb_minutes, e)
        if orb_end - open_bar < max(5, orb_minutes // 2):
            continue

        orb_high = -1e30
        orb_low  = 1e30
        trend_slope = 0.0

        for i in range(open_bar, orb_end):
            if highs[i] > orb_high:
                orb_high = highs[i]
            if lows[i] < orb_low:
                orb_low = lows[i]
            trend_slope = ema_slopes[i]

        orb_size = orb_high - orb_low
        if orb_size < min_range or orb_size > max_range:
            continue

        long_lv  = orb_high + buffer
        short_lv = orb_low - buffer

        # ── Step 3: find breakout entry ──
        deadline_bar = min(orb_end + deadline_bars_after, e)
        ebar = -1
        ep   = 0.0
        ddir = 0

        for i in range(orb_end, deadline_bar):
            if highs[i] >= long_lv:
                if use_trend_filter and trend_slope < 0:
                    pass
                else:
                    ddir = 1
                    ep   = long_lv
                    ebar = i
                    break

            if lows[i] <= short_lv:
                if use_trend_filter and trend_slope > 0:
                    pass
                else:
                    ddir = -1
                    ep   = short_lv
                    ebar = i
                    break

        if ebar == -1:
            continue

        # ── Step 4: SL / TP ──
        if ddir == 1:
            sl = orb_low - buffer
            sl_dist = ep - sl
            tp = ep + sl_dist * rr
        else:
            sl = orb_high + buffer
            sl_dist = sl - ep
            tp = ep - sl_dist * rr

        # ── Step 5: track outcome ──
        exit_bar = min(ebar + hold_bars, e)
        oc = 3  # timeout
        xp = closes[min(exit_bar - 1, ebar)]

        for i in range(ebar + 1, exit_bar):
            if ddir == 1:
                if lows[i] <= sl:
                    oc = 2; xp = sl; break
                if highs[i] >= tp:
                    oc = 1; xp = tp; break
            else:
                if highs[i] >= sl:
                    oc = 2; xp = sl; break
                if lows[i] <= tp:
                    oc = 1; xp = tp; break
            xp = closes[i]

        pnl_arr[count] = (xp - ep) * ddir - spread
        out_arr[count]  = oc
        dir_arr[count]  = ddir
        count += 1

    return pnl_arr[:count], out_arr[:count], dir_arr[:count]


# ── Python wrapper ───────────────────────────────────────────────────────────
def run_backtest(highs, lows, closes, hours, minutes, ema_slopes, days_df,
                 session_open_h, orb_minutes,
                 rr, buffer, min_range, max_range,
                 deadline_bars, hold_bars, use_trend,
                 spread=SPREAD_COST):
    pnl, outcomes, dirs = _fast_orb(
        highs, lows, closes, hours, minutes, ema_slopes,
        days_df['start'].values.astype(np.int64),
        days_df['end'].values.astype(np.int64),
        int(session_open_h), int(orb_minutes),
        float(rr), float(buffer),
        float(min_range), float(max_range),
        int(deadline_bars), int(hold_bars),
        float(spread), bool(use_trend),
    )

    n = len(pnl)
    if n < 10:
        return None

    wins = int((outcomes == 1).sum())
    losses = int((outcomes == 2).sum())
    timeouts = int((outcomes == 3).sum())
    total_pnl = float(pnl.sum())
    avg_pnl = float(pnl.mean())
    std_pnl = float(pnl.std()) if n > 1 else 1e-10
    t_stat = avg_pnl / (std_pnl / np.sqrt(n))
    wr = wins / n * 100

    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - running_max))

    gross_win = float(pnl[pnl > 0].sum()) if (pnl > 0).any() else 0.0
    gross_loss = float(np.abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 1e-10
    pf = gross_win / gross_loss

    longs  = int((dirs == 1).sum())
    shorts = int((dirs == -1).sum())
    long_pnl  = float(pnl[dirs == 1].sum()) if longs > 0 else 0.0
    short_pnl = float(pnl[dirs == -1].sum()) if shorts > 0 else 0.0

    return {
        'trades': n, 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'wr': round(wr, 1), 'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 3), 'std_pnl': round(std_pnl, 3),
        't_stat': round(t_stat, 2), 'pf': round(pf, 2),
        'max_dd': round(max_dd, 2),
        'sharpe': round(avg_pnl / (std_pnl + 1e-10) * np.sqrt(252), 2),
        'longs': longs, 'shorts': shorts,
        'long_pnl': round(long_pnl, 2), 'short_pnl': round(short_pnl, 2),
    }


# ── Optuna objective ─────────────────────────────────────────────────────────
def make_objective(highs, lows, closes, hours, minutes, ema_slopes, train_days):
    def objective(trial):
        session_open = trial.suggest_int('session_open_h', 0, 15)
        orb_min      = trial.suggest_int('orb_minutes', 5, 120, step=5)
        rr           = trial.suggest_float('rr', 1.0, 5.0, step=0.25)
        buffer       = trial.suggest_float('buffer', 0.05, 2.00, step=0.05)
        min_range    = trial.suggest_float('min_range', 0.5, 10.0, step=0.5)
        max_range    = trial.suggest_float('max_range', 5.0, 40.0, step=1.0)
        deadline_b   = trial.suggest_int('deadline_bars', 30, 600, step=30)
        hold_b       = trial.suggest_int('hold_bars', 60, 840, step=30)
        use_trend    = trial.suggest_categorical('use_trend', [True, False])

        if min_range >= max_range:
            return -999.0

        result = run_backtest(
            highs, lows, closes, hours, minutes, ema_slopes, train_days,
            session_open, orb_min, rr, buffer,
            min_range, max_range, deadline_b, hold_b, use_trend,
        )

        if result is None or result['trades'] < 30:
            return -999.0

        return result['t_stat']

    return objective


# ── Reporting ────────────────────────────────────────────────────────────────
def print_metrics(m, label=''):
    if m is None:
        print(f'  {label}: No trades')
        return
    print(f'  {label}')
    print(f'    Trades:    {m["trades"]:>5d}   (W:{m["wins"]} L:{m["losses"]} T:{m["timeouts"]})')
    print(f'    Win Rate:  {m["wr"]:>5.1f}%')
    print(f'    Total P&L: ${m["total_pnl"]:>9,.2f}')
    print(f'    Avg P&L:   ${m["avg_pnl"]:>9.3f}  (std: ${m["std_pnl"]:.3f})')
    print(f'    t-stat:    {m["t_stat"]:>6.2f}')
    print(f'    PF:        {m["pf"]:>6.2f}')
    print(f'    MaxDD:     ${m["max_dd"]:>9,.2f}')
    print(f'    Sharpe:    {m["sharpe"]:>6.2f}')
    print(f'    LONG:  {m["longs"]:>4d} trades  P&L=${m["long_pnl"]:>8,.2f}')
    print(f'    SHORT: {m["shorts"]:>4d} trades  P&L=${m["short_pnl"]:>8,.2f}')


def session_name(h):
    if h <= 1:
        return 'Asian'
    elif h <= 8:
        return 'Pre-London'
    elif h <= 10:
        return 'London'
    elif h <= 14:
        return 'NY Pre-Open'
    else:
        return 'NY Open'


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 72)
    print('  XAUUSD OPENING RANGE BREAKOUT (ORB) — NUMBA + OPTUNA')
    print('=' * 72)

    # ── Load & precompute ──
    df   = load_data(DATA_PATH)
    days = precompute_days(df)

    highs      = df['High'].values.astype(np.float64)
    lows       = df['Low'].values.astype(np.float64)
    closes     = df['Close'].values.astype(np.float64)
    hours      = df.index.hour.values.astype(np.int64)
    minutes    = df.index.minute.values.astype(np.int64)
    ema_slopes = df['ema50_slope'].values.astype(np.float64)

    train_days = days[days['year'] <= TRAIN_END_YEAR].copy()
    test_days  = days[days['year'] >  TRAIN_END_YEAR].copy()

    # ── Warm up Numba ──
    print('\n  Compiling Numba engine...')
    _ = run_backtest(highs, lows, closes, hours, minutes, ema_slopes, train_days,
                     session_open_h=9, orb_minutes=30, rr=2.0, buffer=0.5,
                     min_range=2.0, max_range=25.0,
                     deadline_bars=240, hold_bars=480, use_trend=False)
    print('  Done.\n')

    # ── Test 3 sessions as baselines ──
    sessions = [
        ('Asian ORB (01:00)',  1, 30),
        ('London ORB (09:00)', 9, 30),
        ('NY ORB (15:00)',     15, 15),
    ]

    print(f'{"─"*72}')
    print('  BASELINES — 3 SESSION ORBs (RR=2.0, Buf=$0.50, Range=2-25)')
    print(f'{"─"*72}')

    for name, open_h, orb_min in sessions:
        print(f'\n  === {name}, {orb_min}-min range ===')
        for label, ddf in [('Train', train_days), ('Test', test_days), ('All', days)]:
            m = run_backtest(highs, lows, closes, hours, minutes, ema_slopes, ddf,
                             open_h, orb_min, 2.0, 0.50, 2.0, 25.0, 240, 480, False)
            if m:
                print(f'    {label:5s}: {m["trades"]:>4d} trades  '
                      f'WR={m["wr"]:>5.1f}%  P&L=${m["total_pnl"]:>8,.2f}  '
                      f'PF={m["pf"]:>5.2f}  t={m["t_stat"]:>6.2f}  DD=${m["max_dd"]:>7,.2f}')
            else:
                print(f'    {label:5s}: No trades')

    # ══════════════════════════════════════════════════════════════════════
    #  OPTUNA — FULL SEARCH (all sessions, all params)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print(f'  OPTUNA BAYESIAN OPTIMIZATION ({N_TRIALS} trials — all sessions)')
    print(f'{"="*72}')

    objective = make_objective(highs, lows, closes, hours, minutes, ema_slopes, train_days)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))

    t1 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t1

    bp = study.best_params
    print(f'\n  Completed {N_TRIALS} trials in {elapsed:.1f}s '
          f'({elapsed/N_TRIALS*1000:.0f}ms/trial)')

    sess = session_name(bp['session_open_h'])
    print(f'\n  BEST PARAMETERS (in-sample):')
    print(f'    Session:     {sess} (hour {bp["session_open_h"]})')
    for k, v in bp.items():
        print(f'    {k:20s}: {v}')

    # ── Evaluate best params ──
    print(f'\n{"─"*72}')
    print('  OPTIMIZED RESULTS')
    print(f'{"─"*72}')

    opt_train = run_backtest(highs, lows, closes, hours, minutes, ema_slopes, train_days,
                             bp['session_open_h'], bp['orb_minutes'],
                             bp['rr'], bp['buffer'],
                             bp['min_range'], bp['max_range'],
                             bp['deadline_bars'], bp['hold_bars'],
                             bp['use_trend'])
    opt_test  = run_backtest(highs, lows, closes, hours, minutes, ema_slopes, test_days,
                             bp['session_open_h'], bp['orb_minutes'],
                             bp['rr'], bp['buffer'],
                             bp['min_range'], bp['max_range'],
                             bp['deadline_bars'], bp['hold_bars'],
                             bp['use_trend'])
    opt_all   = run_backtest(highs, lows, closes, hours, minutes, ema_slopes, days,
                             bp['session_open_h'], bp['orb_minutes'],
                             bp['rr'], bp['buffer'],
                             bp['min_range'], bp['max_range'],
                             bp['deadline_bars'], bp['hold_bars'],
                             bp['use_trend'])

    print_metrics(opt_train, f'Train (2022-2023) — OPTIMIZED [{sess}]')
    print_metrics(opt_test,  f'Test  (2024)      — OUT-OF-SAMPLE [{sess}]')
    print_metrics(opt_all,   f'All   (2022-2024) — FULL PERIOD [{sess}]')

    # ══════════════════════════════════════════════════════════════════════
    #  TOP 15 TRIALS (to see which sessions dominate)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print('  TOP 15 PARAMETER COMBINATIONS (by t-stat on train)')
    print(f'{"="*72}\n')

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > -900]
    trials_df = trials_df.sort_values('value', ascending=False).head(15)

    print(f'  {"#":>2s}  {"t_stat":>6s}  {"Sess":>5s}  {"Open":>4s}  {"ORBm":>4s}  '
          f'{"RR":>4s}  {"Buf":>5s}  {"MinR":>5s}  {"MaxR":>5s}  '
          f'{"DL_b":>5s}  {"Hold":>4s}  {"Trend":>5s}')
    print(f'  {"─"*75}')
    for rank, (_, row) in enumerate(trials_df.iterrows(), 1):
        sn = session_name(int(row['params_session_open_h']))[:5]
        trend_str = 'Yes' if row.get('params_use_trend', False) else 'No'
        print(f'  {rank:2d}  {row["value"]:6.2f}  '
              f'{sn:>5s}  '
              f'{int(row["params_session_open_h"]):4d}  '
              f'{int(row["params_orb_minutes"]):4d}  '
              f'{row["params_rr"]:4.1f}  '
              f'{row["params_buffer"]:5.2f}  '
              f'{row["params_min_range"]:5.1f}  '
              f'{row["params_max_range"]:5.0f}  '
              f'{int(row["params_deadline_bars"]):5d}  '
              f'{int(row["params_hold_bars"]):4d}  '
              f'{trend_str:>5s}')

    # ── Validate top-10 on test ──
    print(f'\n{"─"*72}')
    print('  TOP 10 — OUT-OF-SAMPLE VALIDATION (2024)')
    print(f'{"─"*72}\n')

    print(f'  {"#":>2s}  {"t_train":>7s}  {"Sess":>5s}  │  '
          f'{"t_test":>7s}  {"Trades":>6s}  {"WR%":>5s}  {"P&L":>9s}  {"PF":>5s}  {"MaxDD":>8s}')
    print(f'  {"─"*75}')

    for rank, (_, row) in enumerate(trials_df.head(10).iterrows(), 1):
        sn = session_name(int(row['params_session_open_h']))[:5]
        test_m = run_backtest(
            highs, lows, closes, hours, minutes, ema_slopes, test_days,
            int(row['params_session_open_h']), int(row['params_orb_minutes']),
            row['params_rr'], row['params_buffer'],
            row['params_min_range'], row['params_max_range'],
            int(row['params_deadline_bars']), int(row['params_hold_bars']),
            row.get('params_use_trend', False),
        )
        if test_m is None:
            print(f'  {rank:2d}  {row["value"]:7.2f}  {sn:>5s}  │  no trades on test')
            continue

        print(f'  {rank:2d}  {row["value"]:7.2f}  {sn:>5s}  │  '
              f'{test_m["t_stat"]:7.2f}  {test_m["trades"]:6d}  '
              f'{test_m["wr"]:5.1f}  ${test_m["total_pnl"]:>8,.2f}  '
              f'{test_m["pf"]:5.2f}  ${test_m["max_dd"]:>7,.2f}')

    # ── Head-to-head comparison ──
    print(f'\n{"="*72}')
    print('  STRATEGY COMPARISON (all optimized, full period)')
    print(f'{"="*72}')
    print(f'  {"Strategy":<25s}  {"P&L":>9s}  {"PF":>5s}  {"t-stat":>6s}  {"OOS t":>6s}  {"MaxDD":>8s}')
    print(f'  {"─"*65}')
    print(f'  {"Asian Breakout":<25s}  {"$516":>9s}  {"1.51":>5s}  {"2.52":>6s}  {"1.53":>6s}  {"$-96":>8s}')
    print(f'  {"London Breakout":<25s}  {"$558":>9s}  {"1.29":>5s}  {"1.91":>6s}  {"0.58":>6s}  {"$-166":>8s}')
    if opt_all and opt_test:
        print(f'  {"ORB (" + sess + ")":<25s}  '
              f'${opt_all["total_pnl"]:>8,.2f}  '
              f'{opt_all["pf"]:>5.2f}  '
              f'{opt_all["t_stat"]:>6.2f}  '
              f'{opt_test["t_stat"]:>6.2f}  '
              f'${opt_all["max_dd"]:>7,.2f}')

    total_time = time.time() - t0
    print(f'\n  Total time: {total_time:.1f}s')
    print(f'{"="*72}')
