"""
XAUUSD London Open Breakout — Numba + Optuna Optimization
==========================================================
Strategy: Define a pre-London range, trade breakout at London open.
  - Pre-London range: accumulation zone before London session
  - Entry: breakout above/below range + buffer
  - SL: opposite side of range
  - TP: RR × SL distance
  - Forced close before end of session

Params to optimize:
  - range_start_h:  hour to start measuring range (0–6 broker time)
  - range_end_h:    hour range ends / entry window opens (7–10)
  - buffer:         pips above/below range to trigger entry ($0.10–$2.00)
  - rr:             reward-to-risk ratio (1.0–4.0)
  - min_range:      minimum range size to trade ($1–$15)
  - max_range:      maximum range size ($10–$40)
  - deadline_h:     last hour to enter (13–20)
  - force_close_h:  force close hour (18–23)
  - use_trend:      filter by EMA50 trend direction

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
N_TRIALS       = 300
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
    """
    For each trading day, store bar index range, year.
    Range highs/lows computed dynamically per trial (range_start/end vary).
    """
    df = df.copy()
    df['_pos'] = np.arange(len(df))
    df['_hour'] = df.index.hour
    df['_date'] = df.index.date

    rows = []
    for date, grp in df.groupby('_date'):
        if len(grp) < 120:  # skip partial days
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


# ── Numba-compiled backtest ──────────────────────────────────────────────────
@nb.njit(cache=True)
def _fast_london_breakout(highs, lows, closes, hours, ema_slopes,
                          day_starts, day_ends,
                          range_start_h, range_end_h,
                          rr, buffer, min_range, max_range,
                          deadline_h, close_h,
                          spread, use_trend_filter):
    """
    Numba JIT London Open Breakout backtest.
    For each day:
      1. Compute high/low of bars in [range_start_h, range_end_h)
      2. Look for breakout in [range_end_h, deadline_h)
      3. Track SL/TP/timeout
    """
    n_days = len(day_starts)
    pnl_arr = np.empty(n_days, dtype=np.float64)
    out_arr = np.empty(n_days, dtype=np.int32)
    dir_arr = np.empty(n_days, dtype=np.int32)
    count = 0

    for d in range(n_days):
        s = day_starts[d]
        e = day_ends[d]

        # ── Step 1: compute range ──
        rng_high = -1e30
        rng_low  = 1e30
        rng_bars = 0
        trend_slope = 0.0

        for i in range(s, e):
            h = hours[i]
            if h >= range_start_h and h < range_end_h:
                if highs[i] > rng_high:
                    rng_high = highs[i]
                if lows[i] < rng_low:
                    rng_low = lows[i]
                rng_bars += 1
                trend_slope = ema_slopes[i]  # last slope in range

        if rng_bars < 30:
            continue

        rng_size = rng_high - rng_low
        if rng_size < min_range or rng_size > max_range:
            continue

        long_lv  = rng_high + buffer
        short_lv = rng_low - buffer

        # ── Step 2: find breakout entry ──
        ebar = -1
        ep   = 0.0
        ddir = 0

        for i in range(s, e):
            h = hours[i]
            if h < range_end_h or h >= deadline_h:
                continue

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

        # ── Step 3: SL / TP ──
        if ddir == 1:
            sl = rng_low - buffer
            sl_dist = ep - sl
            tp = ep + sl_dist * rr
        else:
            sl = rng_high + buffer
            sl_dist = sl - ep
            tp = ep - sl_dist * rr

        # ── Step 4: track outcome ──
        oc = 3  # timeout
        xp = closes[min(e - 1, ebar)]

        for i in range(ebar + 1, e):
            if hours[i] >= close_h:
                xp = closes[i]
                break

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
def run_backtest(highs, lows, closes, hours, ema_slopes, days_df,
                 range_start_h, range_end_h,
                 rr, buffer, min_range, max_range,
                 deadline_h, close_h, use_trend,
                 spread=SPREAD_COST):
    pnl, outcomes, dirs = _fast_london_breakout(
        highs, lows, closes, hours, ema_slopes,
        days_df['start'].values.astype(np.int64),
        days_df['end'].values.astype(np.int64),
        int(range_start_h), int(range_end_h),
        float(rr), float(buffer),
        float(min_range), float(max_range),
        int(deadline_h), int(close_h),
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
def make_objective(highs, lows, closes, hours, ema_slopes, train_days):
    def objective(trial):
        range_start = trial.suggest_int('range_start_h', 0, 5)
        range_end   = trial.suggest_int('range_end_h', 7, 10)
        rr          = trial.suggest_float('rr', 1.0, 4.0, step=0.25)
        buffer      = trial.suggest_float('buffer', 0.10, 2.00, step=0.10)
        min_range   = trial.suggest_float('min_range', 1.0, 15.0, step=0.5)
        max_range   = trial.suggest_float('max_range', 10.0, 40.0, step=1.0)
        deadline    = trial.suggest_int('deadline_h', 13, 20)
        close_h     = trial.suggest_int('force_close_h', 18, 23)
        use_trend   = trial.suggest_categorical('use_trend', [True, False])

        if range_start >= range_end:
            return -999.0
        if close_h <= deadline:
            return -999.0
        if min_range >= max_range:
            return -999.0

        result = run_backtest(
            highs, lows, closes, hours, ema_slopes, train_days,
            range_start, range_end, rr, buffer,
            min_range, max_range, deadline, close_h, use_trend,
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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 72)
    print('  XAUUSD LONDON OPEN BREAKOUT — NUMBA + OPTUNA OPTIMIZATION')
    print('=' * 72)

    # ── Load & precompute ──
    df   = load_data(DATA_PATH)
    days = precompute_days(df)

    highs      = df['High'].values.astype(np.float64)
    lows       = df['Low'].values.astype(np.float64)
    closes     = df['Close'].values.astype(np.float64)
    hours      = df.index.hour.values.astype(np.int64)
    ema_slopes = df['ema50_slope'].values.astype(np.float64)

    train_days = days[days['year'] <= TRAIN_END_YEAR].copy()
    test_days  = days[days['year'] >  TRAIN_END_YEAR].copy()

    # ── Warm up Numba ──
    print('\n  Compiling Numba engine...')
    _ = run_backtest(highs, lows, closes, hours, ema_slopes, train_days,
                     range_start_h=2, range_end_h=9, rr=2.0, buffer=0.5,
                     min_range=3.0, max_range=25.0,
                     deadline_h=17, close_h=21, use_trend=False)
    print('  Done.\n')

    # ── Baseline (typical London breakout params) ──
    print(f'{"─"*72}')
    print('  BASELINE (Range=02:00-09:00, RR=2.0, Buf=0.50, MinR=3, MaxR=25)')
    print(f'{"─"*72}')
    base_train = run_backtest(highs, lows, closes, hours, ema_slopes, train_days,
                              2, 9, 2.0, 0.50, 3.0, 25.0, 17, 21, False)
    base_test  = run_backtest(highs, lows, closes, hours, ema_slopes, test_days,
                              2, 9, 2.0, 0.50, 3.0, 25.0, 17, 21, False)
    base_all   = run_backtest(highs, lows, closes, hours, ema_slopes, days,
                              2, 9, 2.0, 0.50, 3.0, 25.0, 17, 21, False)
    print_metrics(base_train, 'Train (2022-2023)')
    print_metrics(base_test,  'Test  (2024)')
    print_metrics(base_all,   'All   (2022-2024)')

    # ── Baseline + trend filter ──
    print(f'\n{"─"*72}')
    print('  BASELINE + TREND FILTER')
    print(f'{"─"*72}')
    bt_train = run_backtest(highs, lows, closes, hours, ema_slopes, train_days,
                            2, 9, 2.0, 0.50, 3.0, 25.0, 17, 21, True)
    bt_test  = run_backtest(highs, lows, closes, hours, ema_slopes, test_days,
                            2, 9, 2.0, 0.50, 3.0, 25.0, 17, 21, True)
    bt_all   = run_backtest(highs, lows, closes, hours, ema_slopes, days,
                            2, 9, 2.0, 0.50, 3.0, 25.0, 17, 21, True)
    print_metrics(bt_train, 'Train (2022-2023)')
    print_metrics(bt_test,  'Test  (2024)')
    print_metrics(bt_all,   'All   (2022-2024)')

    # ══════════════════════════════════════════════════════════════════════
    #  OPTUNA OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print(f'  OPTUNA BAYESIAN OPTIMIZATION ({N_TRIALS} trials on train set)')
    print(f'{"="*72}')

    objective = make_objective(highs, lows, closes, hours, ema_slopes, train_days)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))

    t1 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t1

    bp = study.best_params
    print(f'\n  Completed {N_TRIALS} trials in {elapsed:.1f}s '
          f'({elapsed/N_TRIALS*1000:.0f}ms/trial)')

    print(f'\n  BEST PARAMETERS (in-sample):')
    for k, v in bp.items():
        print(f'    {k:20s}: {v}')

    # ── Evaluate best params ──
    print(f'\n{"─"*72}')
    print('  OPTIMIZED RESULTS')
    print(f'{"─"*72}')

    opt_train = run_backtest(highs, lows, closes, hours, ema_slopes, train_days,
                             bp['range_start_h'], bp['range_end_h'],
                             bp['rr'], bp['buffer'],
                             bp['min_range'], bp['max_range'],
                             bp['deadline_h'], bp['force_close_h'],
                             bp['use_trend'])
    opt_test  = run_backtest(highs, lows, closes, hours, ema_slopes, test_days,
                             bp['range_start_h'], bp['range_end_h'],
                             bp['rr'], bp['buffer'],
                             bp['min_range'], bp['max_range'],
                             bp['deadline_h'], bp['force_close_h'],
                             bp['use_trend'])
    opt_all   = run_backtest(highs, lows, closes, hours, ema_slopes, days,
                             bp['range_start_h'], bp['range_end_h'],
                             bp['rr'], bp['buffer'],
                             bp['min_range'], bp['max_range'],
                             bp['deadline_h'], bp['force_close_h'],
                             bp['use_trend'])

    print_metrics(opt_train, 'Train (2022-2023) — OPTIMIZED')
    print_metrics(opt_test,  'Test  (2024)      — OUT-OF-SAMPLE')
    print_metrics(opt_all,   'All   (2022-2024) — FULL PERIOD')

    # ══════════════════════════════════════════════════════════════════════
    #  TOP 10 TRIALS
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print('  TOP 10 PARAMETER COMBINATIONS (by t-stat on train)')
    print(f'{"="*72}\n')

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > -900]
    trials_df = trials_df.sort_values('value', ascending=False).head(10)

    param_cols = [c for c in trials_df.columns if c.startswith('params_')]

    print(f'  {"#":>2s}  {"t_stat":>6s}  {"RngS":>4s}  {"RngE":>4s}  '
          f'{"RR":>4s}  {"Buf":>5s}  {"MinR":>5s}  {"MaxR":>5s}  '
          f'{"DL":>3s}  {"CL":>3s}  {"Trend":>5s}')
    print(f'  {"─"*65}')
    for rank, (_, row) in enumerate(trials_df.iterrows(), 1):
        trend_str = 'Yes' if row.get('params_use_trend', False) else 'No'
        print(f'  {rank:2d}  {row["value"]:6.2f}  '
              f'{int(row["params_range_start_h"]):4d}  '
              f'{int(row["params_range_end_h"]):4d}  '
              f'{row["params_rr"]:4.1f}  '
              f'{row["params_buffer"]:5.2f}  '
              f'{row["params_min_range"]:5.1f}  '
              f'{row["params_max_range"]:5.0f}  '
              f'{int(row["params_deadline_h"]):3d}  '
              f'{int(row["params_force_close_h"]):3d}  '
              f'{trend_str:>5s}')

    # ── Validate top-10 on test ──
    print(f'\n{"─"*72}')
    print('  TOP 10 — OUT-OF-SAMPLE VALIDATION (2024)')
    print(f'{"─"*72}\n')

    print(f'  {"#":>2s}  {"t_train":>7s}  │  '
          f'{"t_test":>7s}  {"Trades":>6s}  {"WR%":>5s}  {"P&L":>9s}  {"PF":>5s}  {"MaxDD":>8s}')
    print(f'  {"─"*70}')

    for rank, (_, row) in enumerate(trials_df.iterrows(), 1):
        test_m = run_backtest(
            highs, lows, closes, hours, ema_slopes, test_days,
            int(row['params_range_start_h']), int(row['params_range_end_h']),
            row['params_rr'], row['params_buffer'],
            row['params_min_range'], row['params_max_range'],
            int(row['params_deadline_h']), int(row['params_force_close_h']),
            row.get('params_use_trend', False),
        )
        if test_m is None:
            print(f'  {rank:2d}  {row["value"]:7.2f}  │  no trades on test')
            continue

        print(f'  {rank:2d}  {row["value"]:7.2f}  │  '
              f'{test_m["t_stat"]:7.2f}  {test_m["trades"]:6d}  '
              f'{test_m["wr"]:5.1f}  ${test_m["total_pnl"]:>8,.2f}  '
              f'{test_m["pf"]:5.2f}  ${test_m["max_dd"]:>7,.2f}')

    # ── Compare with Asian Breakout ──
    print(f'\n{"="*72}')
    print('  LONDON BREAKOUT vs ASIAN BREAKOUT (optimized)')
    print(f'{"="*72}')
    print(f'  Asian Breakout best (from optimize_breakout.py):')
    print(f'    RR=3.75, Buf=0.80, Range=7-12, DL=17, Trend=Yes')
    print(f'    Full period: $516 P&L, PF=1.51, t=2.52')
    if opt_all:
        print(f'  London Breakout best:')
        print(f'    Full period: ${opt_all["total_pnl"]:,.2f} P&L, '
              f'PF={opt_all["pf"]}, t={opt_all["t_stat"]}')

    # ── Summary ──
    total_time = time.time() - t0
    print(f'\n{"="*72}')
    print(f'  Total time: {total_time:.1f}s')
    print(f'{"="*72}')
