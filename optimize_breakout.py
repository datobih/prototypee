"""
XAUUSD Asian Breakout — Numba + Optuna Optimization
====================================================
- Numba JIT:  C-speed backtesting (~100x faster than pure Python loops)
- Optuna:     Bayesian parameter search (smarter than grid search)
- Walk-forward: train on 2022-2023, validate on 2024

Optimizes: RR ratio, buffer, range filters, entry deadline, trend filter
Objective: t-statistic of per-trade P&L (balances return, risk, sample size)
"""

import numpy as np
import pandas as pd
import numba as nb
import optuna
import warnings, time

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH       = 'XAUUSD_backtest.csv'
ASIAN_START     = 1          # 01:00 broker time
ASIAN_END       = 9          # 09:00 broker time (fixed — defines the range)
FORCE_CLOSE     = 21         # 21:00 broker time
SPREAD_COST     = 0.30       # $ conservative avg spread
N_TRIALS        = 300        # Optuna trials
TRAIN_END_YEAR  = 2023       # Train: <=2023, Test: 2024


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data(path):
    print(f'  Loading {path}...')
    df = pd.read_csv(
        path, sep='\t',
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
               'TickVol', 'Vol', 'Spread'],
        skiprows=1
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
def precompute_days(df, asian_start=ASIAN_START, asian_end=ASIAN_END):
    """
    For each trading day, precompute:
      - bar index range (start, end)
      - Asian session high/low/range
      - trend direction at Asian close
      - year (for train/test split)
    """
    df = df.copy()
    df['_pos'] = np.arange(len(df))
    df['_hour'] = df.index.hour
    df['_date'] = df.index.date

    rows = []
    for date, grp in df.groupby('_date'):
        asian = grp[(grp['_hour'] >= asian_start) & (grp['_hour'] < asian_end)]
        if len(asian) < 60:
            continue

        ah = asian['High'].max()
        al = asian['Low'].min()
        slope = grp.loc[asian.index[-1], 'ema50_slope']

        rows.append({
            'date':        date,
            'year':        date.year,
            'start':       int(grp['_pos'].iloc[0]),
            'end':         int(grp['_pos'].iloc[-1]) + 1,
            'asian_high':  ah,
            'asian_low':   al,
            'asian_range': ah - al,
            'trend_dir':   1 if slope > 0 else -1,
        })

    days = pd.DataFrame(rows)
    print(f'  {len(days)} valid trading days '
          f'(train ≤{TRAIN_END_YEAR}: {(days["year"] <= TRAIN_END_YEAR).sum()}, '
          f'test: {(days["year"] > TRAIN_END_YEAR).sum()})')
    return days


# ── Numba-compiled backtest ──────────────────────────────────────────────────
@nb.njit(cache=True)
def _fast_backtest(highs, lows, closes, hours,
                   day_starts, day_ends,
                   asian_highs, asian_lows, trend_dirs,
                   rr, buffer, deadline_h, close_h,
                   spread, use_trend_filter):
    """
    Numba JIT-compiled Asian Range Breakout backtest.
    Returns arrays: pnl_per_trade, outcome_per_trade (1=TP, 2=SL, 3=timeout)
    """
    n_days = len(day_starts)
    pnl_arr = np.empty(n_days, dtype=np.float64)
    out_arr = np.empty(n_days, dtype=np.int32)
    count = 0

    for d in range(n_days):
        ah = asian_highs[d]
        al = asian_lows[d]
        td = trend_dirs[d]
        s  = day_starts[d]
        e  = day_ends[d]

        long_lv  = ah + buffer
        short_lv = al - buffer

        # ── Find first breakout in entry window ──
        ebar = -1
        ep   = 0.0
        ddir = 0

        for i in range(s, e):
            h = hours[i]
            if h < 9 or h >= deadline_h:      # Asian ends at 09:00 (fixed)
                continue

            if highs[i] >= long_lv:
                if use_trend_filter and td < 0:
                    pass                       # skip: against trend
                else:
                    ddir = 1
                    ep   = long_lv
                    ebar = i
                    break

            if lows[i] <= short_lv:
                if use_trend_filter and td > 0:
                    pass
                else:
                    ddir = -1
                    ep   = short_lv
                    ebar = i
                    break

        if ebar == -1:
            continue

        # ── SL / TP ──
        if ddir == 1:
            sl = al - buffer
            tp = ep + (ep - sl) * rr
        else:
            sl = ah + buffer
            tp = ep - (sl - ep) * rr

        # ── Track outcome ──
        oc = 3                                 # default: timeout
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
        count += 1

    return pnl_arr[:count], out_arr[:count]


# ── Python wrapper ───────────────────────────────────────────────────────────
def run_backtest(highs, lows, closes, hours, days_df,
                 rr, buffer, min_range, max_range,
                 deadline, use_trend, spread=SPREAD_COST):
    """Filter days by range, call Numba engine, return metrics dict."""
    valid = (
        (days_df['asian_range'].values >= min_range) &
        (days_df['asian_range'].values <= max_range)
    )
    vd = days_df[valid]
    if len(vd) < 10:
        return None

    pnl, outcomes = _fast_backtest(
        highs, lows, closes, hours,
        vd['start'].values.astype(np.int64),
        vd['end'].values.astype(np.int64),
        vd['asian_high'].values,
        vd['asian_low'].values,
        vd['trend_dir'].values.astype(np.int64),
        float(rr), float(buffer),
        int(deadline), int(FORCE_CLOSE),
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

    return {
        'trades': n, 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'wr': round(wr, 1), 'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 3), 'std_pnl': round(std_pnl, 3),
        't_stat': round(t_stat, 2), 'pf': round(pf, 2),
        'max_dd': round(max_dd, 2), 'sharpe': round(avg_pnl / (std_pnl + 1e-10) * np.sqrt(252), 2),
    }


# ── Optuna objective ─────────────────────────────────────────────────────────
def make_objective(highs, lows, closes, hours, train_days):
    """Returns Optuna objective that optimizes t-statistic on train set."""

    def objective(trial):
        rr        = trial.suggest_float('rr', 1.0, 4.0, step=0.25)
        buffer    = trial.suggest_float('buffer', 0.10, 2.00, step=0.10)
        min_range = trial.suggest_float('min_range', 1.0, 10.0, step=0.5)
        max_range = trial.suggest_float('max_range', 12.0, 35.0, step=1.0)
        deadline  = trial.suggest_int('deadline', 13, 19)
        use_trend = trial.suggest_categorical('use_trend_filter', [True, False])

        result = run_backtest(
            highs, lows, closes, hours, train_days,
            rr, buffer, min_range, max_range, deadline, use_trend,
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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 72)
    print('  XAUUSD ASIAN BREAKOUT — NUMBA + OPTUNA OPTIMIZATION')
    print('=' * 72)

    # ── Load & precompute ──
    df   = load_data(DATA_PATH)
    days = precompute_days(df)

    highs  = df['High'].values.astype(np.float64)
    lows   = df['Low'].values.astype(np.float64)
    closes = df['Close'].values.astype(np.float64)
    hours  = df.index.hour.values.astype(np.int64)

    train_days = days[days['year'] <= TRAIN_END_YEAR].copy()
    test_days  = days[days['year'] >  TRAIN_END_YEAR].copy()

    # ── Warm up Numba (first call compiles) ──
    print('\n  Compiling Numba engine (one-time)...')
    _ = run_backtest(highs, lows, closes, hours, train_days,
                     rr=2.0, buffer=0.5, min_range=3.0, max_range=25.0,
                     deadline=17, use_trend=False)
    print('  Done.\n')

    # ── Baseline (default params) ──
    print('─' * 72)
    print('  BASELINE (RR=2.0, Buf=0.50, Range=3-25, Deadline=17, NoTrend)')
    print('─' * 72)
    base_train = run_backtest(highs, lows, closes, hours, train_days,
                              2.0, 0.50, 3.0, 25.0, 17, False)
    base_test  = run_backtest(highs, lows, closes, hours, test_days,
                              2.0, 0.50, 3.0, 25.0, 17, False)
    base_all   = run_backtest(highs, lows, closes, hours, days,
                              2.0, 0.50, 3.0, 25.0, 17, False)
    print_metrics(base_train, 'Train (2022-2023)')
    print_metrics(base_test,  'Test  (2024)')
    print_metrics(base_all,   'All   (2022-2024)')

    # ── Baseline with trend filter ──
    print(f'\n{"─"*72}')
    print('  BASELINE + TREND FILTER')
    print('─' * 72)
    bt_train = run_backtest(highs, lows, closes, hours, train_days,
                            2.0, 0.50, 3.0, 25.0, 17, True)
    bt_test  = run_backtest(highs, lows, closes, hours, test_days,
                            2.0, 0.50, 3.0, 25.0, 17, True)
    bt_all   = run_backtest(highs, lows, closes, hours, days,
                            2.0, 0.50, 3.0, 25.0, 17, True)
    print_metrics(bt_train, 'Train (2022-2023)')
    print_metrics(bt_test,  'Test  (2024)')
    print_metrics(bt_all,   'All   (2022-2024)')

    # ══════════════════════════════════════════════════════════════════════
    #  OPTUNA OPTIMIZATION (train on 2022-2023)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print(f'  OPTUNA BAYESIAN OPTIMIZATION  ({N_TRIALS} trials on train set)')
    print(f'{"="*72}')

    objective = make_objective(highs, lows, closes, hours, train_days)
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

    # ── Evaluate best params on train / test / all ──
    print(f'\n{"─"*72}')
    print('  OPTIMIZED RESULTS')
    print('─' * 72)

    opt_train = run_backtest(highs, lows, closes, hours, train_days,
                             bp['rr'], bp['buffer'], bp['min_range'],
                             bp['max_range'], bp['deadline'],
                             bp['use_trend_filter'])
    opt_test  = run_backtest(highs, lows, closes, hours, test_days,
                             bp['rr'], bp['buffer'], bp['min_range'],
                             bp['max_range'], bp['deadline'],
                             bp['use_trend_filter'])
    opt_all   = run_backtest(highs, lows, closes, hours, days,
                             bp['rr'], bp['buffer'], bp['min_range'],
                             bp['max_range'], bp['deadline'],
                             bp['use_trend_filter'])

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
    display_cols = ['value'] + param_cols
    top10 = trials_df[display_cols].copy()
    top10.columns = [c.replace('params_', '') for c in top10.columns]
    top10 = top10.rename(columns={'value': 't_stat'})
    print(top10.to_string(index=False))

    # ── Validate each top-10 on test set ──
    print(f'\n{"─"*72}')
    print('  TOP 10 — OUT-OF-SAMPLE VALIDATION (2024)')
    print(f'{"─"*72}\n')

    print(f'  {"#":>2s}  {"t_train":>7s}  {"RR":>4s}  {"Buf":>5s}  '
          f'{"MinR":>5s}  {"MaxR":>5s}  {"DL":>3s}  {"Trend":>5s}  │  '
          f'{"t_test":>7s}  {"Trades":>6s}  {"WR%":>5s}  {"P&L":>9s}  {"PF":>5s}')
    print(f'  {"─"*95}')

    for rank, (_, row) in enumerate(trials_df.iterrows(), 1):
        p = {c.replace('params_', ''): row[c] for c in param_cols}
        test_m = run_backtest(
            highs, lows, closes, hours, test_days,
            p['rr'], p['buffer'], p['min_range'], p['max_range'],
            int(p['deadline']), p['use_trend_filter'],
        )
        if test_m is None:
            print(f'  {rank:2d}  {row["value"]:7.2f}  —  no trades on test set')
            continue

        trend_str = 'Yes' if p['use_trend_filter'] else 'No'
        print(f'  {rank:2d}  {row["value"]:7.2f}  {p["rr"]:4.1f}  '
              f'{p["buffer"]:5.2f}  {p["min_range"]:5.1f}  {p["max_range"]:5.0f}  '
              f'{int(p["deadline"]):3d}  {trend_str:>5s}  │  '
              f'{test_m["t_stat"]:7.2f}  {test_m["trades"]:6d}  '
              f'{test_m["wr"]:5.1f}  ${test_m["total_pnl"]:>8,.2f}  '
              f'{test_m["pf"]:5.2f}')

    # ── Summary ──
    total_time = time.time() - t0
    print(f'\n{"="*72}')
    print(f'  SUMMARY')
    print(f'{"="*72}')
    print(f'  Total time: {total_time:.1f}s')
    print(f'  Baseline t-stat (train/test): '
          f'{base_train["t_stat"] if base_train else "N/A"} / '
          f'{base_test["t_stat"] if base_test else "N/A"}')
    if opt_train and opt_test:
        print(f'  Optimized t-stat (train/test): '
              f'{opt_train["t_stat"]} / {opt_test["t_stat"]}')
        delta = (opt_test['t_stat'] - (base_test['t_stat'] if base_test else 0))
        print(f'  Test improvement: {"+" if delta >= 0 else ""}{delta:.2f} t-stat')
    print(f'{"="*72}')
