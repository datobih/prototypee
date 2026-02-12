"""
XAUUSD Squeeze Momentum — Numba + Optuna Optimization
======================================================
Strategy: Bollinger Bands go inside Keltner Channel = "squeeze" (low vol).
          When squeeze releases (BB expands outside KC), enter in momentum
          direction. The idea: compression → explosion.

Indicators (precomputed in pandas, passed to Numba):
  - Bollinger Bands: SMA ± bb_std * StdDev
  - Keltner Channel: EMA ± kc_mult * ATR
  - Squeeze = BB inside KC (lower_bb > lower_kc AND upper_bb < upper_kc)
  - Momentum: linear regression value of close over mom_len bars

Params to optimize:
  - bb_len:       Bollinger period (10–30)
  - bb_std:       Bollinger std multiplier (1.0–3.0)
  - kc_len:       Keltner EMA period (10–30)
  - kc_mult:      Keltner ATR multiplier (1.0–3.0)
  - mom_len:      Momentum lookback (5–20)
  - atr_sl_mult:  SL = ATR × mult (1.0–4.0)
  - rr:           Reward-to-risk ratio (1.0–5.0)
  - hold_bars:    Max bars to hold (30–600)
  - min_squeeze:  Minimum consecutive squeeze bars before release (3–20)
  - session_filter: Trade only during active sessions? (True/False)

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
    print(f'  {len(df):,} bars loaded')
    return df


# ── Indicator computation ────────────────────────────────────────────────────
def compute_indicators(df, bb_len, bb_std, kc_len, kc_mult, mom_len):
    """Compute BB, KC, squeeze state, and momentum for given params."""
    d = df.copy()

    # Bollinger Bands
    d['bb_mid'] = d['Close'].rolling(bb_len).mean()
    d['bb_sd']  = d['Close'].rolling(bb_len).std()
    d['bb_upper'] = d['bb_mid'] + bb_std * d['bb_sd']
    d['bb_lower'] = d['bb_mid'] - bb_std * d['bb_sd']

    # ATR for Keltner
    d['tr'] = np.maximum(
        d['High'] - d['Low'],
        np.maximum(
            abs(d['High'] - d['Close'].shift(1)),
            abs(d['Low'] - d['Close'].shift(1)),
        )
    )
    d['atr'] = d['tr'].rolling(kc_len).mean()

    # Keltner Channel
    d['kc_mid']   = d['Close'].ewm(span=kc_len).mean()
    d['kc_upper'] = d['kc_mid'] + kc_mult * d['atr']
    d['kc_lower'] = d['kc_mid'] - kc_mult * d['atr']

    # Squeeze: BB inside KC
    d['squeeze'] = ((d['bb_lower'] > d['kc_lower']) &
                    (d['bb_upper'] < d['kc_upper'])).astype(int)

    # Momentum: simple difference over mom_len (approximation of LinReg value)
    d['momentum'] = d['Close'] - d['Close'].shift(mom_len)

    d = d.dropna().copy()
    return d


# ── Precompute day boundaries ────────────────────────────────────────────────
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
    return pd.DataFrame(rows)


# ── Numba-compiled squeeze backtest ──────────────────────────────────────────
@nb.njit(cache=True)
def _fast_squeeze(highs, lows, closes, hours,
                  squeeze_arr, momentum_arr, atr_arr,
                  atr_sl_mult, rr, hold_bars, min_squeeze,
                  session_filter, spread):
    """
    Numba JIT Squeeze Momentum backtest.
    Scans bar-by-bar for squeeze release signals:
      1. Squeeze was ON for >= min_squeeze consecutive bars
      2. Squeeze turns OFF (release)
      3. Enter in direction of momentum at release bar
      4. SL = ATR × mult, TP = SL × rr
      5. Exit at SL, TP, or hold_bars timeout
    """
    n = len(highs)
    max_trades = n // 10  # upper bound
    pnl_arr = np.empty(max_trades, dtype=np.float64)
    out_arr = np.empty(max_trades, dtype=np.int32)
    dir_arr = np.empty(max_trades, dtype=np.int32)
    count = 0

    consec_squeeze = 0
    in_trade = False
    trade_ep = 0.0
    trade_sl = 0.0
    trade_tp = 0.0
    trade_dir = 0
    trade_start = 0

    for i in range(1, n):
        # Track consecutive squeeze bars
        if squeeze_arr[i] == 1:
            consec_squeeze += 1
        else:
            # ── Check for release ──
            if (consec_squeeze >= min_squeeze and
                    not in_trade and
                    squeeze_arr[i] == 0):

                # Session filter: only trade 08:00–20:00 broker time
                if session_filter:
                    h = hours[i]
                    if h < 8 or h > 20:
                        consec_squeeze = 0
                        continue

                mom = momentum_arr[i]
                atr = atr_arr[i]

                if atr < 0.01 or abs(mom) < 0.01:
                    consec_squeeze = 0
                    continue

                # Direction from momentum
                if mom > 0:
                    ddir = 1
                else:
                    ddir = -1

                ep = closes[i]
                sl_dist = atr * atr_sl_mult

                if ddir == 1:
                    sl = ep - sl_dist
                    tp = ep + sl_dist * rr
                else:
                    sl = ep + sl_dist
                    tp = ep - sl_dist * rr

                in_trade = True
                trade_ep = ep
                trade_sl = sl
                trade_tp = tp
                trade_dir = ddir
                trade_start = i

            consec_squeeze = 0

        # ── Track open trade ──
        if in_trade:
            bars_held = i - trade_start

            if bars_held >= hold_bars:
                # Timeout
                pnl_arr[count] = (closes[i] - trade_ep) * trade_dir - spread
                out_arr[count] = 3
                dir_arr[count] = trade_dir
                count += 1
                in_trade = False
                continue

            if trade_dir == 1:
                if lows[i] <= trade_sl:
                    pnl_arr[count] = (trade_sl - trade_ep) * 1 - spread
                    out_arr[count] = 2
                    dir_arr[count] = 1
                    count += 1
                    in_trade = False
                    continue
                if highs[i] >= trade_tp:
                    pnl_arr[count] = (trade_tp - trade_ep) * 1 - spread
                    out_arr[count] = 1
                    dir_arr[count] = 1
                    count += 1
                    in_trade = False
                    continue
            else:
                if highs[i] >= trade_sl:
                    pnl_arr[count] = (trade_sl - trade_ep) * -1 - spread
                    out_arr[count] = 2
                    dir_arr[count] = -1
                    count += 1
                    in_trade = False
                    continue
                if lows[i] <= trade_tp:
                    pnl_arr[count] = (trade_tp - trade_ep) * -1 - spread
                    out_arr[count] = 1
                    dir_arr[count] = -1
                    count += 1
                    in_trade = False
                    continue

    return pnl_arr[:count], out_arr[:count], dir_arr[:count]


# ── Python wrapper ───────────────────────────────────────────────────────────
def run_backtest(df, bb_len, bb_std, kc_len, kc_mult, mom_len,
                 atr_sl_mult, rr, hold_bars, min_squeeze,
                 session_filter, year_mask=None, spread=SPREAD_COST):
    """Compute indicators and run Numba backtest."""
    d = compute_indicators(df, bb_len, bb_std, kc_len, kc_mult, mom_len)

    if year_mask is not None:
        d = d[d.index.year.isin(year_mask)].copy()

    if len(d) < 500:
        return None

    pnl, outcomes, dirs = _fast_squeeze(
        d['High'].values.astype(np.float64),
        d['Low'].values.astype(np.float64),
        d['Close'].values.astype(np.float64),
        d.index.hour.values.astype(np.int64),
        d['squeeze'].values.astype(np.int64),
        d['momentum'].values.astype(np.float64),
        d['atr'].values.astype(np.float64),
        float(atr_sl_mult), float(rr), int(hold_bars), int(min_squeeze),
        bool(session_filter), float(spread),
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
def make_objective(df, train_years):
    def objective(trial):
        bb_len      = trial.suggest_int('bb_len', 10, 30, step=5)
        bb_std      = trial.suggest_float('bb_std', 1.0, 3.0, step=0.25)
        kc_len      = trial.suggest_int('kc_len', 10, 30, step=5)
        kc_mult     = trial.suggest_float('kc_mult', 1.0, 3.0, step=0.25)
        mom_len     = trial.suggest_int('mom_len', 5, 20, step=5)
        atr_sl      = trial.suggest_float('atr_sl_mult', 1.0, 4.0, step=0.25)
        rr          = trial.suggest_float('rr', 1.0, 5.0, step=0.25)
        hold_bars   = trial.suggest_int('hold_bars', 30, 600, step=30)
        min_squeeze = trial.suggest_int('min_squeeze', 3, 20, step=1)
        sess_filter = trial.suggest_categorical('session_filter', [True, False])

        result = run_backtest(
            df, bb_len, bb_std, kc_len, kc_mult, mom_len,
            atr_sl, rr, hold_bars, min_squeeze, sess_filter,
            year_mask=train_years,
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
    print('  XAUUSD SQUEEZE MOMENTUM — NUMBA + OPTUNA OPTIMIZATION')
    print('=' * 72)

    df = load_data(DATA_PATH)

    train_years = [2022, 2023]
    test_years  = [2024]
    all_years   = [2022, 2023, 2024]

    # ── Warm up Numba ──
    print('\n  Compiling Numba engine...')
    _ = run_backtest(df, 20, 2.0, 20, 1.5, 10, 2.0, 2.0, 120, 5, False,
                     year_mask=train_years)
    print('  Done.\n')

    # ── Baselines ──
    print(f'{"─"*72}')
    print('  BASELINE (BB=20/2.0, KC=20/1.5, Mom=10, ATR_SL=2.0, RR=2.0)')
    print(f'{"─"*72}')
    for label, yrs in [('Train', train_years), ('Test', test_years), ('All', all_years)]:
        m = run_backtest(df, 20, 2.0, 20, 1.5, 10, 2.0, 2.0, 120, 5, False,
                         year_mask=yrs)
        print_metrics(m, f'{label}')

    print(f'\n{"─"*72}')
    print('  BASELINE + SESSION FILTER (08:00–20:00)')
    print(f'{"─"*72}')
    for label, yrs in [('Train', train_years), ('Test', test_years), ('All', all_years)]:
        m = run_backtest(df, 20, 2.0, 20, 1.5, 10, 2.0, 2.0, 120, 5, True,
                         year_mask=yrs)
        print_metrics(m, f'{label}')

    # ── Tighter squeeze baseline ──
    print(f'\n{"─"*72}')
    print('  TIGHT SQUEEZE (BB=20/2.0, KC=20/1.0, Mom=10, MinSq=10)')
    print(f'{"─"*72}')
    for label, yrs in [('Train', train_years), ('Test', test_years), ('All', all_years)]:
        m = run_backtest(df, 20, 2.0, 20, 1.0, 10, 2.0, 2.0, 120, 10, True,
                         year_mask=yrs)
        print_metrics(m, f'{label}')

    # ══════════════════════════════════════════════════════════════════════
    #  OPTUNA OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print(f'  OPTUNA BAYESIAN OPTIMIZATION ({N_TRIALS} trials on train set)')
    print(f'{"="*72}')

    objective = make_objective(df, train_years)
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

    for label, yrs in [('Train (2022-2023) — OPTIMIZED', train_years),
                        ('Test  (2024)      — OUT-OF-SAMPLE', test_years),
                        ('All   (2022-2024) — FULL PERIOD', all_years)]:
        m = run_backtest(
            df, bp['bb_len'], bp['bb_std'], bp['kc_len'], bp['kc_mult'],
            bp['mom_len'], bp['atr_sl_mult'], bp['rr'], bp['hold_bars'],
            bp['min_squeeze'], bp['session_filter'],
            year_mask=yrs,
        )
        print_metrics(m, label)

    # ── Get test metrics for comparison ──
    opt_test = run_backtest(
        df, bp['bb_len'], bp['bb_std'], bp['kc_len'], bp['kc_mult'],
        bp['mom_len'], bp['atr_sl_mult'], bp['rr'], bp['hold_bars'],
        bp['min_squeeze'], bp['session_filter'],
        year_mask=test_years,
    )
    opt_all = run_backtest(
        df, bp['bb_len'], bp['bb_std'], bp['kc_len'], bp['kc_mult'],
        bp['mom_len'], bp['atr_sl_mult'], bp['rr'], bp['hold_bars'],
        bp['min_squeeze'], bp['session_filter'],
        year_mask=all_years,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  TOP 15 TRIALS
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print('  TOP 15 PARAMETER COMBINATIONS (by t-stat on train)')
    print(f'{"="*72}\n')

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > -900]
    trials_df = trials_df.sort_values('value', ascending=False).head(15)

    param_cols = [c for c in trials_df.columns if c.startswith('params_')]

    print(f'  {"#":>2s}  {"t":>5s}  {"BBl":>3s}  {"BBs":>4s}  {"KCl":>3s}  {"KCs":>4s}  '
          f'{"Mom":>3s}  {"SLm":>4s}  {"RR":>4s}  {"Hold":>4s}  {"MinS":>4s}  {"Sess":>4s}')
    print(f'  {"─"*60}')
    for rank, (_, row) in enumerate(trials_df.iterrows(), 1):
        sess_str = 'Yes' if row.get('params_session_filter', False) else 'No'
        print(f'  {rank:2d}  {row["value"]:5.2f}  '
              f'{int(row["params_bb_len"]):3d}  {row["params_bb_std"]:4.1f}  '
              f'{int(row["params_kc_len"]):3d}  {row["params_kc_mult"]:4.1f}  '
              f'{int(row["params_mom_len"]):3d}  {row["params_atr_sl_mult"]:4.1f}  '
              f'{row["params_rr"]:4.1f}  {int(row["params_hold_bars"]):4d}  '
              f'{int(row["params_min_squeeze"]):4d}  {sess_str:>4s}')

    # ── Validate top-10 on test ──
    print(f'\n{"─"*72}')
    print('  TOP 10 — OUT-OF-SAMPLE VALIDATION (2024)')
    print(f'{"─"*72}\n')

    print(f'  {"#":>2s}  {"t_train":>7s}  │  '
          f'{"t_test":>7s}  {"Trades":>6s}  {"WR%":>5s}  {"P&L":>9s}  {"PF":>5s}  {"MaxDD":>8s}')
    print(f'  {"─"*70}')

    for rank, (_, row) in enumerate(trials_df.head(10).iterrows(), 1):
        test_m = run_backtest(
            df, int(row['params_bb_len']), row['params_bb_std'],
            int(row['params_kc_len']), row['params_kc_mult'],
            int(row['params_mom_len']), row['params_atr_sl_mult'],
            row['params_rr'], int(row['params_hold_bars']),
            int(row['params_min_squeeze']),
            row.get('params_session_filter', False),
            year_mask=test_years,
        )
        if test_m is None:
            print(f'  {rank:2d}  {row["value"]:7.2f}  │  no trades')
            continue

        print(f'  {rank:2d}  {row["value"]:7.2f}  │  '
              f'{test_m["t_stat"]:7.2f}  {test_m["trades"]:6d}  '
              f'{test_m["wr"]:5.1f}  ${test_m["total_pnl"]:>8,.2f}  '
              f'{test_m["pf"]:5.2f}  ${test_m["max_dd"]:>7,.2f}')

    # ── Head-to-head ──
    print(f'\n{"="*72}')
    print('  STRATEGY COMPARISON (all optimized, full period)')
    print(f'{"="*72}')
    print(f'  {"Strategy":<25s}  {"P&L":>9s}  {"PF":>5s}  {"t-stat":>6s}  {"OOS t":>6s}  {"MaxDD":>8s}')
    print(f'  {"─"*65}')
    print(f'  {"Asian Breakout":<25s}  {"$516":>9s}  {"1.51":>5s}  {"2.52":>6s}  {"1.53":>6s}  {"$-96":>8s}')
    print(f'  {"London Breakout":<25s}  {"$558":>9s}  {"1.29":>5s}  {"1.91":>6s}  {"0.58":>6s}  {"$-166":>8s}')
    print(f'  {"ORB (NY Pre-Open)":<25s}  {"$402":>9s}  {"1.49":>5s}  {"2.57":>6s}  {"0.77":>6s}  {"$-119":>8s}')
    if opt_all and opt_test:
        print(f'  {"Squeeze Momentum":<25s}  '
              f'${opt_all["total_pnl"]:>8,.2f}  '
              f'{opt_all["pf"]:>5.2f}  '
              f'{opt_all["t_stat"]:>6.2f}  '
              f'{opt_test["t_stat"]:>6.2f}  '
              f'${opt_all["max_dd"]:>7,.2f}')

    total_time = time.time() - t0
    print(f'\n  Total time: {total_time:.1f}s')
    print(f'{"="*72}')
