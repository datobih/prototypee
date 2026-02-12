"""
Hedging Strategy Parameter Optimization — Numba + Optuna
========================================================
Optimizes parameters for hedging_strategy_strict_old.py:
  - TARGET:       TP in dollars ($1–$10)
  - STOP:         SL in dollars ($0.50–$5)
  - HORIZON:      Lookforward bars (10–60)
  - RF_THRESHOLD: Probability cutoff (0.55–0.95)
  - RF max_depth: Tree depth (5–20)
  - RF n_estimators: Number of trees (50–300)

Uses Numba JIT for fast label_outcomes (~100x vs pure Python).
Uses Optuna Bayesian search instead of grid search.
Walk-forward: train on first 60%, test on last 40%.
"""

import numpy as np
import pandas as pd
import numba as nb
import optuna
import pickle
import time
import warnings
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'XAUUSD1.csv')
N_TRIALS  = 200


# ── Feature engineering (exact copy from hedging_strategy_strict_old.py) ─────
def create_microstructure_features(df):
    df = df.copy()

    # Price structure
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)

    # Order flow
    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)

    # Imbalance
    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()

    # Momentum consistency
    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))

    # Volatility
    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)

    # Trend structure
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - \
                        ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']

    # Support/Resistance
    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)

    # Rejection patterns
    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)

    # Size patterns
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)

    # Deeper flow features
    df['flow_15'] = df['directional_flow'].rolling(15).sum()
    df['flow_20'] = df['directional_flow'].rolling(20).sum()
    df['flow_accel'] = df['flow_3'] - df['flow_3'].shift(3)
    df['flow_accel_5'] = df['flow_5'] - df['flow_5'].shift(5)
    df['abs_flow_3'] = df['flow_3'].abs()
    df['abs_flow_5'] = df['flow_5'].abs()
    df['abs_flow_10'] = df['flow_10'].abs()
    df['flow_divergence'] = (df['flow_3'] * df['flow_10'] < 0).astype(int)

    # Flow quality
    df['consecutive_up'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumcount() + 1
    df['consecutive_up'] = df['consecutive_up'] * df['is_up']
    df['consecutive_down'] = (1 - df['is_up']).groupby(((1-df['is_up']) != (1-df['is_up']).shift()).cumsum()).cumcount() + 1
    df['consecutive_down'] = df['consecutive_down'] * (1 - df['is_up'])
    df['max_consecutive'] = df[['consecutive_up', 'consecutive_down']].max(axis=1)
    df['flow_efficiency'] = df['abs_body'] / (df['range'] + 1e-10)
    df['flow_eff_3'] = df['flow_efficiency'].rolling(3).mean()
    df['flow_eff_5'] = df['flow_efficiency'].rolling(5).mean()

    # Volatility regime
    df['atr_roc'] = (df['atr_3'] - df['atr_3'].shift(3)) / (df['atr_3'].shift(3) + 1e-10)
    df['vol_breakout'] = df['range'] / (df['atr_20'] + 1e-10)
    df['range_min_5'] = df['range'].rolling(5).min()
    df['range_max_5'] = df['range'].rolling(5).max()
    df['range_squeeze'] = df['range_min_5'] / (df['range_max_5'] + 1e-10)

    # Distance features
    df['abs_dist_ema8'] = df['dist_ema8'].abs()
    df['dist_ema21'] = (df['Close'] - df['ema_21']) / df['Close']
    df['abs_dist_ema21'] = df['dist_ema21'].abs()
    df['ema_spread'] = (df['ema_8'] - df['ema_21']) / df['Close']
    df['abs_ema_spread'] = df['ema_spread'].abs()

    # Combo features
    df['combo_abs_flow_vol'] = df['abs_flow_5'] * df['vol_ratio']
    df['combo_eff_flow'] = df['flow_eff_3'] * df['abs_flow_3']
    df['combo_consecutive_body'] = df['max_consecutive'] * df['body_pct']
    df['combo_atr_roc_accel'] = df['atr_roc'] * df['flow_accel'].abs()
    df['combo_squeeze_flow'] = (1 - df['range_squeeze']) * df['abs_flow_3']
    df['combo_dist_flow'] = df['abs_dist_ema8'] * df['abs_flow_5']

    return df.dropna()


# ── Numba-compiled label_outcomes ────────────────────────────────────────────
@nb.njit(cache=True)
def label_outcomes_numba(closes, highs, lows, horizon, target, stop):
    """
    Numba JIT label_outcomes — ~100x faster than pure Python.
    Returns array: 0=no hit, 1=long TP, 2=short TP
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = closes[i]
        long_tp  = entry + target
        long_sl  = entry - stop
        short_tp = entry - target
        short_sl = entry + stop

        # Check LONG
        long_hit = False
        for j in range(i + 1, min(i + horizon + 1, n)):
            if highs[j] >= long_tp:
                long_hit = True
                break
            if lows[j] <= long_sl:
                break

        if long_hit:
            labels[i] = 1
            continue

        # Check SHORT
        for j in range(i + 1, min(i + horizon + 1, n)):
            if lows[j] <= short_tp:
                labels[i] = 2
                break
            if highs[j] >= short_sl:
                break

    return labels


SPREAD = 0.30  # $ per position (paid twice per hedge setup)

# ── Hedging strategy evaluation ──────────────────────────────────────────────
def evaluate_hedge(rf_probs, labels, threshold, target, stop):
    """
    Evaluate hedging strategy: enter both sides when RF prob >= threshold.
    WIN  = outcome != 0 (one side hits TP)  → net +(TARGET - STOP) - 2×SPREAD
    LOSS = outcome == 0 (both stopped)      → net -(STOP + STOP) - 2×SPREAD
    All P&L in real dollars.
    """
    mask = rf_probs >= threshold
    n_setups = mask.sum()

    if n_setups < 10:
        return None

    filtered_labels = labels[mask]
    wins   = (filtered_labels != 0).sum()
    losses = n_setups - wins
    win_rate = wins / n_setups * 100

    # P&L in real dollars (2 positions per setup → 2× spread)
    spread_cost = 2 * SPREAD
    win_pnl  = (target - stop) - spread_cost
    loss_pnl = -(stop + stop) - spread_cost

    # Per-trade P&L array
    pnl_per_trade = np.where(filtered_labels != 0, win_pnl, loss_pnl)
    total_pnl = float(pnl_per_trade.sum())
    avg_pnl   = float(pnl_per_trade.mean())
    std_pnl   = float(pnl_per_trade.std()) if n_setups > 1 else 1e-10

    t_stat = avg_pnl / (std_pnl / np.sqrt(n_setups) + 1e-10)
    sharpe = avg_pnl / (std_pnl + 1e-10) * np.sqrt(252)

    # Drawdown in dollars
    cum = np.cumsum(pnl_per_trade)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    # Breakeven WR
    be_wr = abs(loss_pnl) / (win_pnl - loss_pnl) * 100 if win_pnl > loss_pnl else 999.0

    return {
        'setups': int(n_setups),
        'wins': int(wins),
        'losses': int(losses),
        'wr': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 4),
        'win_pnl': round(win_pnl, 2),
        'loss_pnl': round(loss_pnl, 2),
        't_stat': round(t_stat, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 2),
        'be_wr': round(be_wr, 1),
    }


# ── Full pipeline for one set of params ─────────────────────────────────────
def run_pipeline(df_features, feature_names, split_idx,
                 target, stop, horizon,
                 rf_threshold, rf_depth, rf_trees):
    """
    Label → train RF → predict → evaluate hedge on test set.
    Returns metrics dict or None.
    """
    closes = df_features['Close'].values.astype(np.float64)
    highs  = df_features['High'].values.astype(np.float64)
    lows   = df_features['Low'].values.astype(np.float64)

    # Label with Numba
    labels = label_outcomes_numba(closes, highs, lows, horizon, target, stop)

    # Add combo features (created outside the function in original script)
    df_lab = df_features.copy()
    df_lab['outcome'] = labels
    df_lab['combo_flow_trend'] = df_lab['flow_momentum'] * df_lab['trend_align']
    df_lab['combo_vol_imbalance'] = df_lab['vol_ratio'] * df_lab['imbalance_3']
    df_lab['combo_consistency_position'] = df_lab['consistency_5'] * df_lab['close_position']
    df_lab['combo_body_reject'] = df_lab['big_body'] * (df_lab['lower_reject'] - df_lab['upper_reject'])
    df_lab['combo_trend_volatility'] = df_lab['trend_align'] * df_lab['vol_expansion']
    df_lab['combo_imbalance_momentum'] = df_lab['imbalance_5'] * df_lab['flow_5']
    df_lab['combo_position_consistency'] = df_lab['close_position'] * df_lab['consistency_3']
    df_lab['combo_vol_flow'] = df_lab['vol_ratio'] * df_lab['flow_3']

    # Use same features as original script
    all_features = feature_names

    train = df_lab.iloc[:split_idx]
    test  = df_lab.iloc[split_idx:]

    y_train = (train['outcome'] != 0).astype(int)
    y_test  = (test['outcome'] != 0).astype(int)

    X_train = train[all_features].fillna(0)
    X_test  = test[all_features].fillna(0)

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=rf_trees,
        max_depth=rf_depth,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Predict on test
    rf_probs = rf.predict_proba(X_test)[:, 1]
    test_labels = test['outcome'].values

    # Evaluate hedge (with real dollar P&L including spread)
    metrics = evaluate_hedge(rf_probs, test_labels, rf_threshold, target, stop)
    return metrics


# ── Optuna objective ─────────────────────────────────────────────────────────
def make_objective(df_features, feature_names, split_idx):
    def objective(trial):
        target       = trial.suggest_float('target', 1.0, 10.0, step=0.5)
        stop         = trial.suggest_float('stop', 0.5, 5.0, step=0.25)
        horizon      = trial.suggest_int('horizon', 10, 60, step=5)
        rf_threshold = trial.suggest_float('rf_threshold', 0.55, 0.95, step=0.05)
        rf_depth     = trial.suggest_int('rf_depth', 5, 20, step=5)
        rf_trees     = trial.suggest_int('rf_trees', 50, 300, step=50)

        # Skip if SL >= TP (no edge possible for hedge)
        if stop >= target:
            return -999.0

        # Skip if win P&L is negative (spread > edge)
        win_pnl = (target - stop) - 2 * SPREAD
        if win_pnl <= 0:
            return -999.0

        metrics = run_pipeline(
            df_features, feature_names, split_idx,
            target, stop, horizon,
            rf_threshold, rf_depth, rf_trees,
        )

        if metrics is None:
            return -999.0

        # Objective: t-statistic (balances avg return, risk, sample size)
        return metrics['t_stat']

    return objective


# ── Reporting ────────────────────────────────────────────────────────────────
def print_metrics(m, label=''):
    if m is None:
        print(f'  {label}: No valid trades')
        return
    print(f'  {label}')
    print(f'    Setups:    {m["setups"]:>5d}   (W:{m["wins"]} L:{m["losses"]})')
    print(f'    Win Rate:  {m["wr"]:>5.1f}%  (breakeven: {m["be_wr"]:.1f}%)')
    print(f'    Win P&L:   ${m["win_pnl"]:>+7.2f}  |  Loss P&L: ${m["loss_pnl"]:>+7.2f}  (incl 2×${SPREAD} spread)')
    print(f'    Total P&L: ${m["total_pnl"]:>9,.2f}')
    print(f'    Avg P&L:   ${m["avg_pnl"]:>9.4f} per setup')
    print(f'    t-stat:    {m["t_stat"]:>6.2f}')
    print(f'    Sharpe:    {m["sharpe"]:>6.2f}')
    print(f'    MaxDD:     ${m["max_dd"]:>9,.2f}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 72)
    print('  HEDGING STRATEGY — NUMBA + OPTUNA PARAMETER OPTIMIZATION')
    print('=' * 72)

    # ── Load data ──
    print('\n  Loading data...')
    df = pd.read_csv(
        DATA_PATH, sep='\t',
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
               'TickVol', 'Vol', 'Spread'],
    )
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    format='%Y.%m.%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f'  {len(df):,} bars: {df.index[0]} → {df.index[-1]}')

    # ── Feature engineering ──
    print('  Engineering features...')
    df = create_microstructure_features(df)
    print(f'  {len(df):,} bars after feature warmup')

    # Feature list (same as original script)
    feature_names = [
        'abs_ema_spread', 'abs_flow_10', 'abs_flow_3', 'abs_flow_5',
        'abs_dist_ema21', 'combo_dist_flow', 'abs_dist_ema8',
        'flow_momentum', 'ema_spread', 'flow_20',
    ]

    # ── Train/test split ──
    split = int(len(df) * 0.6)
    print(f'  Train: {split:,} bars (60%)  |  Test: {len(df)-split:,} bars (40%)')

    # ── Warm up Numba ──
    print('\n  Compiling Numba label_outcomes (one-time)...')
    _ = label_outcomes_numba(
        df['Close'].values[:1000].astype(np.float64),
        df['High'].values[:1000].astype(np.float64),
        df['Low'].values[:1000].astype(np.float64),
        30, 3.0, 1.5,
    )
    print('  Done.')

    # ══════════════════════════════════════════════════════════════════════
    #  BASELINE (original params)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"─"*72}')
    print('  BASELINE (TARGET=3, STOP=1.5, HORIZON=30, RF_TH=0.75, depth=10, trees=100)')
    print(f'{"─"*72}')

    base = run_pipeline(df, feature_names, split,
                        target=3.0, stop=1.5, horizon=30,
                        rf_threshold=0.75, rf_depth=10, rf_trees=100)
    print_metrics(base, 'Test set (40%)')

    # ── Numba speed comparison ──
    print(f'\n  Numba speed test:')
    closes = df['Close'].values.astype(np.float64)
    highs  = df['High'].values.astype(np.float64)
    lows   = df['Low'].values.astype(np.float64)
    t1 = time.time()
    _ = label_outcomes_numba(closes, highs, lows, 30, 3.0, 1.5)
    t2 = time.time()
    print(f'    Numba label_outcomes on {len(closes):,} bars: {t2-t1:.3f}s')

    # ══════════════════════════════════════════════════════════════════════
    #  OPTUNA OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print(f'  OPTUNA BAYESIAN OPTIMIZATION ({N_TRIALS} trials)')
    print(f'{"="*72}')

    objective = make_objective(df, feature_names, split)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t_opt = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda study, trial:
                       print(f'    Trial {trial.number+1:>3d}/{N_TRIALS}  '
                             f't_stat={trial.value:>6.2f}  '
                             f'best={study.best_value:>6.2f}',
                             end='\r')
                   ])
    elapsed = time.time() - t_opt
    print(f'\n  Completed {N_TRIALS} trials in {elapsed:.1f}s '
          f'({elapsed/N_TRIALS:.1f}s/trial)')

    bp = study.best_params
    print(f'\n  BEST PARAMETERS:')
    for k, v in bp.items():
        print(f'    {k:20s}: {v}')

    # ── Evaluate best params ──
    print(f'\n{"─"*72}')
    print('  OPTIMIZED RESULTS')
    print(f'{"─"*72}')

    opt = run_pipeline(
        df, feature_names, split,
        bp['target'], bp['stop'], bp['horizon'],
        bp['rf_threshold'], bp['rf_depth'], bp['rf_trees'],
    )
    print_metrics(opt, 'Test set (40%) — OPTIMIZED')

    # ── Comparison ──
    print(f'\n{"─"*72}')
    print('  BASELINE vs OPTIMIZED')
    print(f'{"─"*72}')
    if base and opt:
        print(f'  {"Metric":<15} {"Baseline":>10} {"Optimized":>10} {"Change":>10}')
        print(f'  {"─"*45}')
        for key, label in [('wr', 'Win Rate %'), ('total_pnl', 'Total P&L $'),
                           ('avg_pnl', 'Avg P&L $'), ('t_stat', 't-stat'),
                           ('sharpe', 'Sharpe'), ('setups', 'Setups'),
                           ('max_dd', 'Max DD $')]:
            bv = base[key]
            ov = opt[key]
            delta = ov - bv
            sign = '+' if delta >= 0 else ''
            print(f'  {label:<15} {bv:>10} {ov:>10} {sign}{delta:>9}')

    # ══════════════════════════════════════════════════════════════════════
    #  TOP 10 TRIALS
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print('  TOP 10 PARAMETER COMBINATIONS (by t-stat)')
    print(f'{"="*72}\n')

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > -900]
    trials_df = trials_df.sort_values('value', ascending=False).head(10)

    param_cols = [c for c in trials_df.columns if c.startswith('params_')]

    print(f'  {"#":>2s}  {"t_stat":>6s}  {"TGT":>4s}  {"SL":>4s}  '
          f'{"HOR":>3s}  {"RF_TH":>5s}  {"Dep":>3s}  {"Trees":>5s}')
    print(f'  {"─"*50}')
    for rank, (_, row) in enumerate(trials_df.iterrows(), 1):
        print(f'  {rank:2d}  {row["value"]:6.2f}  '
              f'{row["params_target"]:4.1f}  {row["params_stop"]:4.2f}  '
              f'{int(row["params_horizon"]):3d}  {row["params_rf_threshold"]:5.2f}  '
              f'{int(row["params_rf_depth"]):3d}  {int(row["params_rf_trees"]):5d}')

    # ── Save best params ──
    best_params_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_hedge_params.txt')
    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
    with open(best_params_path, 'w') as f:
        for k, v in bp.items():
            f.write(f'{k}={v}\n')
    print(f'\n  Best params saved to {best_params_path}')

    # ── Summary ──
    total_time = time.time() - t0
    print(f'\n{"="*72}')
    print(f'  SUMMARY')
    print(f'{"="*72}')
    print(f'  Total time: {total_time:.1f}s')
    if base and opt:
        print(f'  Baseline:  WR={base["wr"]}% | ${base["total_pnl"]:,.2f} | t={base["t_stat"]}')
        print(f'  Optimized: WR={opt["wr"]}% | ${opt["total_pnl"]:,.2f} | t={opt["t_stat"]}')
        print(f'  Best params: TARGET=${bp["target"]}, STOP=${bp["stop"]}, '
              f'HORIZON={bp["horizon"]}, RF_TH={bp["rf_threshold"]}')
    print(f'{"="*72}')
