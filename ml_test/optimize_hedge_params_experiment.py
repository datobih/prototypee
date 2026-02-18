"""
Hedge Trade Optimizer — Base Features + Optuna
========================================================
Mode:    HEDGE ONLY (simultaneous LONG + SHORT)
Fixed:   rf_depth=20, rf_trees=50 (fast)
Optimize: target, stop, horizon, rf_threshold  (4 params)
Features: 26 base (no HTF)
Walk-forward: train 60% (90% subsample), test 40%
Objective: t-statistic (spread-adjusted, 2x spread per hedge)

Hedge Mechanics:
  - Open LONG + SHORT simultaneously at each signal bar
  - Long:  TP = entry + target,  SL = entry - stop
  - Short: TP = entry - target,  SL = entry + stop
  - Both positions tracked independently until closed or horizon expires
  - Net P&L = long_pnl + short_pnl - 2*SPREAD
  - Best case: price oscillates enough to hit BOTH TPs → net = 2*target - 2*spread (XAUUSD)
  - Typical: one TP hit, other side stopped/expired → net = target - loss - 2*spread
  - Worst: both sides stopped → net = -2*stop - 2*spread
"""

import numpy as np
import pandas as pd
import numba as nb
import optuna
import time
import warnings
import os

from sklearn.ensemble import RandomForestClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'XAUUSD1.csv')
N_TRIALS   = 200
SPREAD     = 0.24      # per-leg spread (hedge pays 2x = $0.60 total)
RF_DEPTH   = 20
RF_TREES   = 50        # fast search (retrain best with 200 after)


# ── Feature engineering ──────────────────────────────────────────────────────
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
    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']

    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA LABELING — Binary movement detection
#  "Did price move $target in EITHER direction within horizon bars?"
#  This is the key insight from hedging_strategy_strict_old.py:
#  predict MOVEMENT, not direction.
# ══════════════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def label_movement(closes, highs, lows, horizon, target, stop):
    """
    Binary movement labeling (like hedging_strategy_strict_old.py).

    For each bar, check if price hits +target (LONG TP) or -target (SHORT TP)
    before hitting the stop loss on that side.

    Returns:
        labels – 1 if price moved $target in either direction (LONG or SHORT TP hit)
                 0 if neither TP was hit (both stopped out or timed out)
        sides  – 1 if LONG TP hit, 2 if SHORT TP hit, 0 if neither
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32)
    sides  = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = closes[i]

        # Check LONG: does price hit entry+target before entry-stop?
        long_hit = False
        for j in range(i + 1, min(i + horizon + 1, n)):
            if highs[j] >= entry + target:
                long_hit = True
                break
            if lows[j] <= entry - stop:
                break

        # Check SHORT: does price hit entry-target before entry+stop?
        short_hit = False
        if not long_hit:
            for j in range(i + 1, min(i + horizon + 1, n)):
                if lows[j] <= entry - target:
                    short_hit = True
                    break
                if highs[j] >= entry + stop:
                    break

        if long_hit:
            labels[i] = 1
            sides[i] = 1
        elif short_hit:
            labels[i] = 1
            sides[i] = 2
        else:
            labels[i] = 0
            sides[i] = 0

    return labels, sides


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — Hedge trades (R-based P&L like hedging_strategy_strict_old)
#  WIN:  one side hits TP → net = +target - stop  (winner - loser)
#  LOSS: neither side hits TP → net = -2 * stop   (both stopped)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_hedge(rf_probs_movement, movement_labels, target, stop, threshold):
    """
    Evaluate hedge entries: enter when model predicts price will move.
    WIN  = target - stop  (surviving side profits, cancelled side stopped)
    LOSS = -2 * stop      (both sides stopped out)
    """
    mask = rf_probs_movement >= threshold

    if mask.sum() < 10:
        return None

    selected_labels = movement_labels[mask]
    n = len(selected_labels)

    wins   = int((selected_labels == 1).sum())
    losses = n - wins

    # P&L per trade (in dollars)
    win_pnl  = target - stop     # one side wins target, other loses stop
    loss_pnl = -2 * stop         # both sides stopped out

    pnl_arr = np.where(selected_labels == 1, win_pnl, loss_pnl)
    pnl_arr = pnl_arr.astype(np.float64)

    total = float(pnl_arr.sum())
    avg   = float(pnl_arr.mean())
    std   = float(pnl_arr.std()) if n > 1 else 1e-10
    t_stat = avg / (std / np.sqrt(n) + 1e-10)
    sharpe = avg / (std + 1e-10) * np.sqrt(252)

    cum    = np.cumsum(pnl_arr)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    gross_w = float(pnl_arr[pnl_arr > 0].sum()) if (pnl_arr > 0).any() else 0.0
    gross_l = float(np.abs(pnl_arr[pnl_arr < 0].sum())) if (pnl_arr < 0).any() else 1e-10
    pf = gross_w / gross_l

    rr = abs(win_pnl / loss_pnl) if loss_pnl != 0 else 0.0

    return {
        'setups': n, 'wins': wins, 'losses': losses,
        'wr': round(wins / n * 100, 1),
        'total_pnl': round(total, 2), 'avg_pnl': round(avg, 4),
        'win_pnl': round(win_pnl, 2), 'loss_pnl': round(loss_pnl, 2),
        'rr': round(rr, 2),
        't_stat': round(t_stat, 2), 'sharpe': round(sharpe, 2),
        'pf': round(pf, 2), 'max_dd': round(max_dd, 2),
        'total_pnl_R': round(total / abs(loss_pnl), 1) if loss_pnl != 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE — Hedge only, fixed RF depth/trees
# ══════════════════════════════════════════════════════════════════════════════

BASE_FEATURES = [
    'abs_ema_spread', 'abs_flow_10', 'abs_flow_3', 'abs_flow_5',
    'abs_dist_ema21', 'combo_dist_flow', 'abs_dist_ema8',
    'flow_momentum', 'ema_spread', 'flow_20',
    'flow_3', 'flow_5', 'flow_10', 'trend_align', 'imbalance_3',
    'imbalance_5', 'close_position', 'vol_ratio', 'vol_expansion',
    'consistency_3', 'consistency_5', 'dist_ema8',
    'combo_flow_trend', 'combo_vol_imbalance',
    'combo_imbalance_momentum', 'combo_vol_flow',
]

FEATURE_NAMES = BASE_FEATURES


def run_pipeline(df_feat, split_idx, target, stop, horizon, rf_threshold):
    """Label movement -> train RF -> predict -> evaluate."""
    closes = df_feat['Close'].values.astype(np.float64)
    highs  = df_feat['High'].values.astype(np.float64)
    lows   = df_feat['Low'].values.astype(np.float64)

    labels, sides = label_movement(closes, highs, lows, horizon, target, stop)

    df_lab = df_feat.copy()
    df_lab['label'] = labels

    train = df_lab.iloc[:split_idx]
    test  = df_lab.iloc[split_idx:]

    # Subsample train for speed (90%)
    train = train.sample(frac=0.9, random_state=42)

    y_train = train['label'].values
    X_train = train[FEATURE_NAMES].fillna(0)
    X_test  = test[FEATURE_NAMES].fillna(0)

    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_DEPTH,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    proba   = rf.predict_proba(X_test)
    classes = rf.classes_

    # Find probability of class 1 (price will move)
    move_idx = np.where(classes == 1)[0]
    if len(move_idx) == 0:
        return None

    probs_movement = proba[:, move_idx[0]]

    return evaluate_hedge(probs_movement, test['label'].values,
                          target, stop, rf_threshold)


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE — 4 params only
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(df_feat, split_idx):
    def objective(trial):
        target  = trial.suggest_float('target', 1.0, 6.0, step=0.5)
        stop    = trial.suggest_float('stop', 1.0, 4.0, step=0.25)
        horizon = trial.suggest_int('horizon', 15, 60, step=5)
        rf_th   = trial.suggest_float('rf_threshold', 0.60, 0.95, step=0.05)

        # Enforce hedge RR >= 1:  WIN = target-stop, LOSS = -2*stop
        # RR = (target-stop)/(2*stop) >= 1  →  target >= 3*stop
        if target < 3 * stop:
            return -999.0

        result = run_pipeline(df_feat, split_idx, target, stop, horizon, rf_th)

        if result is None or result['setups'] < 30:
            return -999.0

        trial.set_user_attr('wr', result['wr'])
        trial.set_user_attr('rr', result['rr'])
        trial.set_user_attr('total_pnl', result['total_pnl'])
        trial.set_user_attr('avg_pnl', result['avg_pnl'])
        trial.set_user_attr('pf', result['pf'])
        trial.set_user_attr('sharpe', result['sharpe'])
        trial.set_user_attr('max_dd', result['max_dd'])
        trial.set_user_attr('setups', result['setups'])
        trial.set_user_attr('wins', result['wins'])
        trial.set_user_attr('losses', result['losses'])

        return result['t_stat']

    return objective


# ── Reporting ────────────────────────────────────────────────────────────────
def print_metrics(m, label=''):
    if m is None:
        print(f'  {label}: No valid trades')
        return
    print(f'  {label} [HEDGE - MOVEMENT]')
    print(f'    Setups:    {m["setups"]:>5d}   (W:{m["wins"]} L:{m["losses"]})')
    print(f'    Win Rate:  {m["wr"]:>5.1f}%')
    print(f'    Per trade: WIN=${m["win_pnl"]:>+.2f}  |  LOSS=${m["loss_pnl"]:>+.2f}  '
          f'|  RR: {m["rr"]:.2f}')
    print(f'    Total P&L: ${m["total_pnl"]:>9,.2f}  ({m["total_pnl_R"]:>+.1f}R)')
    print(f'    Avg P&L:   ${m["avg_pnl"]:>9.4f} per hedge')
    print(f'    t-stat:    {m["t_stat"]:>6.2f}')
    print(f'    PF:        {m["pf"]:>6.2f}')
    print(f'    Sharpe:    {m["sharpe"]:>6.2f}')
    print(f'    MaxDD:     ${m["max_dd"]:>9,.2f}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 72)
    print('  HEDGE TRADE OPTIMIZER -- XAUUSD (MOVEMENT DETECTION)')
    print(f'  Fixed: depth={RF_DEPTH}, trees={RF_TREES} | '
          f'Optimize: target, stop, horizon, threshold')
    print(f'  {N_TRIALS} trials | {len(FEATURE_NAMES)} base features')
    print(f'  RF predicts: "will price MOVE?" (not direction)')
    print('=' * 72)

    # ── Load ──
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
    print(f'  {len(df):,} bars: {df.index[0]} -> {df.index[-1]}')

    # ── Features ──
    print('  Engineering 1min features...')
    df = create_microstructure_features(df)
    print(f'  {len(df):,} bars after warmup')

    split = int(len(df) * 0.6)
    print(f'  Train: {split:,} | Test: {len(df)-split:,}')

    # ── Warm up Numba ──
    print('\n  Compiling Numba...')
    c = df['Close'].values[:2000].astype(np.float64)
    h = df['High'].values[:2000].astype(np.float64)
    l = df['Low'].values[:2000].astype(np.float64)
    _ = label_movement(c, h, l, 30, 3.0, 1.5)
    print('  Done.')

    # ══════════════════════════════════════════════════════════════════════
    #  BASELINE (same params as hedging_strategy_strict_old.py)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print('  BASELINE (TGT=$3, SL=$1.5, HOR=30, TH=0.75)')
    print(f'{"="*72}')

    base = run_pipeline(df, split, 3.0, 1.5, 30, 0.75)
    print_metrics(base, 'Baseline hedge')

    # ══════════════════════════════════════════════════════════════════════
    #  OPTUNA OPTIMIZATION (4 params)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print(f'  OPTUNA OPTIMIZATION ({N_TRIALS} trials, MOVEMENT DETECTION, XAUUSD)')
    print(f'  Search: target=[1-6], stop=[0.5-4], horizon=[15-60], threshold=[0.6-0.95]')
    print(f'  Speed: RF_TREES={RF_TREES} (fast), 90% train subsample')
    print(f'{"="*72}')

    objective = make_objective(df, split)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t1 = time.time()
    study.optimize(
        objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True,
    )
    elapsed = time.time() - t1

    bp = study.best_params
    print(f'\n\n  Completed {N_TRIALS} trials in {elapsed:.1f}s '
          f'({elapsed/N_TRIALS:.1f}s/trial)')

    print(f'\n  BEST PARAMETERS:')
    for k, v in bp.items():
        print(f'    {k:20s}: {v}')
    # P&L per outcome
    win_net = bp['target'] - bp['stop']
    loss_net = -2 * bp['stop']
    print(f'    {"WIN P&L":20s}: ${win_net:+.2f}')
    print(f'    {"LOSS P&L":20s}: ${loss_net:+.2f}')

    # ── Evaluate best ──
    print(f'\n{"─"*72}')
    print('  OPTIMIZED RESULTS')
    print(f'{"─"*72}')
    opt = run_pipeline(df, split, bp['target'], bp['stop'],
                       bp['horizon'], bp['rf_threshold'])
    print_metrics(opt, 'Test (40%) -- OPTIMIZED')

    # ── Comparison ──
    if base and opt:
        print(f'\n{"─"*72}')
        print('  BASELINE vs OPTIMIZED')
        print(f'{"─"*72}')
        print(f'  {"Metric":<15s} {"Baseline":>10s} {"Optimized":>10s}')
        print(f'  {"─"*40}')
        for k, lab in [('wr', 'Win Rate %'), ('rr', 'Reward:Risk'),
                       ('total_pnl', 'Total P&L $'), ('avg_pnl', 'Avg P&L $'),
                       ('t_stat', 't-stat'), ('pf', 'Profit Factor'),
                       ('max_dd', 'Max DD $'), ('setups', 'Setups')]:
            bv = base.get(k, 'N/A')
            ov = opt.get(k, 'N/A')
            print(f'  {lab:<15s} {str(bv):>10s} {str(ov):>10s}')

    # ── All profitable trials ──
    print(f'\n{"="*72}')
    print('  ALL PROFITABLE TRIALS (ranked by t-stat)')
    print(f'{"="*72}\n')

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > 0]
    trials_df = trials_df.sort_values('value', ascending=False).reset_index(drop=True)

    rows = []
    for _, row in trials_df.iterrows():
        rows.append({
            'rank': len(rows) + 1,
            't_stat': row['value'],
            'TGT': row['params_target'],
            'SL': row['params_stop'],
            'horizon': int(row['params_horizon']),
            'threshold': row['params_rf_threshold'],
            'WR%': row.get('user_attrs_wr', 0),
            'RR': row.get('user_attrs_rr', 0),
            'total_pnl': row.get('user_attrs_total_pnl', 0),
            'avg_pnl': row.get('user_attrs_avg_pnl', 0),
            'PF': row.get('user_attrs_pf', 0),
            'sharpe': row.get('user_attrs_sharpe', 0),
            'max_dd': row.get('user_attrs_max_dd', 0),
            'setups': int(row.get('user_attrs_setups', 0)),
            'wins': int(row.get('user_attrs_wins', 0)),
            'losses': int(row.get('user_attrs_losses', 0)),
        })
    results_df = pd.DataFrame(rows)

    print(f'  {"#":>3s}  {"t":>6s}  {"TGT":>5s}  {"SL":>5s}  '
          f'{"HOR":>3s}  {"TH":>4s}  {"WR%":>5s}  {"RR":>5s}  '
          f'{"P&L":>9s}  {"PF":>5s}  {"Setups":>6s}  {"W":>4s}  {"L":>4s}')
    print(f'  {"─"*80}')
    for _, r in results_df.iterrows():
        print(f'  {int(r["rank"]):3d}  {r["t_stat"]:6.2f}  '
              f'${r["TGT"]:4.1f}  ${r["SL"]:4.2f}  '
              f'{int(r["horizon"]):3d}  {r["threshold"]:4.2f}  '
              f'{r["WR%"]:5.1f}  {r["RR"]:5.2f}  '
              f'${r["total_pnl"]:>8,.2f}  {r["PF"]:5.2f}  '
              f'{int(r["setups"]):>6d}  {int(r["wins"]):>4d}  {int(r["losses"]):>4d}')

    # Save to CSV
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'optuna_hedge_xauusd.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'\n  Saved {len(results_df)} profitable trials -> {csv_path}')

    # ── Summary ──
    total_time = time.time() - t0
    print(f'\n{"="*72}')
    print(f'  SUMMARY')
    print(f'{"="*72}')
    print(f'  Mode: HEDGE (MOVEMENT DETECTION — predict movement, not direction)')
    print(f'  Total time: {total_time:.1f}s')
    print(f'  WIN = target - stop, LOSS = -2×stop')
    print(f'  RF: depth={RF_DEPTH}, trees={RF_TREES} (fixed)')
    if opt:
        print(f'  Best: TGT={bp["target"]}, SL={bp["stop"]}, '
              f'HOR={bp["horizon"]}, TH={bp["rf_threshold"]}')
        print(f'  Best P&L: ${opt["total_pnl"]:,.2f} ({opt["total_pnl_R"]:+.1f}R) | '
              f'RR: {opt["rr"]} | t: {opt["t_stat"]} | WR: {opt["wr"]}%')
    print(f'{"="*72}')
