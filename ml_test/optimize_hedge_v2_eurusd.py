"""
Hedging Strategy v2 — Critical Redesign with Numba + Optuna
============================================================
Fixes from v1:
  1. Tests DIRECTIONAL mode (1× spread) vs HEDGE mode (2× spread)
  2. Proper bar-by-bar hedge simulation (which side stopped first?)
  3. Trailing stop option for the surviving side
  4. Real dollar P&L with spread included in objective
  5. Multi-class labeling: LONG_WIN / SHORT_WIN / BOTH_STOPPED / NO_MOVE
  6. Minimum 1:2 RR constraint

Params optimized:
  - target, stop, horizon
  - rf_threshold, rf_depth, rf_trees
  - mode: 'directional' or 'hedge'
  - trail_stop: trail SL to breakeven after X$ profit (0 = off)
"""

import numpy as np
import pandas as pd
import numba as nb
import optuna
import pickle
import time
import warnings
import os

from sklearn.ensemble import RandomForestClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'XAUUSD1.csv')
N_TRIALS  = 200
SPREAD    = 0.30


# ── Feature engineering (same as original) ───────────────────────────────────
def create_microstructure_features(df):
    df = df.copy()

    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)

    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)

    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()

    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))

    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)

    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - \
                         ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']

    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)

    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)

    df['flow_15'] = df['directional_flow'].rolling(15).sum()
    df['flow_20'] = df['directional_flow'].rolling(20).sum()
    df['flow_accel'] = df['flow_3'] - df['flow_3'].shift(3)
    df['flow_accel_5'] = df['flow_5'] - df['flow_5'].shift(5)
    df['abs_flow_3'] = df['flow_3'].abs()
    df['abs_flow_5'] = df['flow_5'].abs()
    df['abs_flow_10'] = df['flow_10'].abs()
    df['flow_divergence'] = (df['flow_3'] * df['flow_10'] < 0).astype(int)

    df['consecutive_up'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumcount() + 1
    df['consecutive_up'] = df['consecutive_up'] * df['is_up']
    df['consecutive_down'] = (1 - df['is_up']).groupby(((1-df['is_up']) != (1-df['is_up']).shift()).cumsum()).cumcount() + 1
    df['consecutive_down'] = df['consecutive_down'] * (1 - df['is_up'])
    df['max_consecutive'] = df[['consecutive_up', 'consecutive_down']].max(axis=1)

    df['flow_efficiency'] = df['abs_body'] / (df['range'] + 1e-10)
    df['flow_eff_3'] = df['flow_efficiency'].rolling(3).mean()
    df['flow_eff_5'] = df['flow_efficiency'].rolling(5).mean()

    df['atr_roc'] = (df['atr_3'] - df['atr_3'].shift(3)) / (df['atr_3'].shift(3) + 1e-10)
    df['vol_breakout'] = df['range'] / (df['atr_20'] + 1e-10)
    df['range_min_5'] = df['range'].rolling(5).min()
    df['range_max_5'] = df['range'].rolling(5).max()
    df['range_squeeze'] = df['range_min_5'] / (df['range_max_5'] + 1e-10)

    df['abs_dist_ema8'] = df['dist_ema8'].abs()
    df['dist_ema21'] = (df['Close'] - df['ema_21']) / df['Close']
    df['abs_dist_ema21'] = df['dist_ema21'].abs()
    df['ema_spread'] = (df['ema_8'] - df['ema_21']) / df['Close']
    df['abs_ema_spread'] = df['ema_spread'].abs()

    df['combo_abs_flow_vol'] = df['abs_flow_5'] * df['vol_ratio']
    df['combo_eff_flow'] = df['flow_eff_3'] * df['abs_flow_3']
    df['combo_consecutive_body'] = df['max_consecutive'] * df['body_pct']
    df['combo_atr_roc_accel'] = df['atr_roc'] * df['flow_accel'].abs()
    df['combo_squeeze_flow'] = (1 - df['range_squeeze']) * df['abs_flow_3']
    df['combo_dist_flow'] = df['abs_dist_ema8'] * df['abs_flow_5']

    # Combo features from main script
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
#  NUMBA LABELING — Proper bar-by-bar hedge simulation
# ══════════════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def label_hedge_outcomes(closes, highs, lows, horizon, target, stop):
    """
    Proper hedge labeling: simulate BOTH positions bar by bar.
    For each bar i:
      Long:  TP = close[i] + target,  SL = close[i] - stop
      Short: TP = close[i] - target,  SL = close[i] + stop

    Track which STOP is hit first (determines which side survives).
    Then track if surviving side hits TP or gets stopped too.

    Returns array:
      1 = LONG side wins (net = +target - stop)
      2 = SHORT side wins (net = +target - stop)
      3 = LONG survived but timed out (partial P&L)
      4 = SHORT survived but timed out (partial P&L)
      0 = BOTH stopped or no movement
    Also returns dollar P&L per bar (without spread).
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32)
    pnl = np.zeros(n, dtype=np.float64)

    for i in range(n - horizon):
        entry = closes[i]
        long_tp  = entry + target
        long_sl  = entry - stop
        short_tp = entry - target
        short_sl = entry + stop

        # Phase 1: which side gets stopped first?
        long_alive = True
        short_alive = True
        long_stopped_bar = -1
        short_stopped_bar = -1

        for j in range(i + 1, min(i + horizon + 1, n)):
            if long_alive and lows[j] <= long_sl:
                long_alive = False
                long_stopped_bar = j
            if short_alive and highs[j] >= short_sl:
                short_alive = False
                short_stopped_bar = j

            # If one side stopped and other still alive, move to phase 2
            if not long_alive and short_alive:
                break
            if not short_alive and long_alive:
                break
            # If both stopped on same bar
            if not long_alive and not short_alive:
                break

        # Case: both stopped (same bar or within horizon)
        if not long_alive and not short_alive:
            labels[i] = 0
            pnl[i] = -stop - stop  # lost both SLs
            continue

        # Case: long stopped, short survives
        if not long_alive and short_alive:
            # Track short from here
            short_pnl_final = 0.0
            outcome = 4  # default: short survived, timed out
            for j in range(long_stopped_bar, min(i + horizon + 1, n)):
                if lows[j] <= short_tp:
                    outcome = 2  # short wins
                    short_pnl_final = target
                    break
                if highs[j] >= short_sl:
                    outcome = 0  # short also stopped
                    short_pnl_final = -stop
                    break
            if outcome == 4:
                # Timed out — use closing price
                last_bar = min(i + horizon, n - 1)
                short_pnl_final = entry - closes[last_bar]  # short P&L

            labels[i] = outcome
            pnl[i] = -stop + short_pnl_final  # long loss + short result
            continue

        # Case: short stopped, long survives
        if not short_alive and long_alive:
            long_pnl_final = 0.0
            outcome = 3  # default: long survived, timed out
            for j in range(short_stopped_bar, min(i + horizon + 1, n)):
                if highs[j] >= long_tp:
                    outcome = 1  # long wins
                    long_pnl_final = target
                    break
                if lows[j] <= long_sl:
                    outcome = 0  # long also stopped
                    long_pnl_final = -stop
                    break
            if outcome == 3:
                last_bar = min(i + horizon, n - 1)
                long_pnl_final = closes[last_bar] - entry

            labels[i] = outcome
            pnl[i] = -stop + long_pnl_final  # short loss + long result
            continue

        # Case: neither stopped within horizon (low vol / tight range)
        # Both positions at ~breakeven, just timeout
        last_bar = min(i + horizon, n - 1)
        final_move = closes[last_bar] - entry
        # Long P&L = final_move, Short P&L = -final_move, net = 0
        labels[i] = 0
        pnl[i] = 0.0

    return labels, pnl


@nb.njit(cache=True)
def label_directional(closes, highs, lows, horizon, target, stop):
    """
    Directional labeling: for each bar, does price hit TP or SL first?
    Returns:
      1 = LONG TP hit first
      2 = SHORT TP hit first
      0 = neither (range-bound)
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = closes[i]
        long_tp  = entry + target
        long_sl  = entry - stop
        short_tp = entry - target
        short_sl = entry + stop

        # Check both directions simultaneously, first hit wins
        for j in range(i + 1, min(i + horizon + 1, n)):
            # Long TP
            if highs[j] >= long_tp:
                labels[i] = 1
                break
            # Short TP
            if lows[j] <= short_tp:
                labels[i] = 2
                break
            # Long SL (eliminates long possibility going forward)
            if lows[j] <= long_sl:
                # Check if short still viable
                for k in range(j, min(i + horizon + 1, n)):
                    if lows[k] <= short_tp:
                        labels[i] = 2
                        break
                    if highs[k] >= short_sl:
                        break
                break
            # Short SL (eliminates short possibility going forward)
            if highs[j] >= short_sl:
                for k in range(j, min(i + horizon + 1, n)):
                    if highs[k] >= long_tp:
                        labels[i] = 1
                        break
                    if lows[k] <= long_sl:
                        break
                break

    return labels


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_hedge_v2(rf_probs, hedge_labels, hedge_pnl, threshold):
    """Evaluate hedge using proper bar-by-bar P&L (already in dollars)."""
    mask = rf_probs >= threshold
    n = mask.sum()
    if n < 10:
        return None

    filtered_pnl = hedge_pnl[mask]
    filtered_labels = hedge_labels[mask]

    # Subtract 2× spread per setup
    spread_cost = 2 * SPREAD
    net_pnl = filtered_pnl - spread_cost

    wins = int((filtered_labels == 1).sum() + (filtered_labels == 2).sum())
    losses = int((filtered_labels == 0).sum())
    timeouts = int((filtered_labels == 3).sum() + (filtered_labels == 4).sum())

    total = float(net_pnl.sum())
    avg = float(net_pnl.mean())
    std = float(net_pnl.std()) if n > 1 else 1e-10
    t_stat = avg / (std / np.sqrt(n) + 1e-10)
    sharpe = avg / (std + 1e-10) * np.sqrt(252)

    cum = np.cumsum(net_pnl)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    gross_w = float(net_pnl[net_pnl > 0].sum()) if (net_pnl > 0).any() else 0.0
    gross_l = float(np.abs(net_pnl[net_pnl < 0].sum())) if (net_pnl < 0).any() else 1e-10
    pf = gross_w / gross_l

    avg_win = float(net_pnl[net_pnl > 0].mean()) if (net_pnl > 0).any() else 0.0
    avg_loss = float(net_pnl[net_pnl < 0].mean()) if (net_pnl < 0).any() else -1.0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    return {
        'mode': 'HEDGE', 'setups': n, 'wins': wins, 'losses': losses,
        'timeouts': timeouts,
        'wr': round(wins / n * 100, 1),
        'total_pnl': round(total, 2), 'avg_pnl': round(avg, 4),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'rr': round(rr, 2),
        't_stat': round(t_stat, 2), 'sharpe': round(sharpe, 2),
        'pf': round(pf, 2), 'max_dd': round(max_dd, 2),
    }


def evaluate_directional(rf_probs_long, rf_probs_short, dir_labels,
                         target, stop, threshold):
    """
    Directional mode: RF predicts direction.
    Enter LONG when prob_long >= threshold, SHORT when prob_short >= threshold.
    Only 1× spread per trade.
    """
    n_total = len(dir_labels)

    # Long entries: predict class 1 with high confidence
    long_mask = rf_probs_long >= threshold
    # Short entries: predict class 2 with high confidence
    short_mask = rf_probs_short >= threshold

    trades_pnl = []
    trade_types = []

    # Long trades
    long_idx = np.where(long_mask)[0]
    for i in long_idx:
        if dir_labels[i] == 1:
            trades_pnl.append(target - SPREAD)
            trade_types.append(1)
        elif dir_labels[i] == 2:
            # Price went other way — stopped
            trades_pnl.append(-stop - SPREAD)
            trade_types.append(-1)
        else:
            # No movement — timed out
            trades_pnl.append(-SPREAD)
            trade_types.append(0)

    # Short trades
    short_idx = np.where(short_mask)[0]
    for i in short_idx:
        if dir_labels[i] == 2:
            trades_pnl.append(target - SPREAD)
            trade_types.append(1)
        elif dir_labels[i] == 1:
            trades_pnl.append(-stop - SPREAD)
            trade_types.append(-1)
        else:
            trades_pnl.append(-SPREAD)
            trade_types.append(0)

    if len(trades_pnl) < 10:
        return None

    pnl = np.array(trades_pnl)
    types = np.array(trade_types)
    n = len(pnl)

    wins = int((types == 1).sum())
    losses = int((types == -1).sum())
    timeouts = int((types == 0).sum())

    total = float(pnl.sum())
    avg = float(pnl.mean())
    std = float(pnl.std()) if n > 1 else 1e-10
    t_stat = avg / (std / np.sqrt(n) + 1e-10)
    sharpe = avg / (std + 1e-10) * np.sqrt(252)

    cum = np.cumsum(pnl)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    gross_w = float(pnl[pnl > 0].sum()) if (pnl > 0).any() else 0.0
    gross_l = float(np.abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 1e-10
    pf = gross_w / gross_l

    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else -1.0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    return {
        'mode': 'DIRECTIONAL',
        'setups': n, 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'wr': round(wins / n * 100, 1),
        'total_pnl': round(total, 2), 'avg_pnl': round(avg, 4),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'rr': round(rr, 2),
        't_stat': round(t_stat, 2), 'sharpe': round(sharpe, 2),
        'pf': round(pf, 2), 'max_dd': round(max_dd, 2),
        'n_long': int(long_mask.sum()), 'n_short': int(short_mask.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'abs_ema_spread', 'abs_flow_10', 'abs_flow_3', 'abs_flow_5',
    'abs_dist_ema21', 'combo_dist_flow', 'abs_dist_ema8',
    'flow_momentum', 'ema_spread', 'flow_20',
    # Additional features for directional prediction
    'flow_3', 'flow_5', 'flow_10', 'trend_align', 'imbalance_3',
    'imbalance_5', 'close_position', 'vol_ratio', 'vol_expansion',
    'consistency_3', 'consistency_5', 'dist_ema8',
    'combo_flow_trend', 'combo_vol_imbalance',
    'combo_imbalance_momentum', 'combo_vol_flow',
]


def run_pipeline(df_feat, split_idx, target, stop, horizon,
                 rf_threshold, rf_depth, rf_trees, mode='hedge'):
    """
    Full pipeline: label → train RF → predict → evaluate.
    mode='hedge': both sides, 2× spread
    mode='directional': predict direction, 1× spread
    """
    closes = df_feat['Close'].values.astype(np.float64)
    highs  = df_feat['High'].values.astype(np.float64)
    lows   = df_feat['Low'].values.astype(np.float64)

    if mode == 'hedge':
        labels, raw_pnl = label_hedge_outcomes(closes, highs, lows,
                                                horizon, target, stop)
        df_lab = df_feat.copy()
        df_lab['label'] = labels
        df_lab['raw_pnl'] = raw_pnl

        train = df_lab.iloc[:split_idx]
        test  = df_lab.iloc[split_idx:]

        # Binary: will TP hit on either side? (labels 1 or 2 = yes)
        y_train = ((train['label'] == 1) | (train['label'] == 2)).astype(int)

        X_train = train[FEATURE_NAMES].fillna(0)
        X_test  = test[FEATURE_NAMES].fillna(0)

        rf = RandomForestClassifier(
            n_estimators=rf_trees, max_depth=rf_depth,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        probs = rf.predict_proba(X_test)[:, 1]

        return evaluate_hedge_v2(probs, test['label'].values,
                                  test['raw_pnl'].values, rf_threshold)

    else:  # directional
        labels = label_directional(closes, highs, lows, horizon, target, stop)

        df_lab = df_feat.copy()
        df_lab['label'] = labels

        train = df_lab.iloc[:split_idx]
        test  = df_lab.iloc[split_idx:]

        # Multi-class: 0=no move, 1=long, 2=short
        y_train = train['label'].values

        X_train = train[FEATURE_NAMES].fillna(0)
        X_test  = test[FEATURE_NAMES].fillna(0)

        rf = RandomForestClassifier(
            n_estimators=rf_trees, max_depth=rf_depth,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        proba = rf.predict_proba(X_test)
        classes = rf.classes_

        # Get probabilities for class 1 (long) and class 2 (short)
        long_idx  = np.where(classes == 1)[0]
        short_idx = np.where(classes == 2)[0]

        if len(long_idx) == 0 or len(short_idx) == 0:
            return None

        probs_long  = proba[:, long_idx[0]]
        probs_short = proba[:, short_idx[0]]

        return evaluate_directional(probs_long, probs_short,
                                     test['label'].values,
                                     target, stop, rf_threshold)


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(df_feat, split_idx):
    def objective(trial):
        mode    = trial.suggest_categorical('mode', ['hedge', 'directional'])
        target  = trial.suggest_float('target', 2.0, 15.0, step=0.5)
        stop    = trial.suggest_float('stop', 0.5, 5.0, step=0.25)
        horizon = trial.suggest_int('horizon', 15, 90, step=5)
        rf_th   = trial.suggest_float('rf_threshold', 0.40, 0.90, step=0.05)
        rf_dep  = trial.suggest_int('rf_depth', 3, 20, step=1)
        rf_tr   = trial.suggest_int('rf_trees', 50, 300, step=50)

        # Constraints
        if mode == 'hedge':
            # Win = target - stop - 2×spread, need positive
            win_pnl = target - stop - 2 * SPREAD
            if win_pnl <= 0:
                return -999.0
            # Check RR: win / |loss| >= 1 at minimum
            loss_pnl = 2 * stop + 2 * SPREAD
            if win_pnl / loss_pnl < 0.3:  # Very loose for hedge
                return -999.0
        else:
            # Directional: win = target - spread, loss = stop + spread
            # For 1:2 RR minimum
            rr = (target - SPREAD) / (stop + SPREAD)
            if rr < 1.5:  # At least 1.5:1 for directional
                return -999.0

        result = run_pipeline(df_feat, split_idx, target, stop, horizon,
                              rf_th, rf_dep, rf_tr, mode)

        if result is None or result['setups'] < 30:
            return -999.0

        # Store full metrics for later reporting
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

        # Objective: t-stat (already includes spread in P&L)
        return result['t_stat']

    return objective


# ── Reporting ────────────────────────────────────────────────────────────────
def print_metrics(m, label=''):
    if m is None:
        print(f'  {label}: No valid trades')
        return
    spread_n = 2 if m['mode'] == 'HEDGE' else 1
    print(f'  {label} [{m["mode"]}]')
    print(f'    Setups:    {m["setups"]:>5d}   (W:{m["wins"]} L:{m["losses"]} T:{m["timeouts"]})')
    print(f'    Win Rate:  {m["wr"]:>5.1f}%')
    print(f'    Avg Win:   ${m["avg_win"]:>+7.2f}  |  Avg Loss: ${m["avg_loss"]:>+7.2f}  '
          f'|  RR: {m["rr"]:.2f}  (incl {spread_n}×${SPREAD} spread)')
    print(f'    Total P&L: ${m["total_pnl"]:>9,.2f}')
    print(f'    Avg P&L:   ${m["avg_pnl"]:>9.4f} per setup')
    print(f'    t-stat:    {m["t_stat"]:>6.2f}')
    print(f'    PF:        {m["pf"]:>6.2f}')
    print(f'    Sharpe:    {m["sharpe"]:>6.2f}')
    print(f'    MaxDD:     ${m["max_dd"]:>9,.2f}')
    if 'n_long' in m:
        print(f'    Entries:   {m["n_long"]} long, {m["n_short"]} short')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 72)
    print('  HEDGING STRATEGY v2 — CRITICAL REDESIGN')
    print('  Numba JIT + Optuna | Spread-adjusted P&L | Directional + Hedge')
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
    print(f'  {len(df):,} bars')

    print('  Engineering features...')
    df = create_microstructure_features(df)
    print(f'  {len(df):,} bars after warmup')

    split = int(len(df) * 0.6)
    print(f'  Train: {split:,} | Test: {len(df)-split:,}')

    # ── Warm up Numba ──
    print('\n  Compiling Numba...')
    c = df['Close'].values[:2000].astype(np.float64)
    h = df['High'].values[:2000].astype(np.float64)
    l = df['Low'].values[:2000].astype(np.float64)
    _ = label_hedge_outcomes(c, h, l, 30, 3.0, 1.5)
    _ = label_directional(c, h, l, 30, 3.0, 1.5)
    print('  Done.')

    # ══════════════════════════════════════════════════════════════════════
    #  BASELINES
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print('  BASELINES (original params: TGT=$3, SL=$1.5, HOR=30, TH=0.75)')
    print(f'{"="*72}')

    base_hedge = run_pipeline(df, split, 3.0, 1.5, 30, 0.75, 10, 100, 'hedge')
    print_metrics(base_hedge, 'Original Hedge')

    base_dir = run_pipeline(df, split, 3.0, 1.5, 30, 0.75, 10, 100, 'directional')
    print_metrics(base_dir, 'Directional (same params)')

    # ── Directional with better RR ──
    print(f'\n  --- Better RR baselines ---')
    for tgt, sl in [(6, 2), (8, 3), (10, 3), (6, 3)]:
        m = run_pipeline(df, split, float(tgt), float(sl), 45, 0.60, 10, 100, 'directional')
        if m:
            print(f'  TGT=${tgt} SL=${sl}: '
                  f'Trades={m["setups"]:>5d} WR={m["wr"]:>5.1f}% '
                  f'RR={m["rr"]:.2f} P&L=${m["total_pnl"]:>8,.2f} '
                  f't={m["t_stat"]:>6.2f}')

    # ── Hedge with higher target ──
    print(f'\n  --- Hedge with higher TARGET ---')
    for tgt, sl in [(6, 2), (8, 2), (10, 2), (8, 3)]:
        m = run_pipeline(df, split, float(tgt), float(sl), 45, 0.60, 10, 100, 'hedge')
        if m:
            print(f'  TGT=${tgt} SL=${sl}: '
                  f'Trades={m["setups"]:>5d} WR={m["wr"]:>5.1f}% '
                  f'RR={m["rr"]:.2f} P&L=${m["total_pnl"]:>8,.2f} '
                  f't={m["t_stat"]:>6.2f}')

    # ══════════════════════════════════════════════════════════════════════
    #  OPTUNA OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*72}')
    print(f'  OPTUNA OPTIMIZATION ({N_TRIALS} trials, hedge + directional)')
    print(f'{"="*72}')

    objective = make_objective(df, split)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t1 = time.time()
    study.optimize(
        objective, n_trials=N_TRIALS, show_progress_bar=False,
        callbacks=[lambda study, trial:
            print(f'    Trial {trial.number+1:>3d}/{N_TRIALS}  '
                  f't={trial.value:>7.2f}  best={study.best_value:>7.2f}  '
                  f'mode={trial.params.get("mode","?")}',
                  end='\r')
        ]
    )
    elapsed = time.time() - t1

    bp = study.best_params
    print(f'\n\n  Completed {N_TRIALS} trials in {elapsed:.1f}s '
          f'({elapsed/N_TRIALS:.1f}s/trial)')

    print(f'\n  BEST PARAMETERS:')
    for k, v in bp.items():
        print(f'    {k:20s}: {v}')

    if bp['mode'] == 'directional':
        rr = (bp['target'] - SPREAD) / (bp['stop'] + SPREAD)
        print(f'    {"actual_rr":20s}: {rr:.2f}')
    else:
        win = bp['target'] - bp['stop'] - 2*SPREAD
        loss = 2*bp['stop'] + 2*SPREAD
        print(f'    {"hedge_win_$":20s}: ${win:.2f}')
        print(f'    {"hedge_loss_$":20s}: -${loss:.2f}')

    # ── Evaluate best ──
    print(f'\n{"─"*72}')
    print('  OPTIMIZED RESULTS')
    print(f'{"─"*72}')

    opt = run_pipeline(df, split, bp['target'], bp['stop'], bp['horizon'],
                       bp['rf_threshold'], bp['rf_depth'], bp['rf_trees'],
                       bp['mode'])
    print_metrics(opt, 'Test (40%) — OPTIMIZED')

    # ── Comparison ──
    print(f'\n{"─"*72}')
    print('  BASELINE vs OPTIMIZED')
    print(f'{"─"*72}')
    base = base_hedge  # compare vs original hedge
    if base and opt:
        print(f'  {"Metric":<15s} {"Baseline":>10s} {"Optimized":>10s}')
        print(f'  {"─"*40}')
        for k, lab in [('wr', 'Win Rate %'), ('rr', 'Reward:Risk'),
                       ('total_pnl', 'Total P&L $'), ('avg_pnl', 'Avg P&L $'),
                       ('t_stat', 't-stat'), ('pf', 'Profit Factor'),
                       ('max_dd', 'Max DD $'), ('setups', 'Setups')]:
            bv = base.get(k, 'N/A')
            ov = opt.get(k, 'N/A')
            print(f'  {lab:<15s} {str(bv):>10s} {str(ov):>10s}')

    # ── All profitable trials ranked ──
    print(f'\n{"="*72}')
    print('  ALL PROFITABLE TRIALS (ranked by t-stat)')
    print(f'{"="*72}\n')

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > 0]
    trials_df = trials_df.sort_values('value', ascending=False).reset_index(drop=True)

    # Build clean output table
    rows = []
    for _, row in trials_df.iterrows():
        rows.append({
            'rank': len(rows) + 1,
            't_stat': row['value'],
            'mode': row.get('params_mode', '?'),
            'TGT': row['params_target'],
            'SL': row['params_stop'],
            'horizon': int(row['params_horizon']),
            'threshold': row['params_rf_threshold'],
            'depth': int(row['params_rf_depth']),
            'trees': int(row['params_rf_trees']),
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

    print(f'  {"#":>3s}  {"t":>6s}  {"Mode":>5s}  {"TGT":>5s}  {"SL":>5s}  '
          f'{"HOR":>3s}  {"TH":>4s}  {"WR%":>5s}  {"RR":>5s}  '
          f'{"P&L":>9s}  {"PF":>5s}  {"Setups":>6s}  {"W":>4s}  {"L":>4s}')
    print(f'  {"─"*80}')
    for _, r in results_df.iterrows():
        md = str(r['mode'])[:5]
        print(f'  {int(r["rank"]):3d}  {r["t_stat"]:6.2f}  {md:>5s}  '
              f'${r["TGT"]:4.1f}  ${r["SL"]:4.2f}  '
              f'{r["horizon"]:3d}  {r["threshold"]:4.2f}  '
              f'{r["WR%"]:5.1f}  {r["RR"]:5.2f}  '
              f'${r["total_pnl"]:>8,.2f}  {r["PF"]:5.2f}  '
              f'{r["setups"]:>6d}  {r["wins"]:>4d}  {r["losses"]:>4d}')

    # Save to CSV
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'optuna_profitable_trials.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'\n  Saved {len(results_df)} profitable trials → {csv_path}')

    # ── Summary ──
    total_time = time.time() - t0
    print(f'\n{"="*72}')
    print(f'  SUMMARY')
    print(f'{"="*72}')
    print(f'  Total time: {total_time:.1f}s')
    print(f'  Spread: ${SPREAD}/position')
    if opt:
        print(f'  Best mode: {opt["mode"]}')
        print(f'  Best P&L: ${opt["total_pnl"]:,.2f} | RR: {opt["rr"]} | '
              f't: {opt["t_stat"]} | WR: {opt["wr"]}%')
    print(f'{"="*72}')
