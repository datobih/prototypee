"""
Unified Reversal Strategy Optimizer
=====================================
Combines hedging_strategy_strict.py (model training) +
wait_reversal_strategy.py (reversal execution) into one Optuna run.

Optimizes 8 parameters:
  Model:    model_target, model_stop, model_horizon  (labeling)
  Signal:   rf_threshold                              (filtering)
  Strategy: rev_tp, rev_sl, wait_window, cooldown     (execution)

Flow per trial:
  1. Label outcomes with model params (numba)
  2. Train RF binary classifier (will price move?)
  3. Get RF probs on test set
  4. Simulate reversal strategy with strategy params (numba)
  5. Return objective (total P&L)
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

DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'XAUUSD1.csv')
N_TRIALS     = 300
RF_TREES     = 50     # fast search (retrain best with 200 after)
RF_DEPTH     = 12
RF_MIN_LEAF  = 100
RF_MIN_SPLIT = 200
TRAIN_FRAC   = 0.5    # subsample training for speed
PURGE_GAP    = 60     # fixed purge gap (= max possible horizon)


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — exact copy from hedging_strategy_strict.py
# ═══════════════════════════════════════════════════════════════════════════════

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

    # ── Round 1: Optimized features ──
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

    # ── Round 2: Extended flow, EMA distance, z-scores, time ──
    df = df.copy()

    df['abs_flow_15'] = df['flow_15'].abs()
    df['abs_flow_20'] = df['flow_20'].abs()

    flow10_mean = df['abs_flow_10'].rolling(50).mean()
    flow10_std = df['abs_flow_10'].rolling(50).std()
    df['flow_zscore'] = (df['abs_flow_10'] - flow10_mean) / (flow10_std + 1e-10)

    abs_per_bar = df['directional_flow'].abs()
    df['flow_net_ratio_5'] = df['abs_flow_5'] / (abs_per_bar.rolling(5).sum() + 1e-10)
    df['flow_net_ratio_10'] = df['abs_flow_10'] / (abs_per_bar.rolling(10).sum() + 1e-10)

    df['flow_max_bar_5'] = df['directional_flow'].abs().rolling(5).max()
    df['flow_concentration_5'] = df['flow_max_bar_5'] / (abs_per_bar.rolling(5).sum() + 1e-10)

    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['ema_100'] = df['Close'].ewm(span=100).mean()
    df['dist_ema50'] = (df['Close'] - df['ema_50']) / df['Close']
    df['abs_dist_ema50'] = df['dist_ema50'].abs()
    df['dist_ema100'] = (df['Close'] - df['ema_100']) / df['Close']
    df['abs_dist_ema100'] = df['dist_ema100'].abs()

    df['ema_spread_21_50'] = (df['ema_21'] - df['ema_50']) / df['Close']
    df['abs_ema_spread_21_50'] = df['ema_spread_21_50'].abs()
    df['ema_spread_50_100'] = (df['ema_50'] - df['ema_100']) / df['Close']
    df['abs_ema_spread_50_100'] = df['ema_spread_50_100'].abs()

    roll_mean_20 = df['Close'].rolling(20).mean()
    roll_std_20 = df['Close'].rolling(20).std()
    df['price_zscore_20'] = (df['Close'] - roll_mean_20) / (roll_std_20 + 1e-10)
    df['abs_price_zscore_20'] = df['price_zscore_20'].abs()

    roll_mean_50 = df['Close'].rolling(50).mean()
    roll_std_50 = df['Close'].rolling(50).std()
    df['price_zscore_50'] = (df['Close'] - roll_mean_50) / (roll_std_50 + 1e-10)
    df['abs_price_zscore_50'] = df['price_zscore_50'].abs()

    df['boll_upper_20'] = roll_mean_20 + 2 * roll_std_20
    df['boll_lower_20'] = roll_mean_20 - 2 * roll_std_20
    df['boll_width_20'] = (df['boll_upper_20'] - df['boll_lower_20']) / df['Close']
    df['boll_pct_20'] = (df['Close'] - df['boll_lower_20']) / (df['boll_upper_20'] - df['boll_lower_20'] + 1e-10)
    df['boll_outside_20'] = ((df['Close'] > df['boll_upper_20']) | (df['Close'] < df['boll_lower_20'])).astype(int)

    df['flow_per_atr_3'] = df['abs_flow_3'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['flow_per_atr_10'] = df['abs_flow_10'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['dist_ema8_per_atr'] = df['abs_dist_ema8'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['dist_ema21_per_atr'] = df['abs_dist_ema21'] / (df['atr_10'] / df['Close'] + 1e-10)
    df['ema_spread_per_atr'] = df['abs_ema_spread'] / (df['atr_10'] / df['Close'] + 1e-10)

    df['atr_percentile'] = df['atr_10'].rolling(100).rank(pct=True)

    hour = df.index.hour
    minute = df.index.minute
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_of_week'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

    df['combo_flow3_dist21'] = df['abs_flow_3'] * df['abs_dist_ema21']
    df['combo_flow10_dist8'] = df['abs_flow_10'] * df['abs_dist_ema8']
    df['combo_flow10_spread'] = df['abs_flow_10'] * df['abs_ema_spread']
    df['combo_flow_atr_pct'] = df['abs_flow_10'] * df['atr_percentile']
    df['combo_flow_zscore_dist'] = df['flow_zscore'] * df['abs_dist_ema8']
    df['combo_triple'] = df['abs_flow_5'] * df['abs_dist_ema8'] * df['vol_ratio']
    df['combo_net_ratio_flow'] = df['flow_net_ratio_10'] * df['abs_flow_10']
    df['combo_concentration_flow'] = df['flow_concentration_5'] * df['abs_flow_5']

    # ── Round 3: Volatility regime ──
    log_hl = (np.log(df['High'] / df['Low']))**2
    log_co = (np.log(df['Close'] / df['Open']))**2
    gk_single = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df['gk_vol_10'] = gk_single.rolling(10).mean()
    df['gk_vol_20'] = gk_single.rolling(20).mean()
    df['gk_vol_50'] = gk_single.rolling(50).mean()

    df['parkinson_vol_10'] = (log_hl / (4 * np.log(2))).rolling(10).mean()
    df['parkinson_vol_20'] = (log_hl / (4 * np.log(2))).rolling(20).mean()

    roll_mean_50 = df['Close'].rolling(50).mean()
    roll_std_50 = df['Close'].rolling(50).std()
    df['boll_width_50'] = (4 * roll_std_50) / df['Close']
    df['boll_pct_50'] = (df['Close'] - (roll_mean_50 - 2*roll_std_50)) / (4*roll_std_50 + 1e-10)

    roll_mean_100 = df['Close'].rolling(100).mean()
    roll_std_100 = df['Close'].rolling(100).std()
    df['boll_width_100'] = (4 * roll_std_100) / df['Close']

    df['keltner_width_20'] = (2 * 1.5 * df['atr_20']) / df['Close']
    df['keltner_width_10'] = (2 * 1.5 * df['atr_10']) / df['Close']

    df['boll_inside_keltner'] = (df['boll_width_20'] < df['keltner_width_20']).astype(int)
    df['squeeze_intensity'] = df['keltner_width_20'] / (df['boll_width_20'] + 1e-10)

    df['vol_of_vol_20'] = df['boll_width_20'].rolling(20).std()
    df['vol_of_vol_50'] = df['boll_width_20'].rolling(50).std()

    df['boll_width_roc'] = (df['boll_width_20'] - df['boll_width_20'].shift(5)) / (df['boll_width_20'].shift(5) + 1e-10)
    df['gk_vol_roc'] = (df['gk_vol_10'] - df['gk_vol_10'].shift(10)) / (df['gk_vol_10'].shift(10) + 1e-10)

    df['boll_width_pct_50'] = df['boll_width_20'].rolling(50).rank(pct=True)
    df['boll_width_pct_100'] = df['boll_width_20'].rolling(100).rank(pct=True)
    df['boll_width_pct_200'] = df['boll_width_20'].rolling(200).rank(pct=True)
    df['atr_percentile_200'] = df['atr_10'].rolling(200).rank(pct=True)

    bw_mean = df['boll_width_20'].rolling(50).mean()
    bw_std = df['boll_width_20'].rolling(50).std()
    df['vol_zscore'] = (df['boll_width_20'] - bw_mean) / (bw_std + 1e-10)

    df['is_asian'] = ((hour >= 0) & (hour < 7)).astype(int)
    df['is_london'] = ((hour >= 7) & (hour < 13)).astype(int)
    df['is_ny'] = ((hour >= 13) & (hour < 20)).astype(int)
    df['is_overlap'] = ((hour >= 13) & (hour < 17)).astype(int)

    session_start = np.where(hour < 7, 0, np.where(hour < 13, 7, 13))
    df['mins_since_session'] = (hour - session_start) * 60 + minute
    df['mins_session_sin'] = np.sin(2 * np.pi * df['mins_since_session'] / 360)
    df['mins_session_cos'] = np.cos(2 * np.pi * df['mins_since_session'] / 360)

    df['combo_bw_flow3'] = df['boll_width_20'] * df['abs_flow_3']
    df['combo_bw_flow10'] = df['boll_width_20'] * df['abs_flow_10']
    df['combo_bw_dist8'] = df['boll_width_20'] * df['abs_dist_ema8']
    df['combo_bw_dist100'] = df['boll_width_20'] * df['abs_dist_ema100']
    df['combo_bw_spread'] = df['boll_width_20'] * df['abs_ema_spread']
    df['combo_bw_hour'] = df['boll_width_20'] * df['hour_cos']
    df['combo_bw_momentum'] = df['boll_width_20'] * df['flow_momentum'].abs()
    df['combo_bw_vol_roc'] = df['boll_width_20'] * df['boll_width_roc']
    df['combo_bw_squeeze'] = df['boll_width_20'] * df['squeeze_intensity']
    df['combo_gk_flow'] = df['gk_vol_10'] * df['abs_flow_10']
    df['combo_gk_dist'] = df['gk_vol_10'] * df['abs_dist_ema21']

    # Post-labeling combo features (folded in here)
    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']

    df = df.copy()
    return df.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE LIST — matches hedging_strategy_strict.py all_features + combos
# ═══════════════════════════════════════════════════════════════════════════════

ALL_FEATURES = [
    # Original
    'abs_ema_spread', 'abs_flow_10', 'abs_flow_3', 'abs_flow_5',
    'abs_dist_ema21', 'combo_dist_flow', 'abs_dist_ema8',
    'flow_momentum', 'ema_spread', 'flow_20',
    # Round 2
    'abs_flow_15', 'abs_flow_20', 'flow_zscore',
    'flow_net_ratio_5', 'flow_net_ratio_10', 'flow_concentration_5',
    'abs_dist_ema50', 'abs_dist_ema100',
    'abs_ema_spread_21_50', 'abs_ema_spread_50_100',
    'abs_price_zscore_20', 'abs_price_zscore_50',
    'boll_width_20', 'boll_pct_20', 'boll_outside_20',
    'flow_per_atr_3', 'flow_per_atr_10',
    'dist_ema8_per_atr', 'dist_ema21_per_atr', 'ema_spread_per_atr',
    'atr_percentile',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'combo_flow3_dist21', 'combo_flow10_dist8', 'combo_flow10_spread',
    'combo_flow_atr_pct', 'combo_flow_zscore_dist', 'combo_triple',
    'combo_net_ratio_flow', 'combo_concentration_flow',
    # Round 3
    'gk_vol_10', 'gk_vol_20', 'gk_vol_50',
    'parkinson_vol_10', 'parkinson_vol_20',
    'boll_width_50', 'boll_width_100', 'boll_pct_50',
    'keltner_width_20', 'keltner_width_10',
    'boll_inside_keltner', 'squeeze_intensity',
    'vol_of_vol_20', 'vol_of_vol_50',
    'boll_width_roc', 'gk_vol_roc',
    'boll_width_pct_50', 'boll_width_pct_100', 'boll_width_pct_200',
    'atr_percentile_200', 'vol_zscore',
    'is_asian', 'is_london', 'is_ny', 'is_overlap',
    'mins_since_session', 'mins_session_sin', 'mins_session_cos',
    'combo_bw_flow3', 'combo_bw_flow10', 'combo_bw_dist8',
    'combo_bw_dist100', 'combo_bw_spread', 'combo_bw_hour',
    'combo_bw_momentum', 'combo_bw_vol_roc', 'combo_bw_squeeze',
    'combo_gk_flow', 'combo_gk_dist',
    # Post-labeling combos
    'combo_flow_trend', 'combo_vol_imbalance',
    'combo_consistency_position', 'combo_body_reject',
    'combo_trend_volatility', 'combo_imbalance_momentum',
    'combo_position_consistency', 'combo_vol_flow',
]


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMBA — Label outcomes (replaces slow Python loop)
# ═══════════════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def label_outcomes_numba(closes, highs, lows, horizon, target, stop):
    """
    Binary labeling matching hedging_strategy_strict.py logic:
      1 = LONG TP hit first
      2 = SHORT TP hit first (only if LONG didn't)
      0 = neither
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = closes[i]

        # Check LONG: TP hit before SL?
        long_hit = False
        for j in range(i + 1, i + horizon + 1):
            if highs[j] >= entry + target:
                long_hit = True
                break
            if lows[j] <= entry - stop:
                break

        if long_hit:
            labels[i] = 1
            continue

        # Check SHORT (only if long didn't hit)
        for j in range(i + 1, i + horizon + 1):
            if lows[j] <= entry - target:
                labels[i] = 2
                break
            if highs[j] >= entry + stop:
                break

    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMBA — Reversal strategy simulation
# ═══════════════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def simulate_reversal(rf_probs, candle_dirs, closes, highs, lows,
                      rf_threshold, wait_window, cooldown,
                      rev_tp, rev_sl, hold_horizon):
    """
    Simulate wait-for-reversal strategy on test set.
    candle_dirs: 1=BULL, -1=BEAR, 0=DOJI
    Returns array of trade P&L values.
    """
    n = len(rf_probs)
    pnls = np.empty(n, dtype=np.float64)
    trade_count = 0
    cooldown_until = -1

    for i in range(n - wait_window - hold_horizon - 1):
        if i <= cooldown_until:
            continue
        if rf_probs[i] < rf_threshold:
            continue

        sig_dir = candle_dirs[i]
        if sig_dir == 0:
            continue

        # Wait for reversal candle
        rev_dir = -sig_dir
        entry_bar = -1
        for w in range(1, wait_window + 1):
            if i + w >= n:
                break
            if candle_dirs[i + w] == rev_dir:
                entry_bar = i + w
                break

        if entry_bar == -1:
            cooldown_until = i + cooldown
            continue

        entry_price = closes[entry_bar]
        is_long = rev_dir == 1

        if is_long:
            tp_price = entry_price + rev_tp
            sl_price = entry_price - rev_sl
        else:
            tp_price = entry_price - rev_tp
            sl_price = entry_price + rev_sl

        # Simulate bar-by-bar (SL checked first = strict)
        pnl = 0.0
        end_bar = min(entry_bar + hold_horizon + 1, n)
        result_found = False

        for j in range(entry_bar + 1, end_bar):
            if is_long:
                if lows[j] <= sl_price:
                    pnl = -rev_sl
                    result_found = True
                    break
                if highs[j] >= tp_price:
                    pnl = rev_tp
                    result_found = True
                    break
            else:
                if highs[j] >= sl_price:
                    pnl = -rev_sl
                    result_found = True
                    break
                if lows[j] <= tp_price:
                    pnl = rev_tp
                    result_found = True
                    break

        if not result_found:
            last_idx = min(entry_bar + hold_horizon, n - 1)
            if is_long:
                pnl = closes[last_idx] - entry_price
            else:
                pnl = entry_price - closes[last_idx]

        pnls[trade_count] = pnl
        trade_count += 1
        cooldown_until = i + cooldown

    return pnls[:trade_count]


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE — Full trial: label → train → predict → simulate
# ═══════════════════════════════════════════════════════════════════════════════

def run_trial(df_feat, split_idx, closes, highs, lows,
              model_target, model_stop, model_horizon,
              rf_threshold, rev_tp, rev_sl, wait_window, cooldown):
    """Run one full pipeline iteration."""

    # 1. Label with model params
    labels = label_outcomes_numba(closes, highs, lows,
                                  model_horizon, model_target, model_stop)

    # 2. Split (fixed purge gap for consistent test set)
    train_labels = labels[:split_idx]
    test_start = split_idx + PURGE_GAP
    test_labels = labels[test_start:]

    y_train = (train_labels != 0).astype(np.int32)

    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[test_start:]

    X_train = train_df[ALL_FEATURES].fillna(0).values
    X_test = test_df[ALL_FEATURES].fillna(0).values

    # Subsample train for speed
    rng = np.random.RandomState(42)
    n_train = len(X_train)
    idx = rng.choice(n_train, int(n_train * TRAIN_FRAC), replace=False)
    X_train_sub = X_train[idx]
    y_train_sub = y_train[idx]

    # 3. Train RF (binary: will price move?)
    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_DEPTH,
        min_samples_leaf=RF_MIN_LEAF, min_samples_split=RF_MIN_SPLIT,
        max_features='sqrt', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_sub, y_train_sub)

    # 4. Get probs on test set
    proba = rf.predict_proba(X_test)
    if proba.shape[1] < 2:
        return None
    rf_probs = proba[:, 1].astype(np.float64)

    # 5. Candle directions for test set
    test_closes = test_df['Close'].values.astype(np.float64)
    test_opens = test_df['Open'].values.astype(np.float64)
    test_highs = test_df['High'].values.astype(np.float64)
    test_lows = test_df['Low'].values.astype(np.float64)

    candle_dirs = np.where(test_closes > test_opens, 1,
                  np.where(test_closes < test_opens, -1, 0)).astype(np.int32)

    # 6. Simulate reversal strategy
    pnls = simulate_reversal(
        rf_probs, candle_dirs, test_closes, test_highs, test_lows,
        rf_threshold, wait_window, cooldown,
        rev_tp, rev_sl, model_horizon,
    )

    if len(pnls) < 20:
        return None

    # Metrics
    n = len(pnls)
    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())
    std_pnl = float(pnls.std()) if n > 1 else 1e-10
    t_stat = avg_pnl / (std_pnl / np.sqrt(n) + 1e-10)

    wins = int((pnls > 0).sum())
    losses = int((pnls < 0).sum())
    timeouts = n - wins - losses
    wr = wins / n * 100

    gross_w = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_l = float(np.abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-10
    pf = gross_w / gross_l

    cum = np.cumsum(pnls)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    return {
        'trades': n, 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'wr': round(wr, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 4),
        't_stat': round(t_stat, 2),
        'pf': round(pf, 2),
        'max_dd': round(max_dd, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def make_objective(df_feat, split_idx, closes, highs, lows):
    def objective(trial):
        # Model training params
        model_target  = trial.suggest_float('model_target', 1.0, 8.0, step=0.5)
        model_stop    = trial.suggest_float('model_stop', 0.5, 4.0, step=0.25)
        model_horizon = trial.suggest_int('model_horizon', 15, 60, step=5)
        rf_threshold  = trial.suggest_float('rf_threshold', 0.60, 0.90, step=0.05)

        # Reversal strategy params
        rev_tp      = trial.suggest_float('rev_tp', 0.5, 8.0, step=0.25)
        rev_sl      = trial.suggest_float('rev_sl', 0.5, 4.0, step=0.25)
        wait_window = trial.suggest_int('wait_window', 1, 5)
        cooldown    = trial.suggest_int('cooldown', 3, 10)

        result = run_trial(
            df_feat, split_idx, closes, highs, lows,
            model_target, model_stop, model_horizon,
            rf_threshold, rev_tp, rev_sl, wait_window, cooldown,
        )

        if result is None:
            return -999.0

        trial.set_user_attr('wr', result['wr'])
        trial.set_user_attr('total_pnl', result['total_pnl'])
        trial.set_user_attr('avg_pnl', result['avg_pnl'])
        trial.set_user_attr('pf', result['pf'])
        trial.set_user_attr('max_dd', result['max_dd'])
        trial.set_user_attr('trades', result['trades'])
        trial.set_user_attr('wins', result['wins'])
        trial.set_user_attr('losses', result['losses'])

        return result['total_pnl']

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics(m, label=''):
    if m is None:
        print(f'  {label}: No valid trades')
        return
    print(f'  {label}')
    print(f'    Trades:    {m["trades"]:>5d}   (W:{m["wins"]} L:{m["losses"]} T:{m["timeouts"]})')
    print(f'    Win Rate:  {m["wr"]:>5.1f}%')
    print(f'    Total P&L: ${m["total_pnl"]:>9,.2f}')
    print(f'    Avg P&L:   ${m["avg_pnl"]:>9.4f} per trade')
    print(f'    t-stat:    {m["t_stat"]:>6.2f}')
    print(f'    PF:        {m["pf"]:>6.2f}')
    print(f'    MaxDD:     ${m["max_dd"]:>9,.2f}')


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    print('=' * 72)
    print('  UNIFIED REVERSAL STRATEGY OPTIMIZER')
    print(f'  Model: label + RF binary (trees={RF_TREES}, depth={RF_DEPTH})')
    print(f'  Strategy: wait-for-reversal')
    print(f'  Optimize: 8 params | {N_TRIALS} trials | {len(ALL_FEATURES)} features')
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
    print(f'  {len(df):,} bars: {df.index[0]} -> {df.index[-1]}')

    # ── Features (done once) ──
    print('  Engineering features...')
    df = create_microstructure_features(df)
    print(f'  {len(df):,} bars after warmup | {len(ALL_FEATURES)} features')

    split = int(len(df) * 0.6)
    print(f'  Train: {split:,} | Purge: {PURGE_GAP} | '
          f'Test: {len(df) - split - PURGE_GAP:,}')

    # Pre-extract arrays (used for all trials)
    closes = df['Close'].values.astype(np.float64)
    highs = df['High'].values.astype(np.float64)
    lows = df['Low'].values.astype(np.float64)

    # ── Warm up Numba ──
    print('\n  Compiling Numba...')
    _ = label_outcomes_numba(closes[:2000], highs[:2000], lows[:2000], 30, 3.0, 1.5)
    dummy_probs = np.ones(2000) * 0.8
    dummy_dirs = np.ones(2000, dtype=np.int32)
    _ = simulate_reversal(dummy_probs, dummy_dirs,
                          closes[:2000], highs[:2000], lows[:2000],
                          0.7, 3, 5, 3.0, 1.0, 30)
    print('  Done.')

    # ── Baseline ──
    print(f'\n{"="*72}')
    print('  BASELINE (model: TGT=$3, SL=$1.5, HOR=30, TH=0.75)')
    print('  (strategy: TP=$3, SL=$1, wait=5, cooldown=5)')
    print(f'{"="*72}')

    base = run_trial(df, split, closes, highs, lows,
                     3.0, 1.5, 30,      # model params
                     0.75, 3.0, 1.0,    # strategy params
                     5, 5)              # wait, cooldown
    print_metrics(base, 'Original params')

    # ── Optuna ──
    print(f'\n{"="*72}')
    print(f'  OPTUNA OPTIMIZATION ({N_TRIALS} trials)')
    print(f'  Model:    model_target=[1-8], model_stop=[0.5-4], horizon=[15-60]')
    print(f'  Signal:   rf_threshold=[0.60-0.90]')
    print(f'  Strategy: rev_tp=[0.5-8], rev_sl=[0.5-4], wait=[1-5], cooldown=[3-10]')
    print(f'  Speed:    RF={RF_TREES} trees, {TRAIN_FRAC*100:.0f}% train subsample')
    print(f'{"="*72}')

    objective = make_objective(df, split, closes, highs, lows)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t1 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)
    elapsed = time.time() - t1

    bp = study.best_params
    print(f'\n\n  Completed {N_TRIALS} trials in {elapsed:.1f}s '
          f'({elapsed/N_TRIALS:.1f}s/trial)')

    print(f'\n  BEST PARAMETERS:')
    print(f'    --- Model ---')
    print(f'    model_target:   ${bp["model_target"]:.1f}')
    print(f'    model_stop:     ${bp["model_stop"]:.2f}')
    print(f'    model_horizon:  {bp["model_horizon"]} bars')
    print(f'    rf_threshold:   {bp["rf_threshold"]:.2f}')
    print(f'    --- Strategy ---')
    print(f'    rev_tp:         ${bp["rev_tp"]:.2f}')
    print(f'    rev_sl:         ${bp["rev_sl"]:.2f}')
    print(f'    wait_window:    {bp["wait_window"]}')
    print(f'    cooldown:       {bp["cooldown"]}')
    rr = bp['rev_tp'] / bp['rev_sl']
    print(f'    RR:             {rr:.1f}:1')

    # ── Evaluate best ──
    print(f'\n{"─"*72}')
    print('  OPTIMIZED RESULTS')
    print(f'{"─"*72}')
    opt = run_trial(df, split, closes, highs, lows,
                    bp['model_target'], bp['model_stop'], bp['model_horizon'],
                    bp['rf_threshold'], bp['rev_tp'], bp['rev_sl'],
                    bp['wait_window'], bp['cooldown'])
    print_metrics(opt, 'Test -- OPTIMIZED')

    # ── Comparison ──
    if base and opt:
        print(f'\n{"─"*72}')
        print('  BASELINE vs OPTIMIZED')
        print(f'{"─"*72}')
        print(f'  {"Metric":<15s} {"Baseline":>12s} {"Optimized":>12s}')
        print(f'  {"─"*42}')
        for k, lab in [('wr', 'Win Rate %'), ('total_pnl', 'Total P&L $'),
                       ('avg_pnl', 'Avg P&L $'), ('t_stat', 't-stat'),
                       ('pf', 'Profit Factor'), ('max_dd', 'Max DD $'),
                       ('trades', 'Trades')]:
            bv = base.get(k, 'N/A')
            ov = opt.get(k, 'N/A')
            print(f'  {lab:<15s} {str(bv):>12s} {str(ov):>12s}')

    # ── All profitable trials ──
    print(f'\n{"="*72}')
    print('  ALL PROFITABLE TRIALS (ranked by P&L)')
    print(f'{"="*72}\n')

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > 0]
    trials_df = trials_df.sort_values('value', ascending=False).reset_index(drop=True)

    rows = []
    for _, row in trials_df.iterrows():
        rows.append({
            'rank': len(rows) + 1,
            'pnl': row['value'],
            'model_tgt': row['params_model_target'],
            'model_sl': row['params_model_stop'],
            'model_hor': int(row['params_model_horizon']),
            'rf_th': row['params_rf_threshold'],
            'rev_tp': row['params_rev_tp'],
            'rev_sl': row['params_rev_sl'],
            'wait': int(row['params_wait_window']),
            'cd': int(row['params_cooldown']),
            'WR%': row.get('user_attrs_wr', 0),
            'PF': row.get('user_attrs_pf', 0),
            'trades': int(row.get('user_attrs_trades', 0)),
            'wins': int(row.get('user_attrs_wins', 0)),
            'losses': int(row.get('user_attrs_losses', 0)),
        })
    results_df = pd.DataFrame(rows)

    print(f'  {"#":>3s}  {"P&L":>9s}  {"mTGT":>5s} {"mSL":>5s} {"mH":>3s} {"TH":>4s}  '
          f'{"rTP":>5s} {"rSL":>5s} {"W":>2s} {"C":>2s}  '
          f'{"WR%":>5s}  {"PF":>5s}  {"Trd":>5s}')
    print(f'  {"─"*80}')
    for _, r in results_df.head(50).iterrows():
        print(f'  {int(r["rank"]):3d}  ${r["pnl"]:>8,.2f}  '
              f'${r["model_tgt"]:3.1f} ${r["model_sl"]:4.2f} {int(r["model_hor"]):3d} {r["rf_th"]:4.2f}  '
              f'${r["rev_tp"]:4.2f} ${r["rev_sl"]:4.2f} {int(r["wait"]):2d} {int(r["cd"]):2d}  '
              f'{r["WR%"]:5.1f}  {r["PF"]:5.2f}  {int(r["trades"]):5d}')

    # ── Save ──
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'optuna_reversal_unified.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'\n  Saved {len(results_df)} profitable trials -> {csv_path}')

    total_time = time.time() - t0
    print(f'\n{"="*72}')
    print(f'  SUMMARY')
    print(f'{"="*72}')
    print(f'  Total time: {total_time:.1f}s')
    print(f'  RF: trees={RF_TREES}, depth={RF_DEPTH} (fast search)')
    print(f'  Train subsample: {TRAIN_FRAC*100:.0f}%')
    if opt:
        print(f'  Best P&L: ${opt["total_pnl"]:,.2f} | WR: {opt["wr"]}% | '
              f'PF: {opt["pf"]} | t: {opt["t_stat"]}')
    print(f'{"="*72}')
