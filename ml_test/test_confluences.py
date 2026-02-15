"""
Confluence Filter Testing for Directional Strategy
====================================================
Tests rule-based filters on top of ML signals to find
which confluences improve win rate without killing trade count.

Uses the same data/model pipeline as backtest_directional.py.
"""

import numpy as np
import pandas as pd
import numba as nb
import time
import warnings
import os

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ── Same params as backtest_directional ──
TARGET       = 2.0
STOP         = 0.50
HORIZON      = 15
RF_THRESHOLD = 0.40
RF_DEPTH     = 20
RF_TREES     = 200
SPREAD       = 0.30

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'XAUUSD1.csv')

# ── Feature engineering (identical to backtest_directional.py) ──
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

    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']

    # ── Extra features for confluence rules (not used by RF) ──
    df['rsi_14'] = compute_rsi(df['Close'], 14)
    df['rsi_7'] = compute_rsi(df['Close'], 7)

    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['ema_200'] = df['Close'].ewm(span=200).mean()
    df['above_ema50'] = (df['Close'] > df['ema_50']).astype(int)
    df['above_ema200'] = (df['Close'] > df['ema_200']).astype(int)
    df['ema50_trend'] = ((df['ema_8'] > df['ema_50']) & (df['Close'] > df['ema_8'])).astype(int) - \
                         ((df['ema_8'] < df['ema_50']) & (df['Close'] < df['ema_8'])).astype(int)

    df['prev_body'] = df['body'].shift(1)
    df['prev_range'] = df['range'].shift(1)
    df['engulfing_bull'] = ((df['body'] > 0) & (df['prev_body'] < 0) &
                            (df['abs_body'] > df['prev_range'])).astype(int)
    df['engulfing_bear'] = ((df['body'] < 0) & (df['prev_body'] > 0) &
                            (df['abs_body'] > df['prev_range'])).astype(int)

    df['vol_spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int) if 'Volume' in df else 0

    df['hour'] = df.index.hour

    return df.dropna()


def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ── Higher Timeframe Features ──────────────────────────────────────────────
def create_htf_features(df):
    """
    Resample 1min bars to 15min, 1H, 4H, Daily.
    Compute trend/momentum/structure on each HTF.
    Merge back to 1min index with shift(1) to avoid look-ahead.
    """
    htf_dfs = {}

    timeframes = {
        '15min': '15min',
        '1h':    '1h',
        '4h':    '4h',
        '1d':    '1D',
    }

    for tf_name, resample_rule in timeframes.items():
        vol_col = 'Volume' if 'Volume' in df.columns else 'TickVol'
        htf = df.resample(resample_rule).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            vol_col: 'sum',
        }).dropna()

        htf[f'{tf_name}_ema8']  = htf['Close'].ewm(span=8).mean()
        htf[f'{tf_name}_ema21'] = htf['Close'].ewm(span=21).mean()
        htf[f'{tf_name}_ema50'] = htf['Close'].ewm(span=50).mean()

        htf[f'{tf_name}_trend'] = (
            ((htf['Close'] > htf[f'{tf_name}_ema8']) &
             (htf[f'{tf_name}_ema8'] > htf[f'{tf_name}_ema21'])).astype(int) -
            ((htf['Close'] < htf[f'{tf_name}_ema8']) &
             (htf[f'{tf_name}_ema8'] < htf[f'{tf_name}_ema21'])).astype(int)
        )

        htf[f'{tf_name}_trend50'] = (
            (htf['Close'] > htf[f'{tf_name}_ema50']).astype(int) -
            (htf['Close'] < htf[f'{tf_name}_ema50']).astype(int)
        )

        htf[f'{tf_name}_dist_ema21'] = (htf['Close'] - htf[f'{tf_name}_ema21']) / htf['Close']
        htf[f'{tf_name}_rsi'] = compute_rsi(htf['Close'], 14)

        htf_body = htf['Close'] - htf['Open']
        htf_flow = htf_body / htf['Close']
        htf[f'{tf_name}_flow_3'] = htf_flow.rolling(3).sum()
        htf[f'{tf_name}_flow_5'] = htf_flow.rolling(5).sum()

        htf_range = htf['High'] - htf['Low']
        htf[f'{tf_name}_close_pos'] = (htf['Close'] - htf['Low']) / (htf_range + 1e-10)

        htf_atr3  = htf_range.rolling(3).mean()
        htf_atr10 = htf_range.rolling(10).mean()
        htf[f'{tf_name}_vol_ratio'] = htf_atr3 / (htf_atr10 + 1e-10)
        htf[f'{tf_name}_vol_exp'] = (htf_range > htf_atr10 * 1.2).astype(int)
        htf[f'{tf_name}_range_pct'] = htf_range / htf['Close']

        feat_cols = [c for c in htf.columns if c.startswith(f'{tf_name}_')]
        htf_feat = htf[feat_cols].shift(1)
        htf_dfs[tf_name] = htf_feat

    result = df.copy()
    for tf_name, htf_feat in htf_dfs.items():
        htf_reindexed = htf_feat.reindex(result.index, method='ffill')
        result = pd.concat([result, htf_reindexed], axis=1)

    return result


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

HTF_FEATURES = []
for _tf in ['15min', '1h', '4h', '1d']:
    HTF_FEATURES.extend([
        f'{_tf}_trend', f'{_tf}_trend50', f'{_tf}_dist_ema21',
        f'{_tf}_rsi', f'{_tf}_flow_3', f'{_tf}_flow_5',
        f'{_tf}_close_pos', f'{_tf}_vol_ratio', f'{_tf}_vol_exp',
        f'{_tf}_range_pct',
    ])

FEATURE_NAMES = BASE_FEATURES + HTF_FEATURES


@nb.njit(cache=True)
def label_directional(closes, highs, lows, horizon, target, stop):
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32)
    for i in range(n - horizon):
        entry = closes[i]
        long_tp  = entry + target
        long_sl  = entry - stop
        short_tp = entry - target
        short_sl = entry + stop
        for j in range(i + 1, min(i + horizon + 1, n)):
            if highs[j] >= long_tp:
                labels[i] = 1
                break
            if lows[j] <= short_tp:
                labels[i] = 2
                break
            if lows[j] <= long_sl:
                for k in range(j, min(i + horizon + 1, n)):
                    if lows[k] <= short_tp:
                        labels[i] = 2
                        break
                    if highs[k] >= short_sl:
                        break
                break
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
#  CONFLUENCE FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def define_confluence_filters():
    """
    Each filter returns (long_mask, short_mask) booleans per bar.
    True = allow trade in that direction.
    """
    filters = {}

    # ── 1. TREND ALIGNMENT — EMA8 > EMA21 for LONG, < for SHORT ──
    filters['trend_align'] = {
        'desc': 'Trade with EMA8/EMA21 trend',
        'long':  lambda df: df['trend_align'] == 1,
        'short': lambda df: df['trend_align'] == -1,
    }

    # ── 2. STRONG TREND — EMA8 > EMA50 for LONG ──
    filters['ema50_trend'] = {
        'desc': 'Trade with EMA8/EMA50 trend',
        'long':  lambda df: df['ema50_trend'] == 1,
        'short': lambda df: df['ema50_trend'] == -1,
    }

    # ── 3. RSI FILTER — Avoid overbought/oversold entries ──
    filters['rsi_neutral'] = {
        'desc': 'RSI(14) 30-70 (no extremes)',
        'long':  lambda df: df['rsi_14'] < 70,
        'short': lambda df: df['rsi_14'] > 30,
    }

    # ── 4. RSI MOMENTUM — RSI must confirm direction ──
    filters['rsi_confirm'] = {
        'desc': 'RSI(14) > 50 for LONG, < 50 for SHORT',
        'long':  lambda df: df['rsi_14'] > 50,
        'short': lambda df: df['rsi_14'] < 50,
    }

    # ── 5. RSI MOMENTUM STRICT ──
    filters['rsi_confirm_strict'] = {
        'desc': 'RSI(14) > 55 for LONG, < 45 for SHORT',
        'long':  lambda df: df['rsi_14'] > 55,
        'short': lambda df: df['rsi_14'] < 45,
    }

    # ── 6. VOLATILITY — Only trade in expanding vol ──
    filters['vol_expansion'] = {
        'desc': 'Vol ratio > 1.0 (above-avg volatility)',
        'long':  lambda df: df['vol_ratio'] > 1.0,
        'short': lambda df: df['vol_ratio'] > 1.0,
    }

    # ── 7. NO LOW-VOL — Skip during contracting vol ──
    filters['no_low_vol'] = {
        'desc': 'Skip vol_contraction bars',
        'long':  lambda df: df['vol_contraction'] == 0,
        'short': lambda df: df['vol_contraction'] == 0,
    }

    # ── 8. FLOW CONFIRMS DIRECTION ──
    filters['flow_confirm'] = {
        'desc': 'flow_3 > 0 for LONG, < 0 for SHORT',
        'long':  lambda df: df['flow_3'] > 0,
        'short': lambda df: df['flow_3'] < 0,
    }

    # ── 9. FLOW5 CONFIRMS ──
    filters['flow5_confirm'] = {
        'desc': 'flow_5 > 0 for LONG, < 0 for SHORT',
        'long':  lambda df: df['flow_5'] > 0,
        'short': lambda df: df['flow_5'] < 0,
    }

    # ── 10. BODY PCT — Only take clean candles (not doji/spinning) ──
    filters['clean_candle'] = {
        'desc': 'body_pct > 0.4 (not doji)',
        'long':  lambda df: df['body_pct'] > 0.4,
        'short': lambda df: df['body_pct'] > 0.4,
    }

    # ── 11. STRONG CANDLE — Big decisive move ──
    filters['strong_candle'] = {
        'desc': 'big_body == 1',
        'long':  lambda df: df['big_body'] == 1,
        'short': lambda df: df['big_body'] == 1,
    }

    # ── 12. NO REJECTION — Avoid long if upper wick rejection ──
    filters['no_reject'] = {
        'desc': 'No upper reject for LONG / no lower reject for SHORT',
        'long':  lambda df: df['upper_reject'] == 0,
        'short': lambda df: df['lower_reject'] == 0,
    }

    # ── 13. IMBALANCE CONFIRMS ──
    filters['imbalance_confirm'] = {
        'desc': 'imbalance_3 > 0 for LONG, < 0 for SHORT',
        'long':  lambda df: df['imbalance_3'] > 0,
        'short': lambda df: df['imbalance_3'] < 0,
    }

    # ── 14. ENGULFING CANDLE ──
    filters['engulfing'] = {
        'desc': 'Engulfing pattern present',
        'long':  lambda df: df['engulfing_bull'] == 1,
        'short': lambda df: df['engulfing_bear'] == 1,
    }

    # ── 15. CLOSE POSITION — Long if close near high, short near low ──
    filters['close_pos_confirm'] = {
        'desc': 'close_position > 0.6 for LONG, < 0.4 for SHORT',
        'long':  lambda df: df['close_position'] > 0.6,
        'short': lambda df: df['close_position'] < 0.4,
    }

    # ── 16. SESSION FILTER — London + NY overlap only ──
    filters['session_london_ny'] = {
        'desc': 'Only trade hours 7-17 UTC (London+NY)',
        'long':  lambda df: (df['hour'] >= 7) & (df['hour'] <= 17),
        'short': lambda df: (df['hour'] >= 7) & (df['hour'] <= 17),
    }

    # ── 17. AVOID ASIA ──
    filters['no_asia'] = {
        'desc': 'Skip hours 0-6 UTC',
        'long':  lambda df: df['hour'] >= 7,
        'short': lambda df: df['hour'] >= 7,
    }

    # ── 18. FLOW ACCELERATION POSITIVE ──
    filters['flow_accel_confirm'] = {
        'desc': 'flow_accel > 0 for LONG, < 0 for SHORT',
        'long':  lambda df: df['flow_accel'] > 0,
        'short': lambda df: df['flow_accel'] < 0,
    }

    # ── 19. ATR EXPANDING ──
    filters['atr_expanding'] = {
        'desc': 'atr_roc > 0 (ATR increasing)',
        'long':  lambda df: df['atr_roc'] > 0,
        'short': lambda df: df['atr_roc'] > 0,
    }

    # ── 20. HIGHER THRESHOLD ──
    filters['high_prob_0.50'] = {
        'desc': 'Raise ML threshold to 0.50',
        'threshold': 0.50,
    }
    filters['high_prob_0.55'] = {
        'desc': 'Raise ML threshold to 0.55',
        'threshold': 0.55,
    }
    filters['high_prob_0.60'] = {
        'desc': 'Raise ML threshold to 0.60',
        'threshold': 0.60,
    }

    # ── 21. CONSISTENCY — Bars going same direction ──
    filters['consistency_high'] = {
        'desc': 'consistency_3 >= 3 (all 3 bars same dir)',
        'long':  lambda df: (df['up_count_3'] >= 3),
        'short': lambda df: (df['up_count_3'] <= 0),
    }

    # ── 22. SQUEEZE BREAKOUT — Range was compressed, now expanding ──
    filters['squeeze_breakout'] = {
        'desc': 'range_squeeze < 0.5 + vol_ratio > 1.2',
        'long':  lambda df: (df['range_squeeze'] < 0.5) & (df['vol_ratio'] > 1.2),
        'short': lambda df: (df['range_squeeze'] < 0.5) & (df['vol_ratio'] > 1.2),
    }

    # ── 23. EMA200 TREND ──
    filters['ema200_trend'] = {
        'desc': 'Above EMA200 for LONG, below for SHORT',
        'long':  lambda df: df['above_ema200'] == 1,
        'short': lambda df: df['above_ema200'] == 0,
    }

    # ── COMBOS ──

    # ── C1. TREND + RSI ──
    filters['COMBO_trend+rsi'] = {
        'desc': 'EMA trend align + RSI confirms direction',
        'long':  lambda df: (df['trend_align'] == 1) & (df['rsi_14'] > 50),
        'short': lambda df: (df['trend_align'] == -1) & (df['rsi_14'] < 50),
    }

    # ── C2. TREND + FLOW ──
    filters['COMBO_trend+flow'] = {
        'desc': 'EMA trend align + flow_3 confirms',
        'long':  lambda df: (df['trend_align'] == 1) & (df['flow_3'] > 0),
        'short': lambda df: (df['trend_align'] == -1) & (df['flow_3'] < 0),
    }

    # ── C3. TREND + RSI + SESSION ──
    filters['COMBO_trend+rsi+session'] = {
        'desc': 'Trend + RSI confirm + London/NY only',
        'long':  lambda df: (df['trend_align'] == 1) & (df['rsi_14'] > 50) & (df['hour'] >= 7) & (df['hour'] <= 17),
        'short': lambda df: (df['trend_align'] == -1) & (df['rsi_14'] < 50) & (df['hour'] >= 7) & (df['hour'] <= 17),
    }

    # ── C4. FLOW + VOL + NO REJECT ──
    filters['COMBO_flow+vol+noreject'] = {
        'desc': 'Flow confirms + expanding vol + no rejection',
        'long':  lambda df: (df['flow_3'] > 0) & (df['vol_ratio'] > 1.0) & (df['upper_reject'] == 0),
        'short': lambda df: (df['flow_3'] < 0) & (df['vol_ratio'] > 1.0) & (df['lower_reject'] == 0),
    }

    # ── C5. TREND + IMBALANCE ──
    filters['COMBO_trend+imbalance'] = {
        'desc': 'Trend align + order imbalance confirms',
        'long':  lambda df: (df['trend_align'] == 1) & (df['imbalance_3'] > 0),
        'short': lambda df: (df['trend_align'] == -1) & (df['imbalance_3'] < 0),
    }

    # ── C6. FULL CONFLUENCE ──
    filters['COMBO_full'] = {
        'desc': 'Trend + RSI + Flow + No reject',
        'long':  lambda df: (df['trend_align'] == 1) & (df['rsi_14'] > 50) & (df['flow_3'] > 0) & (df['upper_reject'] == 0),
        'short': lambda df: (df['trend_align'] == -1) & (df['rsi_14'] < 50) & (df['flow_3'] < 0) & (df['lower_reject'] == 0),
    }

    # ── C7. SESSION + VOL ──
    filters['COMBO_session+vol'] = {
        'desc': 'London/NY + above-avg vol',
        'long':  lambda df: (df['hour'] >= 7) & (df['hour'] <= 17) & (df['vol_ratio'] > 1.0),
        'short': lambda df: (df['hour'] >= 7) & (df['hour'] <= 17) & (df['vol_ratio'] > 1.0),
    }

    # ── C8. CLOSE POS + TREND ──
    filters['COMBO_closepos+trend'] = {
        'desc': 'Close near high/low + trend aligned',
        'long':  lambda df: (df['close_position'] > 0.6) & (df['trend_align'] == 1),
        'short': lambda df: (df['close_position'] < 0.4) & (df['trend_align'] == -1),
    }

    # ══════════════════════════════════════════════════════════════════════
    #  COMBOS OF TOP-PERFORMING SINGLES
    # ══════════════════════════════════════════════════════════════════════

    # ── T1. RSI neutral + No Asia ──
    filters['TOP_rsi_neutral+no_asia'] = {
        'desc': 'RSI 30-70 + skip Asia (0-6 UTC)',
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7),
    }

    # ── T2. RSI neutral + No low vol ──
    filters['TOP_rsi_neutral+no_low_vol'] = {
        'desc': 'RSI 30-70 + skip vol_contraction bars',
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['vol_contraction'] == 0),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['vol_contraction'] == 0),
    }

    # ── T3. No Asia + No low vol ──
    filters['TOP_no_asia+no_low_vol'] = {
        'desc': 'Skip Asia + skip vol_contraction bars',
        'long':  lambda df: (df['hour'] >= 7) & (df['vol_contraction'] == 0),
        'short': lambda df: (df['hour'] >= 7) & (df['vol_contraction'] == 0),
    }

    # ── T4. Best 3 singles ──
    filters['TOP_rsi+noasia+novol'] = {
        'desc': 'RSI 30-70 + No Asia + No low vol',
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0),
    }

    # ── T5. Squeeze breakout + No Asia ──
    filters['TOP_squeeze+no_asia'] = {
        'desc': 'Squeeze breakout + skip Asia',
        'long':  lambda df: (df['range_squeeze'] == 0) & (df['hour'] >= 7),
        'short': lambda df: (df['range_squeeze'] == 0) & (df['hour'] >= 7),
    }

    # ── T6. RSI neutral + Vol expansion ──
    filters['TOP_rsi_neutral+vol_exp'] = {
        'desc': 'RSI 30-70 + vol_expansion active',
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['vol_expansion'] == 1),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['vol_expansion'] == 1),
    }

    # ── T7. Session+Vol + RSI neutral ──
    filters['TOP_session_vol+rsi'] = {
        'desc': 'London/NY + above-avg vol + RSI 30-70',
        'long':  lambda df: (df['hour'] >= 7) & (df['hour'] <= 17) & (df['vol_ratio'] > 1.0) & (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70),
        'short': lambda df: (df['hour'] >= 7) & (df['hour'] <= 17) & (df['vol_ratio'] > 1.0) & (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70),
    }

    # ── T8. RSI neutral + No Asia + Strong candle ──
    filters['TOP_rsi+noasia+strong'] = {
        'desc': 'RSI 30-70 + No Asia + body > 60%',
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['big_body'] == 1),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['big_body'] == 1),
    }

    # ── T9. Best 3 + Vol expansion ──
    filters['TOP_best3+vol_exp'] = {
        'desc': 'RSI 30-70 + No Asia + No low vol + Vol expanding',
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0) & (df['vol_expansion'] == 1),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0) & (df['vol_expansion'] == 1),
    }

    # ── T10. Squeeze + RSI neutral + No Asia ──
    filters['TOP_squeeze+rsi+noasia'] = {
        'desc': 'Squeeze breakout + RSI 30-70 + No Asia',
        'long':  lambda df: (df['range_squeeze'] == 0) & (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7),
        'short': lambda df: (df['range_squeeze'] == 0) & (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7),
    }

    # ══════════════════════════════════════════════════════════════════════
    #  HIGH PROB 0.55 + TOP FILTERS
    # ══════════════════════════════════════════════════════════════════════

    # ── HP1. 0.55 + No Asia ──
    filters['HP55_no_asia'] = {
        'desc': 'Prob >= 0.55 + skip Asia (hour >= 7)',
        'threshold': 0.55,
        'long':  lambda df: df['hour'] >= 7,
        'short': lambda df: df['hour'] >= 7,
    }

    # ── HP2. 0.55 + No low vol ──
    filters['HP55_no_low_vol'] = {
        'desc': 'Prob >= 0.55 + skip vol_contraction',
        'threshold': 0.55,
        'long':  lambda df: df['vol_contraction'] == 0,
        'short': lambda df: df['vol_contraction'] == 0,
    }

    # ── HP3. 0.55 + Vol expansion ──
    filters['HP55_vol_exp'] = {
        'desc': 'Prob >= 0.55 + vol_expansion active',
        'threshold': 0.55,
        'long':  lambda df: df['vol_expansion'] == 1,
        'short': lambda df: df['vol_expansion'] == 1,
    }

    # ── HP4. 0.55 + RSI neutral ──
    filters['HP55_rsi_neutral'] = {
        'desc': 'Prob >= 0.55 + RSI 30-70',
        'threshold': 0.55,
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70),
    }

    # ── HP5. 0.55 + No Asia + No low vol ──
    filters['HP55_noasia+novol'] = {
        'desc': 'Prob >= 0.55 + No Asia + No low vol',
        'threshold': 0.55,
        'long':  lambda df: (df['hour'] >= 7) & (df['vol_contraction'] == 0),
        'short': lambda df: (df['hour'] >= 7) & (df['vol_contraction'] == 0),
    }

    # ── HP6. 0.55 + Session vol ──
    filters['HP55_session_vol'] = {
        'desc': 'Prob >= 0.55 + London/NY + vol > 1.0',
        'threshold': 0.55,
        'long':  lambda df: (df['hour'] >= 7) & (df['hour'] <= 17) & (df['vol_ratio'] > 1.0),
        'short': lambda df: (df['hour'] >= 7) & (df['hour'] <= 17) & (df['vol_ratio'] > 1.0),
    }

    # ── HP7. 0.55 + RSI neutral + No Asia ──
    filters['HP55_rsi+noasia'] = {
        'desc': 'Prob >= 0.55 + RSI 30-70 + No Asia',
        'threshold': 0.55,
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7),
    }

    # ── HP8. 0.55 + RSI neutral + No Asia + No low vol ──
    filters['HP55_rsi+noasia+novol'] = {
        'desc': 'Prob >= 0.55 + RSI 30-70 + No Asia + No low vol',
        'threshold': 0.55,
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0),
    }

    # ── HP9. 0.55 + RSI neutral + Vol expansion ──
    filters['HP55_rsi+vol_exp'] = {
        'desc': 'Prob >= 0.55 + RSI 30-70 + Vol expanding',
        'threshold': 0.55,
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['vol_expansion'] == 1),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['vol_expansion'] == 1),
    }

    # ── HP10. 0.55 + Best3 + Vol expansion ──
    filters['HP55_best3+vol_exp'] = {
        'desc': 'Prob >= 0.55 + RSI 30-70 + No Asia + No low vol + Vol exp',
        'threshold': 0.55,
        'long':  lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0) & (df['vol_expansion'] == 1),
        'short': lambda df: (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) & (df['hour'] >= 7) & (df['vol_contraction'] == 0) & (df['vol_expansion'] == 1),
    }

    return filters


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_filter(probs_long, probs_short, dir_labels, closes, df_test,
                    threshold, filt=None):
    """
    Apply a confluence filter and compute performance metrics.
    Returns dict with stats or None if no trades.
    """
    n = len(probs_long)
    trades_pnl = []
    trades_dir = []
    trades_dt = []

    datetimes = df_test.index

    # Determine threshold override
    th = threshold
    if filt and 'threshold' in filt:
        th = filt['threshold']

    for idx in range(n):
        # Long signal
        if probs_long[idx] >= th:
            allowed = True
            if filt and 'long' in filt:
                allowed = bool(filt['_long_mask'][idx])
            if allowed:
                lbl = dir_labels[idx]
                if lbl == 1:
                    trades_pnl.append(TARGET - SPREAD)
                elif lbl == 2:
                    trades_pnl.append(-STOP - SPREAD)
                else:
                    trades_pnl.append(-SPREAD)
                trades_dir.append('LONG')
                trades_dt.append(datetimes[idx])

        # Short signal
        if probs_short[idx] >= th:
            allowed = True
            if filt and 'short' in filt:
                allowed = bool(filt['_short_mask'][idx])
            if allowed:
                lbl = dir_labels[idx]
                if lbl == 2:
                    trades_pnl.append(TARGET - SPREAD)
                elif lbl == 1:
                    trades_pnl.append(-STOP - SPREAD)
                else:
                    trades_pnl.append(-SPREAD)
                trades_dir.append('SHORT')
                trades_dt.append(datetimes[idx])

    if len(trades_pnl) == 0:
        return None

    pnl = np.array(trades_pnl)
    n_trades = len(pnl)
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    wr = wins / n_trades * 100

    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    std_pnl = pnl.std() if n_trades > 1 else 1e-10
    t_stat = avg_pnl / (std_pnl / np.sqrt(n_trades) + 1e-10)

    win_pnl = pnl[pnl > 0]
    loss_pnl = pnl[pnl < 0]
    avg_win = win_pnl.mean() if len(win_pnl) > 0 else 0
    avg_loss = loss_pnl.mean() if len(loss_pnl) > 0 else -1
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    gross_w = win_pnl.sum() if len(win_pnl) > 0 else 0
    gross_l = abs(loss_pnl.sum()) if len(loss_pnl) > 0 else 1e-10
    pf = gross_w / gross_l

    cum = np.cumsum(pnl)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    n_long = sum(1 for d in trades_dir if d == 'LONG')
    n_short = sum(1 for d in trades_dir if d == 'SHORT')

    # Period breakdowns: monthly, weekly, daily
    def compute_period_wr(df_tmp, period_col):
        breakdown = {}
        for period, grp in df_tmp.groupby(period_col):
            p_wins = (grp['pnl'] > 0).sum()
            p_total = len(grp)
            breakdown[str(period)] = {
                'wr': round(p_wins / p_total * 100, 1),
                'trades': p_total,
                'pnl': round(grp['pnl'].sum(), 2),
            }
        return breakdown

    def compute_stability(breakdown):
        if len(breakdown) >= 2:
            wrs = [v['wr'] for v in breakdown.values()]
            profitable = sum(1 for v in breakdown.values() if v['pnl'] > 0)
            return {
                'mean': round(np.mean(wrs), 1),
                'std': round(np.std(wrs), 1),
                'min': round(np.min(wrs), 1),
                'max': round(np.max(wrs), 1),
                'profitable': profitable,
                'total': len(breakdown),
            }
        return {
            'mean': round(wr, 1), 'std': 0.0, 'min': round(wr, 1),
            'max': round(wr, 1),
            'profitable': 1 if total_pnl > 0 else 0, 'total': 1,
        }

    monthly_wr = {}
    weekly_wr = {}
    daily_wr = {}
    m_stab = w_stab = d_stab = compute_stability({})

    if len(trades_dt) > 0:
        df_tmp = pd.DataFrame({'pnl': trades_pnl, 'dt': trades_dt})
        df_tmp['month'] = df_tmp['dt'].dt.to_period('M')
        df_tmp['week'] = df_tmp['dt'].dt.to_period('W')
        df_tmp['day'] = df_tmp['dt'].dt.to_period('D')

        monthly_wr = compute_period_wr(df_tmp, 'month')
        weekly_wr  = compute_period_wr(df_tmp, 'week')
        daily_wr   = compute_period_wr(df_tmp, 'day')

        m_stab = compute_stability(monthly_wr)
        w_stab = compute_stability(weekly_wr)
        d_stab = compute_stability(daily_wr)

    return {
        'trades': n_trades, 'wins': wins, 'losses': losses,
        'wr': round(wr, 1), 'rr': round(rr, 2),
        'total_pnl': round(total_pnl, 2), 'avg_pnl': round(avg_pnl, 4),
        't_stat': round(t_stat, 2), 'pf': round(pf, 2),
        'max_dd': round(max_dd, 2),
        'n_long': n_long, 'n_short': n_short,
        'monthly_wr': monthly_wr, 'weekly_wr': weekly_wr, 'daily_wr': daily_wr,
        'wr_mean_monthly': m_stab['mean'], 'wr_std_monthly': m_stab['std'],
        'wr_min_monthly': m_stab['min'], 'wr_max_monthly': m_stab['max'],
        'profitable_months': m_stab['profitable'], 'total_months': m_stab['total'],
        'wr_mean_weekly': w_stab['mean'], 'wr_std_weekly': w_stab['std'],
        'wr_min_weekly': w_stab['min'], 'wr_max_weekly': w_stab['max'],
        'profitable_weeks': w_stab['profitable'], 'total_weeks': w_stab['total'],
        'wr_mean_daily': d_stab['mean'], 'wr_std_daily': d_stab['std'],
        'wr_min_daily': d_stab['min'], 'wr_max_daily': d_stab['max'],
        'profitable_days': d_stab['profitable'], 'total_days': d_stab['total'],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    print('=' * 80)
    print('  CONFLUENCE FILTER TESTING')
    print('  Testing rule-based filters on top of ML directional signals')
    print('=' * 80)

    # ── Load data ──
    print(f'\n  Loading {DATA_PATH} ...')
    df = pd.read_csv(DATA_PATH, sep='\t', header=None,
                     names=['Date','Time','Open','High','Low','Close',
                            'TickVol','Vol','Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df.rename(columns={'TickVol': 'Volume'}, inplace=True)
    print(f'  {len(df):,} bars loaded')

    print('  Engineering 1min features...')
    df_feat = create_microstructure_features(df)
    print(f'  {len(df_feat):,} bars after warmup')

    print('  Resampling HTF bars (15min/1H/4H/Daily) and computing features...')
    df_feat = create_htf_features(df_feat)
    df_feat = df_feat.dropna(subset=HTF_FEATURES)
    print(f'  {len(df_feat):,} bars after HTF warmup')
    print(f'  Total features: {len(FEATURE_NAMES)} ({len(BASE_FEATURES)} base + {len(HTF_FEATURES)} HTF)')

    split_idx = int(len(df_feat) * 0.6)
    print(f'  Train: {split_idx:,}  |  Test: {len(df_feat) - split_idx:,}')

    # ── Numba compile ──
    print('  Compiling Numba...')
    _ = label_directional(np.zeros(100), np.zeros(100), np.zeros(100), 10, 1.0, 0.5)

    # ── Label ──
    print('  Labeling...')
    closes = df_feat['Close'].values.astype(np.float64)
    highs  = df_feat['High'].values.astype(np.float64)
    lows   = df_feat['Low'].values.astype(np.float64)
    labels = label_directional(closes, highs, lows, HORIZON, TARGET, STOP)
    df_feat['label'] = labels

    # ── Train RF ──
    train = df_feat.iloc[:split_idx]
    test  = df_feat.iloc[split_idx:]

    print(f'  Training RF (depth={RF_DEPTH}, trees={RF_TREES})...')
    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_DEPTH,
        random_state=42, n_jobs=-1,
    )
    rf.fit(train[FEATURE_NAMES].fillna(0), train['label'].values)
    classes = rf.classes_

    # ── Predict on test ──
    X_test = test[FEATURE_NAMES].fillna(0)
    proba = rf.predict_proba(X_test)
    long_ci = np.where(classes == 1)[0][0]
    short_ci = np.where(classes == 2)[0][0]
    probs_long = proba[:, long_ci]
    probs_short = proba[:, short_ci]
    dir_labels = test['label'].values
    closes_test = test['Close'].values

    # ── Baseline (no filter) ──
    print('\n  Computing baseline (no filter)...')
    baseline = evaluate_filter(probs_long, probs_short, dir_labels, closes_test,
                               test, RF_THRESHOLD, None)

    print(f'\n{"="*80}')
    print(f'  BASELINE (ML only, threshold={RF_THRESHOLD})')
    print(f'{"="*80}')
    print(f'  Trades: {baseline["trades"]:,}  |  WR: {baseline["wr"]}%  |  '
          f'RR: {baseline["rr"]}  |  P&L: ${baseline["total_pnl"]:+,.2f}  |  '
          f't: {baseline["t_stat"]}  |  PF: {baseline["pf"]}')
    print(f'  Longs: {baseline["n_long"]}  |  Shorts: {baseline["n_short"]}')

    # ── Add baseline as first result ──
    filters = define_confluence_filters()

    def make_result_row(name, desc, m, wr_delta=0.0, pnl_delta=0.0):
        return {
            'name': name, 'desc': desc,
            'trades': m['trades'], 'wr': m['wr'], 'wr_delta': round(wr_delta, 1),
            'pnl': m['total_pnl'], 'pnl_delta': round(pnl_delta, 2),
            't_stat': m['t_stat'], 'pf': m['pf'], 'rr': m['rr'],
            'max_dd': m['max_dd'],
            'n_long': m['n_long'], 'n_short': m['n_short'],
            'wins': m['wins'], 'losses': m['losses'],
            'wr_mean_m': m['wr_mean_monthly'], 'wr_std_m': m['wr_std_monthly'],
            'wr_min_m': m['wr_min_monthly'], 'wr_max_m': m['wr_max_monthly'],
            'prof_months': f'{m["profitable_months"]}/{m["total_months"]}',
            'wr_mean_w': m['wr_mean_weekly'], 'wr_std_w': m['wr_std_weekly'],
            'wr_min_w': m['wr_min_weekly'], 'wr_max_w': m['wr_max_weekly'],
            'prof_weeks': f'{m["profitable_weeks"]}/{m["total_weeks"]}',
            'wr_mean_d': m['wr_mean_daily'], 'wr_std_d': m['wr_std_daily'],
            'wr_min_d': m['wr_min_daily'], 'wr_max_d': m['wr_max_daily'],
            'prof_days': f'{m["profitable_days"]}/{m["total_days"]}',
            'monthly_wr': m['monthly_wr'],
            'weekly_wr': m['weekly_wr'],
            'daily_wr': m['daily_wr'],
        }

    results = [make_result_row('BASELINE (no filter)', f'ML only, threshold={RF_THRESHOLD}', baseline)]

    print(f'\n{"="*80}')
    print(f'  TESTING {len(filters)} CONFLUENCE FILTERS')
    print(f'{"="*80}\n')

    for name, filt in filters.items():
        # Pre-compute masks for speed
        if 'long' in filt:
            filt['_long_mask'] = filt['long'](test).values
        if 'short' in filt:
            filt['_short_mask'] = filt['short'](test).values

        m = evaluate_filter(probs_long, probs_short, dir_labels, closes_test,
                            test, RF_THRESHOLD, filt)

        if m is None:
            results.append({'name': name, 'desc': filt['desc'], 'trades': 0,
                            'wr': 0, 'wr_delta': 0, 'pnl': 0, 'pnl_delta': 0,
                            't_stat': 0, 'pf': 0, 'rr': 0, 'max_dd': 0,
                            'n_long': 0, 'n_short': 0, 'wins': 0, 'losses': 0,
                            'wr_mean_m': 0, 'wr_std_m': 0, 'wr_min_m': 0,
                            'wr_max_m': 0, 'prof_months': '0/0',
                            'wr_mean_w': 0, 'wr_std_w': 0, 'wr_min_w': 0,
                            'wr_max_w': 0, 'prof_weeks': '0/0',
                            'wr_mean_d': 0, 'wr_std_d': 0, 'wr_min_d': 0,
                            'wr_max_d': 0, 'prof_days': '0/0',
                            'monthly_wr': {}, 'weekly_wr': {}, 'daily_wr': {}})
            continue

        wr_delta = m['wr'] - baseline['wr']
        pnl_delta = m['total_pnl'] - baseline['total_pnl']
        results.append(make_result_row(name, filt['desc'], m, wr_delta, pnl_delta))

    # ── Sort by WR improvement ──
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('wr', ascending=False).reset_index(drop=True)

    # ── Print results ──
    print(f'  {"Filter":<28s}  {"Trd":>5s}  {"WR%":>5s}  {"ΔWR":>5s}  '
          f'{"P&L":>9s}  {"t":>5s}  '
          f'{"MoWR":>5s}  {"±":>4s}  {"PrfM":>4s}  '
          f'{"WkWR":>5s}  {"±":>4s}  {"PrfW":>5s}  '
          f'{"DyWR":>5s}  {"±":>4s}  {"PrfD":>5s}')
    print(f'  {"─"*115}')

    for _, r in results_df.iterrows():
        wr_flag = '++' if r['wr_delta'] > 2 else ('+' if r['wr_delta'] > 0 else '')
        print(f'  {r["name"]:<28s}  {r["trades"]:>5d}  {r["wr"]:>5.1f}  {r["wr_delta"]:>+5.1f}  '
              f'${r["pnl"]:>+8.2f}  {r["t_stat"]:>5.2f}  '
              f'{r["wr_mean_m"]:>5.1f}  {r["wr_std_m"]:>4.1f}  {r["prof_months"]:>4s}  '
              f'{r["wr_mean_w"]:>5.1f}  {r["wr_std_w"]:>4.1f}  {r["prof_weeks"]:>5s}  '
              f'{r["wr_mean_d"]:>5.1f}  {r["wr_std_d"]:>4.1f}  {r["prof_days"]:>5s}  {wr_flag}')

    # ── Summary of best filters ──
    improved = results_df[results_df['wr_delta'] > 0].copy()
    if len(improved) > 0:
        print(f'\n{"="*80}')
        print(f'  FILTERS THAT IMPROVED WIN RATE (sorted by WR)')
        print(f'{"="*80}')
        for _, r in improved.iterrows():
            print(f'  {r["name"]:<30s}  WR: {r["wr"]:5.1f}% ({r["wr_delta"]:+.1f})  '
                  f'P&L: ${r["pnl"]:+,.2f} ({r["pnl_delta"]:+,.2f})  '
                  f'Trades: {r["trades"]:,}  t: {r["t_stat"]}  '
                  f'[{r["desc"]}]')
    else:
        print('\n  No filters improved win rate.')

    # ── Best by P&L ──
    pnl_improved = results_df[results_df['pnl_delta'] > 0].sort_values('pnl', ascending=False)
    if len(pnl_improved) > 0:
        print(f'\n{"="*80}')
        print(f'  FILTERS THAT IMPROVED TOTAL P&L (sorted by P&L)')
        print(f'{"="*80}')
        for _, r in pnl_improved.iterrows():
            print(f'  {r["name"]:<30s}  P&L: ${r["pnl"]:+,.2f} ({r["pnl_delta"]:+,.2f})  '
                  f'WR: {r["wr"]:5.1f}%  Trades: {r["trades"]:,}  '
                  f'[{r["desc"]}]')

    # ══════════════════════════════════════════════════════════════════════
    #  WR STABILITY — ALL PERIODS, ALL FILTERS
    # ══════════════════════════════════════════════════════════════════════

    flat_rows = []  # One row per filter × period × date

    for period_label, wr_key in [('MONTHLY', 'monthly_wr'),
                                  ('WEEKLY', 'weekly_wr'),
                                  ('DAILY', 'daily_wr')]:
        print(f'\n{"="*80}')
        print(f'  {period_label} WIN RATE STABILITY — ALL FILTERS')
        print(f'{"="*80}')

        # Collect all period keys
        all_periods = set()
        for _, r in results_df.iterrows():
            bk = r.get(wr_key, {})
            if bk:
                all_periods.update(bk.keys())
        all_periods = sorted(all_periods)

        for _, r in results_df.iterrows():
            if r['trades'] == 0:
                continue
            bk = r.get(wr_key, {})
            if not bk:
                continue

            stab_key_m = 'wr_mean_m' if period_label == 'MONTHLY' else (
                         'wr_mean_w' if period_label == 'WEEKLY' else 'wr_mean_d')
            stab_key_s = 'wr_std_m' if period_label == 'MONTHLY' else (
                         'wr_std_w' if period_label == 'WEEKLY' else 'wr_std_d')
            stab_key_min = 'wr_min_m' if period_label == 'MONTHLY' else (
                           'wr_min_w' if period_label == 'WEEKLY' else 'wr_min_d')
            stab_key_max = 'wr_max_m' if period_label == 'MONTHLY' else (
                           'wr_max_w' if period_label == 'WEEKLY' else 'wr_max_d')
            stab_key_prf = 'prof_months' if period_label == 'MONTHLY' else (
                           'prof_weeks' if period_label == 'WEEKLY' else 'prof_days')

            print(f'\n  ── {r["name"]} ({r["desc"]}) ──')
            print(f'  Net WR: {r["wr"]}%  |  {period_label} avg: {r[stab_key_m]}% ± {r[stab_key_s]}%  '
                  f'|  Range: {r[stab_key_min]}% – {r[stab_key_max]}%  '
                  f'|  Profitable: {r[stab_key_prf]}')

            # Print table (limit daily to summary only — too many rows)
            if period_label != 'DAILY':
                print(f'  {"Period":<12s}  {"Trades":>6s}  {"WR%":>6s}  {"P&L":>9s}  {"Bar":>1s}')
                print(f'  {"─"*48}')
                for p in all_periods:
                    if p in bk:
                        mv = bk[p]
                        bar_len = int(mv['wr'] / 2)
                        bar = '█' * bar_len
                        pnl_sign = '+' if mv['pnl'] >= 0 else ''
                        print(f'  {p:<12s}  {mv["trades"]:>6d}  {mv["wr"]:>5.1f}%  '
                              f'${pnl_sign}{mv["pnl"]:>8.2f}  {bar}')

            # Collect flat rows for CSV
            for p in all_periods:
                if p in bk:
                    mv = bk[p]
                    flat_rows.append({
                        'filter': r['name'], 'period': period_label,
                        'date': p, 'trades': mv['trades'],
                        'wr': mv['wr'], 'pnl': mv['pnl'],
                        'net_wr': r['wr'], 'wr_delta': r['wr_delta'],
                    })
                else:
                    flat_rows.append({
                        'filter': r['name'], 'period': period_label,
                        'date': p, 'trades': 0, 'wr': None, 'pnl': 0,
                        'net_wr': r['wr'], 'wr_delta': r['wr_delta'],
                    })

    # ── Save files ──
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Summary CSV (one row per filter)
    save_df = results_df.drop(columns=['monthly_wr', 'weekly_wr', 'daily_wr'], errors='ignore')
    csv_path = os.path.join(out_dir, 'confluence_results_htf.csv')
    save_df.to_csv(csv_path, index=False)
    print(f'\n  Summary saved → {csv_path}')

    # 2. Full period stability CSV (filter × period × date)
    flat_df = pd.DataFrame(flat_rows)
    flat_csv = os.path.join(out_dir, 'confluence_wr_stability_htf.csv')
    flat_df.to_csv(flat_csv, index=False)
    print(f'  Stability (monthly+weekly+daily) saved → {flat_csv}')

    elapsed = time.time() - t0
    print(f'\n  Done in {elapsed:.1f}s')
    print('=' * 80)
