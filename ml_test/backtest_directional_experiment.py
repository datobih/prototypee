"""
Directional ML Strategy — Production Backtest
===============================================
Best params from Optuna optimization (200 trials):
  MODE: directional | TGT=$2.0 | SL=$0.50 | HOR=15
  RF: threshold=0.40, depth=20, trees=200

Walk-forward: 60% train, 40% test.
Spread: $0.30 per position (1x for directional).
"""

import numpy as np
import pandas as pd
import numba as nb
import pickle
import time
import warnings
import os

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ── Best params from Optuna ──────────────────────────────────────────────────
TARGET       = 2.0
STOP         = 0.50
HORIZON      = 15
RF_THRESHOLD = 0.40
RF_DEPTH     = 20
RF_TREES     = 200
SPREAD       = 0.30

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'XAUUSD1.csv')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'output')


# ── Feature engineering ──────────────────────────────────────────────────────
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

    return df.dropna()


# ── RSI helper ─────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
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
        htf = df.resample(resample_rule).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'TickVol': 'sum',
        }).dropna()

        # ── Trend: EMA crossovers ──
        htf[f'{tf_name}_ema8']  = htf['Close'].ewm(span=8).mean()
        htf[f'{tf_name}_ema21'] = htf['Close'].ewm(span=21).mean()
        htf[f'{tf_name}_ema50'] = htf['Close'].ewm(span=50).mean()

        # Trend direction: +1 bullish, -1 bearish, 0 neutral
        htf[f'{tf_name}_trend'] = (
            ((htf['Close'] > htf[f'{tf_name}_ema8']) &
             (htf[f'{tf_name}_ema8'] > htf[f'{tf_name}_ema21'])).astype(int) -
            ((htf['Close'] < htf[f'{tf_name}_ema8']) &
             (htf[f'{tf_name}_ema8'] < htf[f'{tf_name}_ema21'])).astype(int)
        )

        # EMA50 trend (longer-term bias)
        htf[f'{tf_name}_trend50'] = (
            (htf['Close'] > htf[f'{tf_name}_ema50']).astype(int) -
            (htf['Close'] < htf[f'{tf_name}_ema50']).astype(int)
        )

        # Distance from EMA21 (normalized)
        htf[f'{tf_name}_dist_ema21'] = (htf['Close'] - htf[f'{tf_name}_ema21']) / htf['Close']

        # ── Momentum: RSI ──
        htf[f'{tf_name}_rsi'] = compute_rsi(htf['Close'], 14)

        # ── Momentum: directional flow ──
        htf_body = htf['Close'] - htf['Open']
        htf_flow = htf_body / htf['Close']
        htf[f'{tf_name}_flow_3'] = htf_flow.rolling(3).sum()
        htf[f'{tf_name}_flow_5'] = htf_flow.rolling(5).sum()

        # ── Structure: close position in range ──
        htf_range = htf['High'] - htf['Low']
        htf[f'{tf_name}_close_pos'] = (htf['Close'] - htf['Low']) / (htf_range + 1e-10)

        # ── Volatility: ATR ratio ──
        htf_atr3  = htf_range.rolling(3).mean()
        htf_atr10 = htf_range.rolling(10).mean()
        htf[f'{tf_name}_vol_ratio'] = htf_atr3 / (htf_atr10 + 1e-10)
        htf[f'{tf_name}_vol_exp'] = (htf_range > htf_atr10 * 1.2).astype(int)

        # ── Structure: daily range position (high-low context) ──
        htf[f'{tf_name}_range_pct'] = htf_range / htf['Close']

        # Select only the computed features (drop raw OHLC)
        feat_cols = [c for c in htf.columns if c.startswith(f'{tf_name}_')]
        htf_feat = htf[feat_cols]

        # CRITICAL: shift(1) so we only use COMPLETED HTF candles
        htf_feat = htf_feat.shift(1)

        htf_dfs[tf_name] = htf_feat

    # Merge all HTF features back to 1min index via forward-fill
    result = df.copy()
    for tf_name, htf_feat in htf_dfs.items():
        # Reindex to 1min and forward-fill
        htf_reindexed = htf_feat.reindex(result.index, method='ffill')
        result = pd.concat([result, htf_reindexed], axis=1)

    return result


# ── Feature names ──────────────────────────────────────────────────────────
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


# ── Numba directional labeling ──────────────────────────────────────────────
@nb.njit(cache=True)
def label_directional(closes, highs, lows, horizon, target, stop):
    """
    For each bar, does price hit long TP or short TP first?
      1 = LONG TP hit first
      2 = SHORT TP hit first
      0 = neither (range-bound / timeout)
    """
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


# ── Trade execution with full logging ────────────────────────────────────────
def run_backtest(df_test, rf, classes, threshold, target, stop, feature_names=None):
    """
    Run directional backtest on test set. Returns DataFrame of trades.
    """
    feat = feature_names if feature_names is not None else FEATURE_NAMES
    X_test = df_test[feat].fillna(0)
    proba = rf.predict_proba(X_test)

    long_ci  = np.where(classes == 1)[0]
    short_ci = np.where(classes == 2)[0]
    if len(long_ci) == 0 or len(short_ci) == 0:
        return pd.DataFrame()

    probs_long  = proba[:, long_ci[0]]
    probs_short = proba[:, short_ci[0]]

    dir_labels = df_test['label'].values
    datetimes  = df_test.index if isinstance(df_test.index, pd.DatetimeIndex) else pd.to_datetime(df_test['Datetime'])
    closes     = df_test['Close'].values

    trades = []

    # Long entries
    long_mask = probs_long >= threshold
    for idx in np.where(long_mask)[0]:
        entry_price = closes[idx]
        dt = datetimes[idx]
        prob = probs_long[idx]
        lbl = dir_labels[idx]

        if lbl == 1:
            pnl = target - SPREAD
            result = 'WIN'
        elif lbl == 2:
            pnl = -stop - SPREAD
            result = 'LOSS'
        else:
            pnl = -SPREAD
            result = 'TIMEOUT'

        trades.append({
            'datetime': dt, 'direction': 'LONG', 'entry': round(entry_price, 2),
            'tp': round(entry_price + target, 2), 'sl': round(entry_price - stop, 2),
            'prob': round(prob, 4), 'result': result, 'pnl': round(pnl, 2),
        })

    # Short entries
    short_mask = probs_short >= threshold
    for idx in np.where(short_mask)[0]:
        entry_price = closes[idx]
        dt = datetimes[idx]
        prob = probs_short[idx]
        lbl = dir_labels[idx]

        if lbl == 2:
            pnl = target - SPREAD
            result = 'WIN'
        elif lbl == 1:
            pnl = -stop - SPREAD
            result = 'LOSS'
        else:
            pnl = -SPREAD
            result = 'TIMEOUT'

        trades.append({
            'datetime': dt, 'direction': 'SHORT', 'entry': round(entry_price, 2),
            'tp': round(entry_price - target, 2), 'sl': round(entry_price + stop, 2),
            'prob': round(prob, 4), 'result': result, 'pnl': round(pnl, 2),
        })

    df_trades = pd.DataFrame(trades)
    if len(df_trades) > 0:
        df_trades = df_trades.sort_values('datetime').reset_index(drop=True)
    return df_trades


# ── Reporting ────────────────────────────────────────────────────────────────
def print_header(title):
    print(f'\n{"="*72}')
    print(f'  {title}')
    print(f'{"="*72}')


def report_summary(df_trades):
    """Print comprehensive performance report."""
    if len(df_trades) == 0:
        print('  No trades.')
        return

    pnl = df_trades['pnl'].values
    n = len(pnl)
    wins = (df_trades['result'] == 'WIN').sum()
    losses = (df_trades['result'] == 'LOSS').sum()
    timeouts = (df_trades['result'] == 'TIMEOUT').sum()
    n_long = (df_trades['direction'] == 'LONG').sum()
    n_short = (df_trades['direction'] == 'SHORT').sum()

    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    std_pnl = pnl.std() if n > 1 else 1e-10
    t_stat = avg_pnl / (std_pnl / np.sqrt(n) + 1e-10)
    sharpe = avg_pnl / (std_pnl + 1e-10) * np.sqrt(252)

    cum = np.cumsum(pnl)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))
    peak_equity = float(np.max(cum))

    win_pnl = pnl[pnl > 0]
    loss_pnl = pnl[pnl < 0]
    avg_win = win_pnl.mean() if len(win_pnl) > 0 else 0.0
    avg_loss = loss_pnl.mean() if len(loss_pnl) > 0 else -1.0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    gross_w = win_pnl.sum() if len(win_pnl) > 0 else 0.0
    gross_l = abs(loss_pnl.sum()) if len(loss_pnl) > 0 else 1e-10
    pf = gross_w / gross_l

    # Win streaks / loss streaks
    results = (pnl > 0).astype(int)
    max_win_streak = max_streak(results, 1)
    max_loss_streak = max_streak(results, 0)

    print(f'  Total Trades:    {n:,}  (L:{n_long} S:{n_short})')
    print(f'  Wins:            {wins:,}  |  Losses: {losses:,}  |  Timeouts: {timeouts:,}')
    print(f'  Win Rate:        {wins/n*100:.1f}%')
    print(f'  Avg Win:       $ {avg_win:+.2f}  |  Avg Loss: $ {avg_loss:+.2f}  |  RR: {rr:.2f}')
    print(f'  Total P&L:     $ {total_pnl:+,.2f}')
    print(f'  Avg P&L:       $ {avg_pnl:+.4f} per trade')
    print(f'  t-statistic:     {t_stat:.2f}')
    print(f'  Profit Factor:   {pf:.2f}')
    print(f'  Sharpe (√252):   {sharpe:.2f}')
    print(f'  Peak Equity:   $ {peak_equity:+,.2f}')
    print(f'  Max Drawdown:  $ {max_dd:,.2f}')
    print(f'  Win Streak:      {max_win_streak}  |  Loss Streak: {max_loss_streak}')
    print(f'  Spread cost:   $ {SPREAD:.2f}/trade  (total: ${SPREAD*n:,.2f})')


def max_streak(arr, val):
    streak = 0
    best = 0
    for x in arr:
        if x == val:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def report_by_direction(df_trades):
    """Breakdown by LONG vs SHORT."""
    print_header('BREAKDOWN BY DIRECTION')
    for d in ['LONG', 'SHORT']:
        sub = df_trades[df_trades['direction'] == d]
        if len(sub) == 0:
            continue
        pnl = sub['pnl'].values
        wins = (sub['result'] == 'WIN').sum()
        n = len(sub)
        wr = wins / n * 100 if n > 0 else 0
        print(f'  {d:5s}:  {n:5d} trades  |  WR: {wr:5.1f}%  |  P&L: ${pnl.sum():+,.2f}  |  Avg: ${pnl.mean():+.4f}')


def report_monthly(df_trades):
    """Monthly P&L breakdown."""
    print_header('MONTHLY P&L')
    df_trades['month'] = df_trades['datetime'].dt.to_period('M')
    monthly = df_trades.groupby('month').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wins=('result', lambda x: (x == 'WIN').sum()),
    )
    monthly['wr'] = (monthly['wins'] / monthly['trades'] * 100).round(1)

    print(f'  {"Month":<10s}  {"Trades":>6s}  {"WR%":>6s}  {"P&L":>10s}  {"Cumulative":>10s}')
    print(f'  {"─"*50}')
    cum = 0.0
    for period, row in monthly.iterrows():
        cum += row['pnl']
        print(f'  {str(period):<10s}  {int(row["trades"]):>6d}  {row["wr"]:>5.1f}%  ${row["pnl"]:>+9.2f}  ${cum:>+9.2f}')

    # Yearly summary
    df_trades['year'] = df_trades['datetime'].dt.year
    yearly = df_trades.groupby('year').agg(trades=('pnl', 'count'), pnl=('pnl', 'sum'))
    print(f'\n  {"Year":<10s}  {"Trades":>6s}  {"P&L":>10s}')
    print(f'  {"─"*30}')
    for yr, row in yearly.iterrows():
        print(f'  {yr:<10d}  {int(row["trades"]):>6d}  ${row["pnl"]:>+9.2f}')


def report_by_hour(df_trades):
    """Hourly P&L breakdown."""
    print_header('P&L BY HOUR (UTC)')
    df_trades['hour'] = df_trades['datetime'].dt.hour
    hourly = df_trades.groupby('hour').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wins=('result', lambda x: (x == 'WIN').sum()),
    )
    hourly['wr'] = (hourly['wins'] / hourly['trades'] * 100).round(1)

    print(f'  {"Hour":>4s}  {"Trades":>6s}  {"WR%":>6s}  {"P&L":>10s}')
    print(f'  {"─"*35}')
    for hr, row in hourly.iterrows():
        print(f'  {hr:>4d}  {int(row["trades"]):>6d}  {row["wr"]:>5.1f}%  ${row["pnl"]:>+9.2f}')


def report_equity_curve(df_trades):
    """Print text-based equity curve."""
    print_header('EQUITY CURVE (top 50 points)')
    cum = np.cumsum(df_trades['pnl'].values)
    n = len(cum)
    step = max(1, n // 50)
    sampled = cum[::step]
    mn, mx = sampled.min(), sampled.max()
    width = 50

    for i, val in enumerate(sampled):
        bar_pos = int((val - mn) / (mx - mn + 1e-10) * width)
        bar = '█' * max(1, bar_pos)
        trade_num = i * step
        print(f'  T{trade_num:>5d} | ${val:>+9.2f} |{bar}')


def report_feature_importance(rf, top_n=15):
    """Print top feature importances."""
    print_header(f'TOP {top_n} FEATURE IMPORTANCES')
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    for rank, idx in enumerate(indices):
        bar = '█' * int(importances[idx] * 200)
        print(f'  {rank+1:>2d}. {FEATURE_NAMES[idx]:<30s}  {importances[idx]:.4f}  {bar}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    print_header('DIRECTIONAL ML STRATEGY — PRODUCTION BACKTEST')
    print(f'  Params: TGT=${TARGET}  SL=${STOP}  HOR={HORIZON}  TH={RF_THRESHOLD}')
    print(f'  RF: depth={RF_DEPTH}  trees={RF_TREES}')
    print(f'  Spread: ${SPREAD}/trade')

    # ── Load data ────────────────────────────────────────────────────────
    print(f'\n  Loading {DATA_PATH} ...')
    df = pd.read_csv(DATA_PATH, sep='\t',
                     header=None,
                     names=['Date','Time','Open','High','Low','Close',
                            'TickVol','Vol','Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    print(f'  {len(df):,} bars loaded')

    # ── Features ─────────────────────────────────────────────────────────
    print('  Engineering 1min microstructure features...')
    df_feat = create_microstructure_features(df)
    print(f'  {len(df_feat):,} bars after warmup')

    # ── HTF features ───────────────────────────────────────────────────
    print('  Resampling HTF bars (15min/1H/4H/Daily) and computing features...')
    df_feat = create_htf_features(df_feat)
    htf_nans = df_feat[HTF_FEATURES].isna().any(axis=1).sum()
    df_feat = df_feat.dropna(subset=HTF_FEATURES)
    print(f'  {len(df_feat):,} bars after HTF warmup ({htf_nans:,} rows dropped)')
    print(f'  Total features: {len(FEATURE_NAMES)} ({len(BASE_FEATURES)} base + {len(HTF_FEATURES)} HTF)')

    # ── Walk-forward split ───────────────────────────────────────────────
    split_idx = int(len(df_feat) * 0.6)
    print(f'  Train: {split_idx:,}  |  Test: {len(df_feat) - split_idx:,}')

    # ── Numba compile ────────────────────────────────────────────────────
    print('  Compiling Numba labeling...')
    _ = label_directional(np.zeros(100), np.zeros(100), np.zeros(100), 10, 1.0, 0.5)
    print('  Done.')

    # ── Label ────────────────────────────────────────────────────────────
    print('  Labeling directional outcomes...')
    closes = df_feat['Close'].values.astype(np.float64)
    highs  = df_feat['High'].values.astype(np.float64)
    lows   = df_feat['Low'].values.astype(np.float64)

    labels = label_directional(closes, highs, lows, HORIZON, TARGET, STOP)
    df_feat['label'] = labels

    n_long_lbl = (labels == 1).sum()
    n_short_lbl = (labels == 2).sum()
    n_neutral = (labels == 0).sum()
    print(f'  Labels: {n_long_lbl:,} long wins, {n_short_lbl:,} short wins, {n_neutral:,} neutral')

    # ── Train / Test ─────────────────────────────────────────────────────
    train = df_feat.iloc[:split_idx]
    test  = df_feat.iloc[split_idx:]
    y_train = train['label'].values

    # ══════════════════════════════════════════════════════════════════════
    #  A/B COMPARISON: BASE-ONLY vs BASE+HTF
    # ══════════════════════════════════════════════════════════════════════

    results_comparison = {}

    for run_name, feat_list in [('BASE_ONLY', BASE_FEATURES), ('BASE+HTF', FEATURE_NAMES)]:
        print_header(f'RUN: {run_name} ({len(feat_list)} features)')

        X_train_run = train[feat_list].fillna(0)
        print(f'  Training RF (depth={RF_DEPTH}, trees={RF_TREES})...')
        rf = RandomForestClassifier(
            n_estimators=RF_TREES, max_depth=RF_DEPTH,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_run, y_train)
        classes = rf.classes_
        print(f'  RF classes: {classes}')

        print('  Running backtest on test set...')
        df_trades = run_backtest(test, rf, classes, RF_THRESHOLD, TARGET, STOP,
                                feature_names=feat_list)

        if len(df_trades) == 0:
            print('  ERROR: No trades generated.')
            continue

        report_summary(df_trades)
        report_by_direction(df_trades)
        report_monthly(df_trades)

        pnl = df_trades['pnl'].values
        wins = (df_trades['result'] == 'WIN').sum()
        results_comparison[run_name] = {
            'trades': len(df_trades),
            'wr': wins / len(df_trades) * 100,
            'pnl': pnl.sum(),
            'avg_pnl': pnl.mean(),
            't_stat': pnl.mean() / (pnl.std() / np.sqrt(len(pnl)) + 1e-10),
            'pf': pnl[pnl > 0].sum() / (abs(pnl[pnl < 0].sum()) + 1e-10),
        }

        if run_name == 'BASE+HTF':
            rf_htf = rf  # save reference for model export
            report_by_hour(df_trades)
            report_equity_curve(df_trades)
            report_feature_importance(rf, top_n=20)

    # ══════════════════════════════════════════════════════════════════════
    #  SIDE-BY-SIDE COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print_header('A/B COMPARISON: BASE vs BASE+HTF')
    print(f'  {"Metric":<15s}  {"BASE_ONLY":>12s}  {"BASE+HTF":>12s}  {"Delta":>10s}')
    print(f'  {"─"*55}')
    for metric in ['trades', 'wr', 'pnl', 'avg_pnl', 't_stat', 'pf']:
        base_val = results_comparison.get('BASE_ONLY', {}).get(metric, 0)
        htf_val  = results_comparison.get('BASE+HTF', {}).get(metric, 0)
        delta = htf_val - base_val
        if metric == 'wr':
            print(f'  {"Win Rate %":<15s}  {base_val:>11.1f}%  {htf_val:>11.1f}%  {delta:>+9.1f}%')
        elif metric in ('pnl', 'avg_pnl'):
            label = 'Total P&L $' if metric == 'pnl' else 'Avg P&L $'
            print(f'  {label:<15s}  ${base_val:>+10.2f}  ${htf_val:>+10.2f}  ${delta:>+9.2f}')
        elif metric == 'trades':
            print(f'  {"Trades":<15s}  {int(base_val):>12,}  {int(htf_val):>12,}  {int(delta):>+10,}')
        else:
            label = 't-statistic' if metric == 't_stat' else 'Profit Factor'
            print(f'  {label:<15s}  {base_val:>12.2f}  {htf_val:>12.2f}  {delta:>+10.2f}')

    # ── Save outputs (HTF model) ─────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    model_path = os.path.join(OUT_DIR, 'directional_rf_model_htf.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': rf_htf, 'features': FEATURE_NAMES,
                     'base_features': BASE_FEATURES,
                     'htf_features': HTF_FEATURES,
                     'params': {'target': TARGET, 'stop': STOP,
                                'horizon': HORIZON, 'threshold': RF_THRESHOLD,
                                'depth': RF_DEPTH, 'trees': RF_TREES,
                                'spread': SPREAD}}, f)
    print(f'\n  HTF Model saved: {model_path}')

    trades_path = os.path.join(OUT_DIR, 'directional_trades_htf_experiment.csv')
    df_trades.to_csv(trades_path, index=False)
    print(f'  Trades saved: {trades_path}')

    elapsed = time.time() - t0
    print_header('DONE')
    print(f'  Elapsed: {elapsed:.1f}s')
    print(f'  Params: TGT=${TARGET}  SL=${STOP}  HOR={HORIZON}  TH={RF_THRESHOLD}')
    print(f'  Features: {len(BASE_FEATURES)} base + {len(HTF_FEATURES)} HTF = {len(FEATURE_NAMES)} total')
    print(f'{"="*72}\n')
