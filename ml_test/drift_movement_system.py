"""
drift_movement_system.py

Integrated Movement Predictor + Drift Detection System for XAUUSD 1-min.

Pipeline:
  1. Train RF movement model on a reference window
  2. Store reference feature distributions for drift monitoring
  3. Walk forward through time:
     a. Compute drift score (PSI/KS) on current window vs reference
     b. If GREEN/YELLOW: run RF prediction → if high probability → execute OCO
     c. If RED: retrain model on recent data, reset reference window
  4. Track performance with and without drift gating

Backtest mode: simulates the full pipeline on historical data to validate.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from numpy.lib.stride_tricks import sliding_window_view
from hedge_strategy_strict import create_microstructure_features

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
TRAIN_WINDOW    = 4000    # bars to train RF on (optimal from drift study)
DRIFT_CHECK     = 500     # check drift every N bars
LABEL_HORIZON   = 5       # bars to look ahead for movement label (5 min)
LABEL_THRESHOLD = 5.0     # $5 movement threshold → ~37.7% positive rate
RF_THRESHOLD    = 0.55    # min RF probability to trigger a trade signal
OCO_TP          = 5.0     # OCO take profit ($)
OCO_SL          = 2.5     # OCO stop loss ($)
OCO_EXPIRY      = 10      # bars before OCO expires if neither TP/SL hit

# Drift thresholds (from concept_drift_window.py results)
PSI_YELLOW      = 0.10    # moderate drift
PSI_RED         = 0.25    # significant drift — retrain

# Model features (matches concept_drift_window.py)
MODEL_FEATURES = [
    'boll_width_20', 'atr_percentile', 'squeeze_intensity',
    'vol_of_vol_20', 'gk_vol_10', 'vol_zscore',
    'abs_flow_10', 'flow_zscore', 'flow_net_ratio_10',
    'abs_dist_ema8', 'abs_dist_ema21', 'abs_ema_spread',
    'abs_price_zscore_20', 'boll_width_roc',
    'hour_sin', 'hour_cos', 'is_overlap',
    'combo_bw_flow10', 'combo_bw_squeeze',
    'parkinson_vol_10', 'keltner_width_20',
    'flow_per_atr_10', 'dist_ema8_per_atr',
]

# Drift monitoring features (top drifters from study)
DRIFT_FEATURES = [
    'gk_vol_10', 'boll_width_20', 'vol_of_vol_20',
    'abs_flow_10', 'abs_dist_ema8', 'abs_dist_ema21',
    'abs_ema_spread', 'squeeze_intensity', 'vol_zscore',
    'atr_percentile', 'flow_zscore', 'flow_net_ratio_10',
    'abs_price_zscore_20', 'boll_width_roc', 'atr_percentile_200',
    'flow_concentration_5', 'keltner_width_20', 'parkinson_vol_10',
]


# =============================================================================
# DRIFT METRICS
# =============================================================================
def compute_psi(reference, test, bins=10):
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    ref_counts = np.histogram(reference, bins=breakpoints)[0] + 1
    test_counts = np.histogram(test, bins=breakpoints)[0] + 1
    ref_pct = ref_counts / ref_counts.sum()
    test_pct = test_counts / test_counts.sum()
    return np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))


def compute_drift_score(ref_df, test_df, features):
    """Compute aggregate PSI across drift features."""
    psi_scores = []
    for feat in features:
        ref_vals = ref_df[feat].dropna().values
        test_vals = test_df[feat].dropna().values
        if len(ref_vals) < 50 or len(test_vals) < 50:
            continue
        psi_scores.append(compute_psi(ref_vals, test_vals))
    return np.mean(psi_scores) if psi_scores else 0.0


def classify_drift(psi):
    if psi >= PSI_RED:
        return 'RED'
    elif psi >= PSI_YELLOW:
        return 'YELLOW'
    return 'GREEN'


# =============================================================================
# MOVEMENT LABELING
# =============================================================================
def label_movement(df, horizon=LABEL_HORIZON, threshold=LABEL_THRESHOLD):
    """Label bars where future range >= threshold as 'movement'."""
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    if n <= horizon:
        df['movement'] = np.nan
        return df

    high_windows = sliding_window_view(highs[1:], horizon)
    low_windows = sliding_window_view(lows[1:], horizon)
    future_range = np.full(n, np.nan)
    future_range[:len(high_windows)] = high_windows.max(axis=1) - low_windows.min(axis=1)
    df['movement'] = (future_range >= threshold).astype(float)
    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_model(train_data):
    """Train RF movement predictor on given data. Returns (model, auc) or (None, 0)."""
    X = train_data[MODEL_FEATURES].fillna(0)
    y = train_data['movement']

    if y.nunique() < 2 or len(y) < 200:
        return None, 0.0

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, max_features='sqrt',
        min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    # In-sample AUC as sanity check
    probs = rf.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = 0.5
    return rf, auc


# =============================================================================
# OCO TRADE SIMULATION
# =============================================================================
def simulate_oco(df, bar_idx, tp=OCO_TP, sl=OCO_SL, expiry=OCO_EXPIRY):
    """
    Simulate OCO at bar_idx: place both long and short.
    Long: TP = entry + tp, SL = entry - sl
    Short: TP = entry - tp, SL = entry + sl
    Whichever side's SL hits first gets cancelled; surviving side runs to TP/SL/expiry.

    Returns dict with trade result.
    """
    if bar_idx + 1 >= len(df):
        return {'pnl': 0, 'result': 'NO_BARS', 'direction': None, 'bars_held': 0}

    entry = df['Close'].iloc[bar_idx]
    end_idx = min(bar_idx + 1 + expiry, len(df))

    long_tp = entry + tp
    long_sl = entry - sl
    short_tp = entry - tp
    short_sl = entry + sl

    # Phase 1: determine which side survives (which SL gets hit first)
    long_alive = True
    short_alive = True

    for k in range(bar_idx + 1, end_idx):
        h = df['High'].iloc[k]
        l = df['Low'].iloc[k]

        # Check SL hits
        long_sl_hit = l <= long_sl
        short_sl_hit = h >= short_sl

        if long_sl_hit and short_sl_hit:
            # Both SL hit same bar — both cancelled, no trade
            return {'pnl': 0, 'result': 'DOUBLE_SL', 'direction': None, 'bars_held': k - bar_idx}

        if long_sl_hit:
            # Long cancelled, short survives
            long_alive = False
            break
        if short_sl_hit:
            # Short cancelled, long survives
            short_alive = False
            break

        # Check TP hits (if one TP hits before any SL, that side wins)
        long_tp_hit = h >= long_tp
        short_tp_hit = l <= short_tp

        if long_tp_hit and short_tp_hit:
            # Both TP hit same bar — take the one with better R (both = tp)
            return {'pnl': tp, 'result': 'DOUBLE_TP', 'direction': 'LONG', 'bars_held': k - bar_idx}

        if long_tp_hit:
            return {'pnl': tp, 'result': 'TP', 'direction': 'LONG', 'bars_held': k - bar_idx}
        if short_tp_hit:
            return {'pnl': tp, 'result': 'TP', 'direction': 'SHORT', 'bars_held': k - bar_idx}

    # No SL hit in phase 1 and no TP — expiry with no clear direction
    if long_alive and short_alive:
        # Close at current price, take the better side
        final_price = df['Close'].iloc[min(end_idx - 1, len(df) - 1)]
        long_pnl = final_price - entry
        short_pnl = entry - final_price
        if abs(long_pnl) > abs(short_pnl):
            return {'pnl': long_pnl, 'result': 'EXPIRY', 'direction': 'LONG', 'bars_held': expiry}
        else:
            return {'pnl': short_pnl, 'result': 'EXPIRY', 'direction': 'SHORT', 'bars_held': expiry}

    # Phase 2: one side survived, run it to TP/SL/expiry
    if long_alive:
        direction = 'LONG'
        for k2 in range(k + 1, end_idx):
            h2 = df['High'].iloc[k2]
            l2 = df['Low'].iloc[k2]
            if h2 >= long_tp:
                return {'pnl': tp, 'result': 'TP', 'direction': direction, 'bars_held': k2 - bar_idx}
            if l2 <= long_sl:
                return {'pnl': -sl, 'result': 'SL', 'direction': direction, 'bars_held': k2 - bar_idx}
        # Expiry
        final = df['Close'].iloc[min(end_idx - 1, len(df) - 1)]
        return {'pnl': final - entry, 'result': 'EXPIRY', 'direction': direction, 'bars_held': end_idx - bar_idx - 1}

    else:  # short_alive
        direction = 'SHORT'
        for k2 in range(k + 1, end_idx):
            h2 = df['High'].iloc[k2]
            l2 = df['Low'].iloc[k2]
            if l2 <= short_tp:
                return {'pnl': tp, 'result': 'TP', 'direction': direction, 'bars_held': k2 - bar_idx}
            if h2 >= short_sl:
                return {'pnl': -sl, 'result': 'SL', 'direction': direction, 'bars_held': k2 - bar_idx}
        final = df['Close'].iloc[min(end_idx - 1, len(df) - 1)]
        return {'pnl': entry - final, 'result': 'EXPIRY', 'direction': direction, 'bars_held': end_idx - bar_idx - 1}


# =============================================================================
# MAIN BACKTEST
# =============================================================================
print('='*80)
print('INTEGRATED SYSTEM: Movement Predictor + Drift Detection')
print('='*80)

print('\nLoading data...')
df_raw = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t',
                     names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df_raw['Datetime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'],
                                     format='%Y.%m.%d %H:%M:%S')
df_raw.set_index('Datetime', inplace=True)
df_raw = df_raw[['Open','High','Low','Close']].copy()
print(f'  Raw bars: {len(df_raw):,}')

print('Engineering features...')
df = create_microstructure_features(df_raw)
print(f'  Featured bars: {len(df):,}')

print('Labeling movement...')
df = label_movement(df)
df = df.dropna(subset=['movement'])
print(f'  Labeled bars: {len(df):,}, movement rate: {df["movement"].mean()*100:.1f}%')

# =============================================================================
# Walk-forward backtest
# =============================================================================
# Start after enough data for initial training
START_BAR = TRAIN_WINDOW + 1000  # extra buffer for feature warm-up
total_bars = len(df)

print(f'\n  Train window: {TRAIN_WINDOW} bars')
print(f'  Drift check every: {DRIFT_CHECK} bars')
print(f'  RF threshold: {RF_THRESHOLD}')
print(f'  OCO: TP=${OCO_TP}, SL=${OCO_SL}, expiry={OCO_EXPIRY} bars')
print(f'  PSI thresholds: YELLOW={PSI_YELLOW}, RED={PSI_RED}')
print(f'  Backtesting from bar {START_BAR:,} to {total_bars:,}')

# State
model = None
ref_data = None
model_auc = 0.0
retrain_count = 0
last_drift_check = 0
current_drift = 'GREEN'
current_psi = 0.0

# Trade tracking
trades_with_drift = []      # trades taken WITH drift gating
trades_without_drift = []   # trades taken WITHOUT drift gating (always trade)
drift_log = []
retrain_log = []

# Initial training
print(f'\n--- Initial training on bars 0-{START_BAR} ---')
train_slice = df.iloc[START_BAR - TRAIN_WINDOW:START_BAR]
model, model_auc = train_model(train_slice)
ref_data = train_slice.copy()
retrain_count += 1
print(f'  Model AUC (in-sample): {model_auc:.3f}')

if model is None:
    print('ERROR: Failed to train initial model. Exiting.')
    sys.exit(1)

# Walk forward
print(f'\n--- Walking forward ---')
bar = START_BAR
trade_cooldown = 0  # bars to wait after a trade (avoid overlapping)
signal_count = 0
skipped_red = 0
skipped_yellow = 0

while bar < total_bars - OCO_EXPIRY - 1:
    # --- Drift check ---
    if bar - last_drift_check >= DRIFT_CHECK:
        recent_start = max(0, bar - DRIFT_CHECK)
        recent_data = df.iloc[recent_start:bar]
        current_psi = compute_drift_score(ref_data, recent_data, DRIFT_FEATURES)
        current_drift = classify_drift(current_psi)
        last_drift_check = bar

        drift_log.append({
            'bar': bar,
            'datetime': df.index[bar],
            'psi': current_psi,
            'drift': current_drift,
        })

        # RED → retrain
        if current_drift == 'RED':
            retrain_start = max(0, bar - TRAIN_WINDOW)
            train_slice = df.iloc[retrain_start:bar]
            new_model, new_auc = train_model(train_slice)
            if new_model is not None:
                model = new_model
                model_auc = new_auc
                ref_data = train_slice.copy()
                retrain_count += 1
                retrain_log.append({
                    'bar': bar,
                    'datetime': df.index[bar],
                    'psi': current_psi,
                    'new_auc': new_auc,
                })
                current_drift = 'GREEN'  # reset after retrain
                current_psi = 0.0

    # --- Cooldown ---
    if trade_cooldown > 0:
        trade_cooldown -= 1
        bar += 1
        continue

    # --- RF prediction ---
    X_bar = df.iloc[[bar]][MODEL_FEATURES].fillna(0)
    prob = model.predict_proba(X_bar)[0, 1]

    if prob >= RF_THRESHOLD:
        signal_count += 1

        # WITHOUT drift gating: always trade
        trade_result = simulate_oco(df, bar)
        trades_without_drift.append({
            'bar': bar,
            'datetime': df.index[bar],
            'prob': prob,
            'psi': current_psi,
            'drift': current_drift,
            **trade_result,
        })

        # WITH drift gating: only trade if GREEN or YELLOW (reduced size)
        if current_drift == 'GREEN':
            trades_with_drift.append({
                'bar': bar,
                'datetime': df.index[bar],
                'prob': prob,
                'psi': current_psi,
                'drift': current_drift,
                'size_mult': 1.0,
                **trade_result,
            })
        elif current_drift == 'YELLOW':
            # Trade with half size
            adjusted = trade_result.copy()
            adjusted['pnl'] = trade_result['pnl'] * 0.5
            trades_with_drift.append({
                'bar': bar,
                'datetime': df.index[bar],
                'prob': prob,
                'psi': current_psi,
                'drift': current_drift,
                'size_mult': 0.5,
                **adjusted,
            })
            skipped_yellow += 1
        else:
            skipped_red += 1

        # Cooldown: don't trade again for OCO_EXPIRY bars
        trade_cooldown = OCO_EXPIRY

    bar += 1

    # Progress
    if (bar - START_BAR) % 20000 == 0:
        pct = (bar - START_BAR) / (total_bars - START_BAR) * 100
        print(f'  Bar {bar:,}/{total_bars:,} ({pct:.0f}%) | '
              f'signals={signal_count} | retrains={retrain_count} | '
              f'drift={current_drift} PSI={current_psi:.3f}')


# =============================================================================
# RESULTS
# =============================================================================
print(f'\n\n{"="*80}')
print('RESULTS — Movement Predictor + Drift Detection Backtest')
print(f'{"="*80}')

print(f'\n  Model retrains: {retrain_count}')
print(f'  Total RF signals (prob >= {RF_THRESHOLD}): {signal_count}')
print(f'  Signals skipped (RED drift): {skipped_red}')
print(f'  Signals at half size (YELLOW): {skipped_yellow}')

# --- Without drift gating ---
df_no = pd.DataFrame(trades_without_drift)
if len(df_no) > 0:
    print(f'\n  {"─"*60}')
    print(f'  WITHOUT DRIFT GATING (always trade on RF signal)')
    print(f'  {"─"*60}')
    print(f'  Total trades:   {len(df_no)}')
    print(f'  Total PnL:      ${df_no["pnl"].sum():+.2f}')
    print(f'  Avg PnL/trade:  ${df_no["pnl"].mean():+.2f}')
    print(f'  Win rate:        {(df_no["pnl"] > 0).mean()*100:.1f}%')
    print(f'  Avg winner:      ${df_no[df_no["pnl"]>0]["pnl"].mean():+.2f}' if (df_no['pnl']>0).any() else '  Avg winner: N/A')
    print(f'  Avg loser:       ${df_no[df_no["pnl"]<0]["pnl"].mean():+.2f}' if (df_no['pnl']<0).any() else '  Avg loser: N/A')

    # Results by outcome type
    print(f'\n  By result type:')
    for rt in ['TP', 'SL', 'EXPIRY', 'DOUBLE_SL', 'NO_BARS']:
        sub = df_no[df_no['result'] == rt]
        if len(sub) > 0:
            print(f'    {rt:<12} {len(sub):>4} trades  PnL=${sub["pnl"].sum():>+8.2f}  avg=${sub["pnl"].mean():>+6.2f}')

    # Cumulative PnL
    cum = df_no['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    print(f'\n  Max drawdown:    ${max_dd:+.2f}')
    print(f'  Final equity:    ${cum.iloc[-1]:+.2f}')

# --- With drift gating ---
df_yes = pd.DataFrame(trades_with_drift)
if len(df_yes) > 0:
    print(f'\n  {"─"*60}')
    print(f'  WITH DRIFT GATING (skip RED, half-size YELLOW)')
    print(f'  {"─"*60}')
    print(f'  Total trades:   {len(df_yes)}')
    print(f'  Total PnL:      ${df_yes["pnl"].sum():+.2f}')
    print(f'  Avg PnL/trade:  ${df_yes["pnl"].mean():+.2f}')
    print(f'  Win rate:        {(df_yes["pnl"] > 0).mean()*100:.1f}%')
    print(f'  Avg winner:      ${df_yes[df_yes["pnl"]>0]["pnl"].mean():+.2f}' if (df_yes['pnl']>0).any() else '  Avg winner: N/A')
    print(f'  Avg loser:       ${df_yes[df_yes["pnl"]<0]["pnl"].mean():+.2f}' if (df_yes['pnl']<0).any() else '  Avg loser: N/A')

    cum2 = df_yes['pnl'].cumsum()
    max_dd2 = (cum2 - cum2.cummax()).min()
    print(f'\n  Max drawdown:    ${max_dd2:+.2f}')
    print(f'  Final equity:    ${cum2.iloc[-1]:+.2f}')

    # Green vs Yellow trades
    green = df_yes[df_yes['drift'] == 'GREEN']
    yellow = df_yes[df_yes['drift'] == 'YELLOW']
    if len(green) > 0:
        print(f'\n  GREEN trades:  {len(green)} trades, PnL=${green["pnl"].sum():+.2f}, '
              f'win={((green["pnl"]>0).mean()*100):.0f}%, avg=${green["pnl"].mean():+.2f}')
    if len(yellow) > 0:
        print(f'  YELLOW trades: {len(yellow)} trades, PnL=${yellow["pnl"].sum():+.2f}, '
              f'win={((yellow["pnl"]>0).mean()*100):.0f}%, avg=${yellow["pnl"].mean():+.2f}')

# --- Comparison ---
if len(df_no) > 0 and len(df_yes) > 0:
    print(f'\n  {"─"*60}')
    print(f'  COMPARISON')
    print(f'  {"─"*60}')
    no_total = df_no['pnl'].sum()
    yes_total = df_yes['pnl'].sum()
    improvement = yes_total - no_total

    print(f'  Without drift gate:  ${no_total:+.2f} ({len(df_no)} trades)')
    print(f'  With drift gate:     ${yes_total:+.2f} ({len(df_yes)} trades)')
    print(f'  Improvement:         ${improvement:+.2f}')

    if no_total != 0:
        print(f'  Pct improvement:     {improvement / abs(no_total) * 100:+.0f}%')

    no_dd = (df_no['pnl'].cumsum() - df_no['pnl'].cumsum().cummax()).min()
    yes_dd = (df_yes['pnl'].cumsum() - df_yes['pnl'].cumsum().cummax()).min()
    print(f'  Drawdown without:    ${no_dd:+.2f}')
    print(f'  Drawdown with:       ${yes_dd:+.2f}')

# --- Drift timeline ---
if drift_log:
    df_drift = pd.DataFrame(drift_log)
    print(f'\n  {"─"*60}')
    print(f'  DRIFT TIMELINE (sampled)')
    print(f'  {"─"*60}')

    # Show every 10th entry or all if small
    step = max(1, len(df_drift) // 30)
    print(f'  {"Bar":<10} {"Datetime":<22} {"PSI":<10} {"Status":<8}')
    for i in range(0, len(df_drift), step):
        row = df_drift.iloc[i]
        print(f'  {int(row["bar"]):<10} {str(row["datetime"])[:19]:<22} '
              f'{row["psi"]:<10.4f} {row["drift"]:<8}')

# --- Retrain log ---
if retrain_log:
    print(f'\n  {"─"*60}')
    print(f'  RETRAIN EVENTS')
    print(f'  {"─"*60}')
    for r in retrain_log:
        print(f'  Bar {r["bar"]:>8,} | {str(r["datetime"])[:19]} | '
              f'PSI={r["psi"]:.3f} → retrained (AUC={r["new_auc"]:.3f})')

# Save results
output = 'data/processed/drift_movement_backtest.csv'
if len(df_no) > 0:
    df_no.to_csv(output.replace('.csv', '_no_gate.csv'), index=False)
if len(df_yes) > 0:
    df_yes.to_csv(output.replace('.csv', '_with_gate.csv'), index=False)
print(f'\n  Saved trade logs to data/processed/')

print(f'\n{"="*80}')
print('DONE')
print(f'{"="*80}')
