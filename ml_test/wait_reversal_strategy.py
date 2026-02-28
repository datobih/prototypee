"""
Wait-for-Reversal Strategy
===========================
Logic:
  1. RF >= threshold fires -> detect signal candle direction
  2. Wait up to WAIT_WINDOW bars for a candle that moves OPPOSITE to signal
  3. When reversal candle forms, enter at its Close
  4. Set TP/SL and simulate bar-by-bar from there
  5. Cooldown after each signal (skip next N bars for new signals)

This exploits the finding that ~89% of RF signals see a reversal within 3 bars,
but the immediate next bar only reverses 57% of the time.
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
RF_THRESHOLD = 0.85
WAIT_WINDOW = 2      # max bars to wait for reversal candle
COOLDOWN = 3          # bars to skip after a signal
TARGET = 3.0          # TP in dollars
STOP = 1.0            # SL in dollars
HORIZON = 35          # max bars to hold after entry

# ============================================================================
# LOAD DATA
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'processed', 'XAUUSD1_feature_analysis.csv')

print('='*80)
print('WAIT-FOR-REVERSAL STRATEGY')
print('='*80)
print(f'RF Threshold:  >= {RF_THRESHOLD}')
print(f'Wait Window:   {WAIT_WINDOW} bars for reversal candle')
print(f'Cooldown:      {COOLDOWN} bars after each signal')
print(f'TP: ${TARGET}   SL: ${STOP}   Horizon: {HORIZON} bars')
print(f'RR: {TARGET/STOP:.1f}:1   Breakeven WR: {STOP/(TARGET+STOP)*100:.1f}%')
print()

df = pd.read_csv(data_path)
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f'Loaded {len(df)} bars')

# Candle direction
df['candle_dir'] = np.where(df['Close'] > df['Open'], 'BULL',
                   np.where(df['Close'] < df['Open'], 'BEAR', 'DOJI'))

# Session
df['hour'] = df['Datetime'].dt.hour
def get_session(hour):
    if 0 <= hour < 8:
        return 'ASIAN'
    elif 8 <= hour < 13:
        return 'LONDON'
    elif 13 <= hour < 17:
        return 'NY_OVERLAP'
    elif 17 <= hour < 22:
        return 'NY'
    else:
        return 'LATE'

df['session'] = df['hour'].apply(get_session)

# ============================================================================
# STRATEGY SIMULATION
# ============================================================================
# Pre-extract arrays for speed
highs_arr = df['High'].values
lows_arr = df['Low'].values
closes_arr = df['Close'].values
opens_arr = df['Open'].values
rf_probs_arr = df['rf_prob'].values
candle_dirs_arr = df['candle_dir'].values
datetimes_arr = df['Datetime'].values
sessions_arr = df['session'].values
n_bars = len(df)

trades = []
cooldown_until = -1
no_reversal_count = 0

for i in range(n_bars - WAIT_WINDOW - HORIZON - 1):
    # Skip if in cooldown
    if i <= cooldown_until:
        continue

    # Check for RF signal
    if rf_probs_arr[i] < RF_THRESHOLD:
        continue

    signal_dir = candle_dirs_arr[i]
    if signal_dir == 'DOJI':
        continue

    # Expected reversal direction
    reversal_dir = 'BEAR' if signal_dir == 'BULL' else 'BULL'

    # --- PHASE 1: Wait for reversal candle within WAIT_WINDOW ---
    entry_bar = None
    for w in range(1, WAIT_WINDOW + 1):
        if candle_dirs_arr[i + w] == reversal_dir:
            entry_bar = i + w
            break

    if entry_bar is None:
        # No reversal candle found in window — skip
        no_reversal_count += 1
        cooldown_until = i + COOLDOWN
        continue

    # --- PHASE 2: Enter at reversal candle's Close ---
    entry_price = closes_arr[entry_bar]
    entry_time = datetimes_arr[entry_bar]

    # Trade direction = reversal direction
    if reversal_dir == 'BULL':
        trade_dir = 'LONG'
        tp_price = entry_price + TARGET
        sl_price = entry_price - STOP
    else:
        trade_dir = 'SHORT'
        tp_price = entry_price - TARGET
        sl_price = entry_price + STOP

    # --- PHASE 3: Simulate trade bar-by-bar ---
    result = 'TIMEOUT'
    exit_price = entry_price
    exit_bar = entry_bar + HORIZON  # default

    end_bar = min(entry_bar + HORIZON + 1, n_bars)
    for j in range(entry_bar + 1, end_bar):
        bar_h = highs_arr[j]
        bar_l = lows_arr[j]

        if trade_dir == 'LONG':
            if bar_l <= sl_price:
                result = 'LOSS'
                exit_price = sl_price
                exit_bar = j
                break
            if bar_h >= tp_price:
                result = 'WIN'
                exit_price = tp_price
                exit_bar = j
                break
        else:  # SHORT
            if bar_h >= sl_price:
                result = 'LOSS'
                exit_price = sl_price
                exit_bar = j
                break
            if bar_l <= tp_price:
                result = 'WIN'
                exit_price = tp_price
                exit_bar = j
                break

    # Timeout P&L
    if result == 'WIN':
        pnl = TARGET
    elif result == 'LOSS':
        pnl = -STOP
    else:
        last_idx = min(entry_bar + HORIZON, n_bars - 1)
        exit_price = closes_arr[last_idx]
        if trade_dir == 'LONG':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

    trades.append({
        'signal_time': datetimes_arr[i],
        'entry_time': entry_time,
        'session': sessions_arr[i],
        'rf_prob': round(rf_probs_arr[i], 3),
        'signal_dir': signal_dir,
        'wait_bars': entry_bar - i,
        'trade_dir': trade_dir,
        'entry_price': round(entry_price, 2),
        'exit_price': round(exit_price, 2),
        'result': result,
        'pnl': round(pnl, 2),
        'hold_bars': exit_bar - entry_bar
    })

    # Apply cooldown from the signal bar
    cooldown_until = i + COOLDOWN

# ============================================================================
# RESULTS
# ============================================================================
trades_df = pd.DataFrame(trades)

print(f'\n{"="*80}')
print('RESULTS')
print('='*80)

print(f'\nTotal RF signals: {len(trades) + no_reversal_count}')
print(f'Reversal found in {WAIT_WINDOW}-bar window: {len(trades)} ({len(trades)/(len(trades)+no_reversal_count)*100:.1f}%)')
print(f'No reversal (skipped): {no_reversal_count}')

if len(trades_df) == 0:
    print('No trades generated.')
    exit()

wins = (trades_df['result'] == 'WIN').sum()
losses = (trades_df['result'] == 'LOSS').sum()
timeouts = (trades_df['result'] == 'TIMEOUT').sum()
total = len(trades_df)
win_rate = wins / total * 100

total_pnl = trades_df['pnl'].sum()
avg_pnl = trades_df['pnl'].mean()

print(f'\n--- OVERALL PERFORMANCE ---')
print(f'Trades:   {total}')
print(f'Wins:     {wins}')
print(f'Losses:   {losses}')
print(f'Timeouts: {timeouts}')
print(f'Win Rate: {win_rate:.1f}%')
print(f'\nTotal P&L:   ${total_pnl:,.2f}')
print(f'Avg P&L:     ${avg_pnl:.4f} per trade')
print(f'RR:          {TARGET}:{STOP} = {TARGET/STOP:.1f}:1')
print(f'Breakeven:   {STOP/(TARGET+STOP)*100:.1f}%')

# Profit factor
gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
print(f'Profit Factor: {pf:.2f}')

# Wait bar distribution
print(f'\n--- WAIT BARS DISTRIBUTION ---')
for w in range(1, WAIT_WINDOW + 1):
    w_trades = trades_df[trades_df['wait_bars'] == w]
    if len(w_trades) > 0:
        w_wins = (w_trades['result'] == 'WIN').sum()
        w_wr = w_wins / len(w_trades) * 100
        w_pnl = w_trades['pnl'].sum()
        print(f'Entered on bar +{w}: {len(w_trades):>4} trades  '
              f'WR: {w_wr:>5.1f}%  P&L: ${w_pnl:>8,.2f}')

# By signal direction
print(f'\n--- BY SIGNAL DIRECTION ---')
for sig_dir in ['BULL', 'BEAR']:
    sub = trades_df[trades_df['signal_dir'] == sig_dir]
    if len(sub) > 0:
        sub_wins = (sub['result'] == 'WIN').sum()
        sub_wr = sub_wins / len(sub) * 100
        sub_pnl = sub['pnl'].sum()
        trade_d = 'SHORT' if sig_dir == 'BULL' else 'LONG'
        print(f'{sig_dir} signal -> {trade_d}: {len(sub):>4} trades  '
              f'WR: {sub_wr:>5.1f}%  P&L: ${sub_pnl:>8,.2f}')

# By session
print(f'\n--- BY SESSION ---')
for sess in ['NY_OVERLAP', 'NY', 'LONDON', 'ASIAN', 'LATE']:
    sub = trades_df[trades_df['session'] == sess]
    if len(sub) > 0:
        sub_wins = (sub['result'] == 'WIN').sum()
        sub_wr = sub_wins / len(sub) * 100
        sub_pnl = sub['pnl'].sum()
        print(f'{sess:<12} {len(sub):>4} trades  '
              f'WR: {sub_wr:>5.1f}%  P&L: ${sub_pnl:>8,.2f}')

# By RF probability bucket
print(f'\n--- BY RF PROBABILITY ---')
buckets = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]
for lo, hi in buckets:
    sub = trades_df[(trades_df['rf_prob'] >= lo) & (trades_df['rf_prob'] < hi)]
    if len(sub) > 0:
        sub_wins = (sub['result'] == 'WIN').sum()
        sub_wr = sub_wins / len(sub) * 100
        sub_pnl = sub['pnl'].sum()
        print(f'RF [{lo:.2f}, {hi:.2f}):  {len(sub):>4} trades  '
              f'WR: {sub_wr:>5.1f}%  P&L: ${sub_pnl:>8,.2f}')

# ============================================================================
# OPTUNA PARAMETER OPTIMIZATION
# ============================================================================
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Pre-compute signal entries (same for every TP/SL combo)
# Reuse arrays from main simulation
entries = []  # list of (entry_bar_idx, trade_dir, entry_price)
cd_until = -1
for i in range(n_bars - WAIT_WINDOW - HORIZON - 1):
    if i <= cd_until:
        continue
    if rf_probs_arr[i] < RF_THRESHOLD:
        continue
    sig_d = candle_dirs_arr[i]
    if sig_d == 'DOJI':
        continue
    rev_d = 'BEAR' if sig_d == 'BULL' else 'BULL'

    eb = None
    for w in range(1, WAIT_WINDOW + 1):
        if candle_dirs_arr[i + w] == rev_d:
            eb = i + w
            break
    if eb is None:
        cd_until = i + COOLDOWN
        continue

    ep = closes_arr[eb]
    td = 'LONG' if rev_d == 'BULL' else 'SHORT'
    entries.append((eb, td, ep))
    cd_until = i + COOLDOWN

print(f'\n--- OPTUNA PARAMETER OPTIMIZATION ---')
print(f'Pre-computed {len(entries)} entry points')

def simulate_params(tp_val, sl_val):
    """Simulate all trades with given TP/SL and return stats."""
    pnls = []
    for eb, td, ep in entries:
        tp_p = ep + tp_val if td == 'LONG' else ep - tp_val
        sl_p = ep - sl_val if td == 'LONG' else ep + sl_val

        res = 'TIMEOUT'
        end_j = min(eb + HORIZON + 1, n_bars)
        for j in range(eb + 1, end_j):
            bh = highs_arr[j]
            bl = lows_arr[j]
            if td == 'LONG':
                if bl <= sl_p:
                    res = 'LOSS'
                    break
                if bh >= tp_p:
                    res = 'WIN'
                    break
            else:
                if bh >= sl_p:
                    res = 'LOSS'
                    break
                if bl <= tp_p:
                    res = 'WIN'
                    break

        if res == 'WIN':
            pnls.append(tp_val)
        elif res == 'LOSS':
            pnls.append(-sl_val)
        else:
            lp = closes_arr[min(eb + HORIZON, n_bars - 1)]
            p = (lp - ep) if td == 'LONG' else (ep - lp)
            pnls.append(p)

    arr = np.array(pnls)
    wins = (arr > 0).sum()
    total = len(arr)
    wr = wins / total * 100 if total > 0 else 0
    total_pnl = arr.sum()
    gp = arr[arr > 0].sum()
    gl = abs(arr[arr < 0].sum())
    pf = gp / gl if gl > 0 else 0
    return total_pnl, wr, pf, total

def objective(trial):
    tp_val = trial.suggest_float('tp', 0.5, 8.0, step=0.25)
    sl_val = trial.suggest_float('sl', 0.5, 8.0, step=0.25)
    total_pnl, wr, pf, total = simulate_params(tp_val, sl_val)
    # Store extra metrics for later
    trial.set_user_attr('wr', wr)
    trial.set_user_attr('pf', pf)
    trial.set_user_attr('trades', total)
    return total_pnl

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
print('Running 200 Optuna trials...')
study.optimize(objective, n_trials=200, show_progress_bar=False)

# Show top 20 results
print(f'\n--- TOP 20 PARAMETER COMBOS (by P&L) ---')
print(f'{"TP":>6} {"SL":>6} {"RR":>5} {"WR":>6} {"P&L":>10} {"PF":>6} {"Trades":>6}')
print('-' * 55)

trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
for t in trials_sorted[:20]:
    tp_v = t.params['tp']
    sl_v = t.params['sl']
    rr = tp_v / sl_v
    wr = t.user_attrs.get('wr', 0)
    pf = t.user_attrs.get('pf', 0)
    trades_n = t.user_attrs.get('trades', 0)
    pnl = t.value
    marker = ' <<<' if pnl > 0 else ''
    print(f'{tp_v:>6.2f} {sl_v:>6.2f} {rr:>5.1f} '
          f'{wr:>5.1f}% ${pnl:>8,.2f} {pf:>5.2f} {trades_n:>6}{marker}')

best = study.best_trial
print(f'\n*** BEST: TP=${best.params["tp"]:.2f}  SL=${best.params["sl"]:.2f}  '
      f'P&L=${best.value:,.2f}  WR={best.user_attrs["wr"]:.1f}%  '
      f'PF={best.user_attrs["pf"]:.2f}  '
      f'RR={best.params["tp"]/best.params["sl"]:.1f}:1 ***')

# Save all trial results to CSV
trials_data = []
for t in study.trials:
    if t.value is not None:
        tp_v = t.params['tp']
        sl_v = t.params['sl']
        trials_data.append({
            'tp': tp_v,
            'sl': sl_v,
            'rr': round(tp_v / sl_v, 2),
            'win_rate': round(t.user_attrs.get('wr', 0), 2),
            'pnl': round(t.value, 2),
            'profit_factor': round(t.user_attrs.get('pf', 0), 2),
            'trades': t.user_attrs.get('trades', 0)
        })
trials_results_df = pd.DataFrame(trials_data)
trials_results_df = trials_results_df.sort_values('pnl', ascending=False).reset_index(drop=True)
optuna_out_path = os.path.join(script_dir, '..', 'data', 'processed', 'WAIT_REVERSAL_optuna_trials.csv')
trials_results_df.to_csv(optuna_out_path, index=False)
print(f'\nOptuna trials saved: {optuna_out_path} ({len(trials_results_df)} trials)')

# ============================================================================
# SAVE RESULTS
# ============================================================================
out_path = os.path.join(script_dir, '..', 'data', 'processed', 'WAIT_REVERSAL_strategy.csv')
trades_df['signal_time'] = trades_df['signal_time'].dt.strftime('%Y-%m-%d %H:%M')
trades_df['entry_time'] = trades_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
trades_df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')

# Sample trades
print(f'\n--- SAMPLE TRADES (first 25) ---')
print(trades_df.head(25).to_string(index=False))

print(f'\n{"="*80}')
print('STRATEGY TEST COMPLETE')
print('='*80)
