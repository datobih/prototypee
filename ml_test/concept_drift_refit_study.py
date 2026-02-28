"""
concept_drift_refit_study.py

Tests TWO ideas to improve carry-forward performance of SMA+RSI on XAUUSD 15m:

  A) SHORTER REFIT CYCLES
     - Lookback windows: 1w, 2w, 3w
     - Refit (step) every: 3d, 5d, 1w
     - OOS test = the step window itself

  B) TOP-N ENSEMBLE
     - Instead of betting on the single #1 combo, use the top-N combos
     - Average their OOS PnL (equal weight)
     - Test N = 1, 3, 5, 10

  Output: grid of (lookback × refit_freq × top_N) → OOS metrics
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
from itertools import product
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load Data
# =============================================================================
print('='*80)
print('REFIT STUDY — Shorter Cycles + Top-N Ensemble on XAUUSD 15m')
print('='*80)

print('\nLoading data...')
df = pd.read_csv(
    'data/raw/XAUUSD15.csv', sep='\t',
    names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread']
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df = df.set_index('Datetime').dropna(subset=['Close'])
df = df.loc['2024-01-01':'2025-12-31']
print(f'  Bars: {len(df):,} ({df.index[0].date()} → {df.index[-1].date()})')

# =============================================================================
# 2. Parameter Grid (matches vbt_sma_rsi_sweep.py)
# =============================================================================
fast_windows = np.arange(10, 41, 10)
slow_windows = np.arange(50, 151, 50)
rsi_long_thresholds  = [50, 55, 60]
rsi_short_thresholds = [50, 45, 40]

combinations = [
    (f, s, rl, rs)
    for f in fast_windows for s in slow_windows if f < s
    for rl in rsi_long_thresholds for rs in rsi_short_thresholds
]
n_combos = len(combinations)
print(f'  Parameter combos: {n_combos}')

def combo_name(c):
    return f"F{c[0]}_S{c[1]}_RL{c[2]}_RS{c[3]}"

# =============================================================================
# 3. Sweep function — returns PnL array for all combos
# =============================================================================
def sweep_period(data_slice):
    """
    Run all SMA+RSI combos on a data slice.
    Returns dict: {combo_tuple: {'pnl': float, 'trades': int}} or None.
    """
    if len(data_slice) < 200:
        return None

    price = data_slice['Close']
    high  = data_slice['High']
    low   = data_slice['Low']
    open_ = data_slice['Open']

    n = len(combinations)
    c_fast, c_slow, c_rsi_l, c_rsi_s = zip(*combinations)

    try:
        sma_fast = vbt.MA.run(price, window=list(c_fast), short_name='fast')
        sma_slow = vbt.MA.run(price, window=list(c_slow), short_name='slow')
        rsi = vbt.RSI.run(price, window=14).rsi
    except Exception:
        return None

    col_idx = pd.MultiIndex.from_tuples(combinations)
    rsi_df   = pd.DataFrame(np.tile(rsi.values, (n, 1)).T, index=price.index, columns=col_idx)
    rsi_l_df = pd.DataFrame(np.tile(c_rsi_l, (len(price), 1)), index=price.index, columns=col_idx)
    rsi_s_df = pd.DataFrame(np.tile(c_rsi_s, (len(price), 1)), index=price.index, columns=col_idx)

    ma_f_df = sma_fast.ma.copy()
    ma_s_df = sma_slow.ma.copy()
    ma_f_df.columns = col_idx
    ma_s_df.columns = col_idx

    ma_f_prev = ma_f_df.shift(1)
    ma_s_prev = ma_s_df.shift(1)

    long_entries  = (ma_f_prev <= ma_s_prev) & (ma_f_df > ma_s_df) & (rsi_df > rsi_l_df)
    short_entries = (ma_f_prev >= ma_s_prev) & (ma_f_df < ma_s_df) & (rsi_df < rsi_s_df)

    exits       = pd.DataFrame(False, index=price.index, columns=col_idx)
    short_exits = pd.DataFrame(False, index=price.index, columns=col_idx)

    try:
        portfolio = vbt.Portfolio.from_signals(
            price, open=open_, high=high, low=low,
            entries=long_entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            upon_opposite_entry='close',
            size=1.0, size_type='amount',
            init_cash=1000000, fees=0.0001,
            sl_stop=0.002, tp_stop=0.005, freq='15min'
        )
        pnl    = portfolio.total_profit()
        trades = portfolio.trades.count()
    except Exception:
        return None

    result = {}
    for i, combo in enumerate(combinations):
        result[combo] = {
            'pnl':    pnl.values[i] if i < len(pnl) else 0,
            'trades': int(trades.values[i]) if i < len(trades) else 0,
        }
    return result


# =============================================================================
# 4. Configuration grid
# =============================================================================
BARS_PER_DAY = 96  # 15-min bars per day

lookback_configs = {
    '1w':  7 * BARS_PER_DAY,
    '2w': 14 * BARS_PER_DAY,
    '3w': 21 * BARS_PER_DAY,
}
refit_configs = {
    '3d':  3 * BARS_PER_DAY,
    '5d':  5 * BARS_PER_DAY,
    '1w':  7 * BARS_PER_DAY,
}
top_n_values = [1, 3, 5, 10]

total_bars = len(df)

# =============================================================================
# 5. Walk-forward engine
# =============================================================================
print(f'\n--- Running walk-forward grid ---')
print(f'  Lookbacks: {list(lookback_configs.keys())}')
print(f'  Refit every: {list(refit_configs.keys())}')
print(f'  Top-N: {top_n_values}')

grid_results = []

for lb_name, lb_bars in lookback_configs.items():
    for rf_name, rf_bars in refit_configs.items():
        # Number of walk-forward steps
        n_steps = (total_bars - lb_bars) // rf_bars
        if n_steps < 5:
            continue

        config_label = f'LB={lb_name} RF={rf_name}'
        print(f'\n  {config_label}: {n_steps} steps ... ', end='', flush=True)

        # Pre-compute: for each step, get IS results and OOS results
        step_data = []
        for step in range(n_steps):
            oos_start = lb_bars + step * rf_bars
            oos_end   = oos_start + rf_bars

            if oos_end > total_bars:
                break

            is_data  = df.iloc[oos_start - lb_bars : oos_start]
            oos_data = df.iloc[oos_start : oos_end]

            is_result  = sweep_period(is_data)
            oos_result = sweep_period(oos_data)

            if is_result is None or oos_result is None:
                continue

            # Rank combos by IS PnL (min 2 trades to be valid)
            valid_is = {k: v for k, v in is_result.items() if v['trades'] >= 2}
            if len(valid_is) < 10:
                continue

            ranked = sorted(valid_is.keys(), key=lambda k: valid_is[k]['pnl'], reverse=True)

            step_data.append({
                'ranked': ranked,
                'is_result': is_result,
                'oos_result': oos_result,
            })

        if len(step_data) < 5:
            print(f'SKIP ({len(step_data)} valid)')
            continue

        print(f'{len(step_data)} valid', end='')

        # Now evaluate each top-N
        for top_n in top_n_values:
            oos_pnls = []
            oos_positive = 0

            for sd in step_data:
                top_combos = sd['ranked'][:top_n]
                # Average OOS PnL of top-N combos
                pnls = [sd['oos_result'][c]['pnl'] for c in top_combos if c in sd['oos_result']]
                if pnls:
                    avg_pnl = np.mean(pnls)
                    oos_pnls.append(avg_pnl)
                    if avg_pnl > 0:
                        oos_positive += 1

            if not oos_pnls:
                continue

            mean_oos = np.mean(oos_pnls)
            total_oos = np.sum(oos_pnls)
            win_rate = oos_positive / len(oos_pnls) * 100
            # Sharpe-like: mean / std
            sharpe = np.mean(oos_pnls) / (np.std(oos_pnls) + 1e-10)

            grid_results.append({
                'lookback': lb_name,
                'refit': rf_name,
                'top_n': top_n,
                'n_steps': len(oos_pnls),
                'mean_oos': mean_oos,
                'total_oos': total_oos,
                'win_rate': win_rate,
                'sharpe': sharpe,
            })

        print(f' ✓')


# =============================================================================
# 6. Results
# =============================================================================
print(f'\n\n{"="*80}')
print('RESULTS — Full Grid: Lookback × Refit × Top-N')
print(f'{"="*80}')

df_grid = pd.DataFrame(grid_results)

if df_grid.empty:
    print('\n  No valid results.')
else:
    # Sort by total OOS PnL
    df_grid = df_grid.sort_values('total_oos', ascending=False)

    print(f'\n  {"LB":<6} {"Refit":<7} {"TopN":<6} {"Steps":<7} '
          f'{"OOS_avg$":<10} {"OOS_tot$":<11} {"Win%":<8} {"Sharpe":<8}')
    print('  ' + '-'*70)

    for _, r in df_grid.iterrows():
        print(f'  {r["lookback"]:<6} {r["refit"]:<7} {int(r["top_n"]):<6} {int(r["n_steps"]):<7} '
              f'${r["mean_oos"]:>+8.1f}  ${r["total_oos"]:>+9.1f}  '
              f'{r["win_rate"]:>5.0f}%  {r["sharpe"]:>+6.3f}')

    # Best configs
    print(f'\n\n{"="*80}')
    print('TOP 5 CONFIGS by Total OOS PnL')
    print(f'{"="*80}')
    top5 = df_grid.head(5)
    for _, r in top5.iterrows():
        print(f'  LB={r["lookback"]} Refit={r["refit"]} Top-{int(r["top_n"])}:  '
              f'total=${r["total_oos"]:+.1f}  avg=${r["mean_oos"]:+.1f}  '
              f'win={r["win_rate"]:.0f}%  sharpe={r["sharpe"]:+.3f}')

    # Best by Sharpe
    print(f'\nTOP 5 CONFIGS by Sharpe')
    print('-'*80)
    top5s = df_grid.sort_values('sharpe', ascending=False).head(5)
    for _, r in top5s.iterrows():
        print(f'  LB={r["lookback"]} Refit={r["refit"]} Top-{int(r["top_n"])}:  '
              f'sharpe={r["sharpe"]:+.3f}  total=${r["total_oos"]:+.1f}  '
              f'avg=${r["mean_oos"]:+.1f}  win={r["win_rate"]:.0f}%')

    # Top-N effect analysis
    print(f'\n\n{"="*80}')
    print('TOP-N EFFECT — Does ensemble help? (averaged across all LB/Refit configs)')
    print(f'{"="*80}')
    for tn in top_n_values:
        subset = df_grid[df_grid['top_n'] == tn]
        if subset.empty:
            continue
        print(f'  Top-{tn:<3}  avg_oos=${subset["mean_oos"].mean():+.1f}  '
              f'avg_win%={subset["win_rate"].mean():.0f}%  '
              f'avg_sharpe={subset["sharpe"].mean():+.3f}')

    # Lookback effect analysis
    print(f'\n{"="*80}')
    print('LOOKBACK EFFECT — Which lookback works best? (averaged across Refit/TopN)')
    print(f'{"="*80}')
    for lb in lookback_configs:
        subset = df_grid[df_grid['lookback'] == lb]
        if subset.empty:
            continue
        print(f'  {lb:<4}  avg_oos=${subset["mean_oos"].mean():+.1f}  '
              f'avg_win%={subset["win_rate"].mean():.0f}%  '
              f'avg_sharpe={subset["sharpe"].mean():+.3f}')

    # Refit freq effect analysis
    print(f'\n{"="*80}')
    print('REFIT FREQ EFFECT — How often to refit? (averaged across LB/TopN)')
    print(f'{"="*80}')
    for rf in refit_configs:
        subset = df_grid[df_grid['refit'] == rf]
        if subset.empty:
            continue
        print(f'  {rf:<4}  avg_oos=${subset["mean_oos"].mean():+.1f}  '
              f'avg_win%={subset["win_rate"].mean():.0f}%  '
              f'avg_sharpe={subset["sharpe"].mean():+.3f}')

    # Save
    output_path = 'data/processed/concept_drift_refit_grid.csv'
    df_grid.to_csv(output_path, index=False)
    print(f'\n  Saved to {output_path}')

print(f'\n{"="*80}')
print('DONE')
print(f'{"="*80}')
