"""
concept_drift_walkforward.py

2-Week Rolling Sweep: Can you rely on the previous 2 weeks' top combo
for the next 2 weeks?

Methodology:
  1. Split 2024-2025 XAUUSD 15m data into consecutive 2-week periods
  2. Run full SMA+RSI parameter sweep on each period → find that period's #1 combo
  3. For each period (starting from #2), also measure how the PREVIOUS period's
     winner performed in THIS period
  4. Compare: prev_winner PnL vs current_winner PnL → quantify the drift/decay
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load Data
# =============================================================================
print('='*80)
print('CONCEPT DRIFT — 2-Week Rolling Sweep: Prev Winner vs Current Winner')
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
# 3. Sweep function — run all combos on a data slice
# =============================================================================
def sweep_period(data_slice):
    """
    Run all SMA+RSI combos on a data slice.
    Returns dict: {combo_tuple: {'pnl': float, 'trades': int, 'win_rate': float}}
    or None if data is too short.
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
        wr     = portfolio.trades.win_rate()
    except Exception:
        return None

    result = {}
    for i, combo in enumerate(combinations):
        result[combo] = {
            'pnl':      pnl.values[i] if i < len(pnl) else 0,
            'trades':   int(trades.values[i]) if i < len(trades) else 0,
            'win_rate': wr.values[i] * 100 if i < len(wr) and not np.isnan(wr.values[i]) else 0,
        }
    return result


# =============================================================================
# 4. Split into 2-week periods & sweep each
# =============================================================================
PERIOD_DAYS = 14
periods = []
start = df.index[0]
end   = df.index[-1]

cursor = start
while cursor + pd.Timedelta(days=PERIOD_DAYS) <= end:
    p_end = cursor + pd.Timedelta(days=PERIOD_DAYS)
    chunk = df.loc[cursor:p_end]
    if len(chunk) > 200:
        periods.append({
            'start': cursor,
            'end':   p_end,
            'data':  chunk,
        })
    cursor = p_end

print(f'\n  Total 2-week periods: {len(periods)}')
print(f'  Running sweep on each period...\n')

period_results = []
for i, p in enumerate(periods):
    label = f"P{i:02d} {p['start'].strftime('%Y-%m-%d')} → {p['end'].strftime('%Y-%m-%d')}"
    print(f'  [{i+1}/{len(periods)}] {label} ({len(p["data"])} bars) ... ', end='', flush=True)

    result = sweep_period(p['data'])
    if result is None:
        print('FAILED')
        period_results.append(None)
        continue

    # Find top combo (min 3 trades)
    valid = {k: v for k, v in result.items() if v['trades'] >= 3}
    if not valid:
        print('NO VALID COMBOS')
        period_results.append(None)
        continue

    top_combo = max(valid, key=lambda k: valid[k]['pnl'])
    top = valid[top_combo]
    print(f'TOP: {combo_name(top_combo)}  PnL=${top["pnl"]:+.1f}  '
          f'trades={top["trades"]}  WR={top["win_rate"]:.0f}%')

    period_results.append({
        'period_idx': i,
        'label': label,
        'start': p['start'],
        'end': p['end'],
        'all_results': result,
        'top_combo': top_combo,
        'top_pnl': top['pnl'],
        'top_trades': top['trades'],
        'top_wr': top['win_rate'],
    })


# =============================================================================
# 5. Compare: Previous winner vs Current winner per period
# =============================================================================
print(f'\n\n{"="*80}')
print('PERIOD-BY-PERIOD COMPARISON: Prev Winner vs Current Winner')
print(f'{"="*80}')

header = (f'  {"Period":<8} {"Dates":<27} '
          f'{"Prev Winner":<28} {"Prev$":<9} '
          f'{"Curr Winner":<28} {"Curr$":<9} '
          f'{"Gap$":<9} {"Verdict":<10}')
print(f'\n{header}')
print('  ' + '-'*130)

comparison_records = []

for i in range(1, len(period_results)):
    prev = period_results[i - 1]
    curr = period_results[i]

    if prev is None or curr is None:
        continue

    prev_combo = prev['top_combo']
    curr_combo = curr['top_combo']

    # How did prev's top combo do in THIS period?
    prev_winner_this_period = curr['all_results'].get(prev_combo)
    if prev_winner_this_period is None:
        continue

    prev_pnl_here = prev_winner_this_period['pnl']
    prev_trades_here = prev_winner_this_period['trades']
    curr_pnl = curr['top_pnl']

    gap = curr_pnl - prev_pnl_here
    same_combo = prev_combo == curr_combo

    if same_combo:
        verdict = 'SAME'
    elif prev_pnl_here > 0 and prev_pnl_here >= curr_pnl * 0.5:
        verdict = 'GOOD'
    elif prev_pnl_here > 0:
        verdict = 'OK'
    elif prev_pnl_here > -abs(curr_pnl) * 0.25:
        verdict = 'WEAK'
    else:
        verdict = 'FAILED'

    dates = f"{curr['start'].strftime('%Y-%m-%d')} → {curr['end'].strftime('%Y-%m-%d')}"
    print(f'  P{i:<5} {dates:<27} '
          f'{combo_name(prev_combo):<28} ${prev_pnl_here:<+8.1f} '
          f'{combo_name(curr_combo):<28} ${curr_pnl:<+8.1f} '
          f'${gap:<+8.1f} {verdict:<10}')

    comparison_records.append({
        'period': i,
        'start': curr['start'],
        'end': curr['end'],
        'prev_winner': combo_name(prev_combo),
        'prev_winner_pnl': prev_pnl_here,
        'prev_winner_trades': prev_trades_here,
        'curr_winner': combo_name(curr_combo),
        'curr_winner_pnl': curr_pnl,
        'gap': gap,
        'same_combo': same_combo,
        'verdict': verdict,
    })

if not comparison_records:
    print('\n  No valid comparisons.')
else:
    df_comp = pd.DataFrame(comparison_records)

    # ==========================================================================
    # 6. Summary Statistics
    # ==========================================================================
    print(f'\n\n{"="*80}')
    print('SUMMARY — Can you rely on the previous 2 weeks\' top combo?')
    print(f'{"="*80}')

    n = len(df_comp)
    same_pct = df_comp['same_combo'].mean() * 100
    prev_positive = (df_comp['prev_winner_pnl'] > 0).mean() * 100
    prev_avg = df_comp['prev_winner_pnl'].mean()
    prev_total = df_comp['prev_winner_pnl'].sum()
    curr_avg = df_comp['curr_winner_pnl'].mean()
    curr_total = df_comp['curr_winner_pnl'].sum()
    gap_avg = df_comp['gap'].mean()

    # How often prev winner beats at least half of all combos this period
    beat_half_count = 0
    for _, row in df_comp.iterrows():
        idx = row['period']
        pr = period_results[idx]
        if pr is None:
            continue
        all_pnls = [v['pnl'] for v in pr['all_results'].values() if v['trades'] >= 1]
        if all_pnls:
            median_pnl = np.median(all_pnls)
            if row['prev_winner_pnl'] > median_pnl:
                beat_half_count += 1
    beat_median_pct = beat_half_count / n * 100

    print(f'\n  Periods analyzed:            {n}')
    print(f'  Same combo persists:         {same_pct:.0f}%')
    print(f'  Prev winner profitable:      {prev_positive:.0f}% of periods')
    print(f'  Prev winner beats median:    {beat_median_pct:.0f}% of periods')
    print(f'')
    print(f'  Prev winner avg PnL:         ${prev_avg:+.1f} / period')
    print(f'  Prev winner total PnL:       ${prev_total:+.1f}')
    print(f'  Curr winner avg PnL:         ${curr_avg:+.1f} / period')
    print(f'  Curr winner total PnL:       ${curr_total:+.1f}')
    print(f'  Avg gap (curr - prev):       ${gap_avg:+.1f}')

    print(f'\n  Verdict breakdown:')
    for v in ['SAME', 'GOOD', 'OK', 'WEAK', 'FAILED']:
        cnt = (df_comp['verdict'] == v).sum()
        if cnt > 0:
            print(f'    {v:<8} {cnt:>3} ({cnt/n*100:.0f}%)')

    # ==========================================================================
    # 7. Rolling cumulative PnL comparison
    # ==========================================================================
    print(f'\n\n{"="*80}')
    print('CUMULATIVE PnL — Prev Winner Strategy vs Always-Current-Best')
    print(f'{"="*80}')

    cum_prev = df_comp['prev_winner_pnl'].cumsum().values
    cum_curr = df_comp['curr_winner_pnl'].cumsum().values

    print(f'\n  {"Period":<8} {"Dates":<27} {"Prev_cum$":<12} {"Curr_cum$":<12}')
    print('  ' + '-'*60)
    for j in range(len(df_comp)):
        row = df_comp.iloc[j]
        dates = f"{row['start'].strftime('%Y-%m-%d')} → {row['end'].strftime('%Y-%m-%d')}"
        print(f'  P{int(row["period"]):<5} {dates:<27} '
              f'${cum_prev[j]:>+9.1f}   ${cum_curr[j]:>+9.1f}')

    print(f'\n  FINAL: Prev-winner strategy = ${cum_prev[-1]:+.1f}  |  '
          f'Always-best = ${cum_curr[-1]:+.1f}  |  '
          f'Efficiency = {cum_prev[-1] / (cum_curr[-1] + 1e-10) * 100:.0f}%')

    # Save
    output_path = 'data/processed/concept_drift_2w_comparison.csv'
    df_comp.to_csv(output_path, index=False)
    print(f'\n  Saved detail to {output_path}')

print(f'\n{"="*80}')
print('DONE')
print(f'{"="*80}')
