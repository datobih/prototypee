"""
vbt_session_analysis.py

Breaks down the performance of the best strategy found:
  - SMA Fast: 40 / Slow: 50
  - RSI Long: > 60, Short: < 40
  - Continuation logic: SMA cross + RSI momentum confirmation

Tests each Forex trading session separately:
  - Asian session:   00:00 - 07:59 UTC
  - London session:  08:00 - 12:59 UTC
  - New York session:13:00 - 20:59 UTC
  - London/NY Overlap: 13:00 - 16:59 UTC (subset of NY)
"""
import pandas as pd
import numpy as np
import vectorbt as vbt

# --------------------------------------------------------------------------
# 1. Load Full Year Data (15-min bars, same as what the user ran)
# --------------------------------------------------------------------------
print("Loading 15-minute price data...")
df = pd.read_csv(
    'data/raw/XAUUSD30.csv',
    sep='\t',
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
price_full = df.set_index('Datetime').dropna(subset=['Close']).loc['2022-01-01':'2022-12-31']['Close']
print(f"Loaded {len(price_full)} bars.")

# --------------------------------------------------------------------------
# 2. Define Sessions (UTC hours)
# --------------------------------------------------------------------------
sessions = {
    'Asian':          (0,  8),     # 00:00 - 07:59 UTC
    'London':         (8,  13),    # 08:00 - 12:59 UTC
    'NY_Full':        (13, 21),    # 13:00 - 20:59 UTC
    'LDN_NY_Overlap': (13, 17),    # 13:00 - 16:59 UTC (hottest hours)
    'All_Sessions':   (0,  24),    # baseline: no filter
}

# --------------------------------------------------------------------------
# 3. Signal Generator Function
# --------------------------------------------------------------------------
def run_strategy(price, freq='15min'):
    """Compute signals and run portfolio for Fast40/Slow50/RSI50 strategy."""
    if len(price) < 200:
        return None

    sma_fast = vbt.MA.run(price, window=40).ma
    sma_slow = vbt.MA.run(price, window=50).ma
    rsi      = vbt.RSI.run(price, window=14).rsi

    ma_f_prev = sma_fast.shift(1)
    ma_s_prev = sma_slow.shift(1)

    long_entries  = (ma_f_prev <= ma_s_prev) & (sma_fast > sma_slow) & (rsi > 60)
    short_entries = (ma_f_prev >= ma_s_prev) & (sma_fast < sma_slow) & (rsi < 40)

    portfolio = vbt.Portfolio.from_signals(
        price,
        entries=long_entries,
        short_entries=short_entries,
        upon_opposite_entry='close',
        size=1.0,
        size_type='amount',
        init_cash=100000,
        fees=0.0001,
        sl_stop=0.002,
        tp_stop=0.005,
        freq=freq
    )
    return portfolio

# --------------------------------------------------------------------------
# 4. Run Per Session
# --------------------------------------------------------------------------
print("\n{'='*60}")
print("SESSION PERFORMANCE BREAKDOWN: Fast40_Slow50_RSI_L60_S40")
print("=" * 60)

results = []
for session_name, (h_start, h_end) in sessions.items():
    if h_end == 24:
        mask = price_full.index.hour >= 0  # all hours
    else:
        mask = (price_full.index.hour >= h_start) & (price_full.index.hour < h_end)

    price_session = price_full[mask]

    if len(price_session) < 200:
        print(f"\n{session_name}: Not enough bars ({len(price_session)}), skipping.")
        continue

    pf = run_strategy(price_session, freq='15min')
    if pf is None:
        continue

    total_trades = pf.trades.count()
    win_rate     = pf.trades.win_rate() * 100
    total_pnl    = pf.total_profit()
    pf_ratio     = pf.trades.profit_factor()
    avg_win      = pf.trades.records_readable['PnL'][pf.trades.records_readable['PnL'] > 0].mean()
    avg_loss     = pf.trades.records_readable['PnL'][pf.trades.records_readable['PnL'] <= 0].mean()

    results.append({
        'Session':       session_name,
        'Hours (UTC)':   f"{h_start:02d}:00-{h_end if h_end<24 else 24:02d}:00",
        'Bars':          len(price_session),
        'Trades':        total_trades,
        'Win_Rate_%':    round(win_rate, 1),
        'Total_PnL_$':   round(total_pnl, 2),
        'Profit_Factor': round(pf_ratio, 3),
        'Avg_Win_$':     round(avg_win, 2) if not np.isnan(avg_win) else 0,
        'Avg_Loss_$':    round(avg_loss, 2) if not np.isnan(avg_loss) else 0,
    })

    status = "[PROFIT]" if total_pnl > 0 else "[LOSS]"
    print(f"\n{session_name} ({h_start:02d}:00-{h_end if h_end<24 else 24:02d}:00 UTC) {status}")
    print(f"  Bars: {len(price_session):,}  |  Trades: {total_trades}  |  Win Rate: {win_rate:.1f}%")
    print(f"  Total PnL: ${total_pnl:.2f}  |  Profit Factor: {pf_ratio:.3f}")
    print(f"  Avg Win: ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")

# --------------------------------------------------------------------------
# 5. Summary Table
# --------------------------------------------------------------------------
summary = pd.DataFrame(results).sort_values('Total_PnL_$', ascending=False)
print("\n\n" + "=" * 60)
print("SUMMARY (sorted by P&L):")
print("=" * 60)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(summary.to_string(index=False))

output_path = 'data/processed/vbt_session_analysis.csv'
summary.to_csv(output_path, index=False)
print(f"\nSession breakdown saved to: {output_path}")
