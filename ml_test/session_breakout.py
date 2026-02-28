"""
session_breakout.py

Session Open Breakout Strategy for XAUUSD 1-min.

Logic:
  - At London open (8am GMT) and NY open (1:30pm GMT), measure the
    prior 30-min range (pre-session consolidation).
  - If ATR(14) at open < 30th percentile of last 20 session opens → compressed.
  - Place OCO bracket: buy stop above range high + offset, sell stop below range low - offset.
  - TP = 2x range from entry, SL = range midpoint.
  - One side triggers → cancel other.
  - Track stats for compressed vs all setups, London vs NY.

Assumptions:
  - Data timestamps are in MT4 server time (GMT+2, no DST adjustment).
  - London open = 10:00 server time  (8:00 GMT)
  - NY open     = 15:30 server time  (13:30 GMT)
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = 'data/raw/XAUUSD1.csv'

# Session opens in SERVER TIME (assumed GMT+2)
LONDON_OPEN_HOUR   = 10    # 8:00 GMT → 10:00 server
LONDON_OPEN_MINUTE = 0
NY_OPEN_HOUR       = 15    # 13:30 GMT → 15:30 server
NY_OPEN_MINUTE     = 30

PRE_RANGE_MINUTES  = 30    # look back 30 min for consolidation range
ATR_PERIOD         = 14    # ATR lookback (in session-opens, not bars)
COMPRESSION_PCT    = 30    # ATR < Nth percentile = compressed
LOOKBACK_SESSIONS  = 20    # compare ATR against last N session opens

ENTRY_OFFSET_MULT  = 0.10  # entry offset = 10% of range above/below range edges
TP_MULT            = 2.0   # TP = 2x range from entry
SL_TO_MIDPOINT     = True  # SL = range midpoint

MAX_TRADE_BARS     = 120   # max bars (2 hours) before force-close

# =============================================================================
# LOAD DATA
# =============================================================================
print('=' * 80)
print('SESSION OPEN BREAKOUT STRATEGY — XAUUSD 1-MIN')
print('=' * 80)

df = pd.read_csv(DATA_PATH, sep='\t',
                 names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                        'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'TickVol']].copy()
df.sort_index(inplace=True)

print(f'Loaded {len(df):,} bars  |  {df.index[0]} → {df.index[-1]}')

# =============================================================================
# COMPUTE BAR-LEVEL ATR (for compression check)
# =============================================================================
df['TR'] = np.maximum(
    df['High'] - df['Low'],
    np.maximum(
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    )
)
df['ATR14'] = df['TR'].rolling(14).mean()

# =============================================================================
# IDENTIFY SESSION OPENS
# =============================================================================
def find_session_opens(df, hour, minute):
    """Find all bars that match the session open time."""
    mask = (df.index.hour == hour) & (df.index.minute == minute)
    return df.index[mask]

london_opens = find_session_opens(df, LONDON_OPEN_HOUR, LONDON_OPEN_MINUTE)
ny_opens     = find_session_opens(df, NY_OPEN_HOUR, NY_OPEN_MINUTE)

print(f'\nLondon opens found: {len(london_opens)}')
print(f'NY opens found:     {len(ny_opens)}')

# =============================================================================
# MEASURE PRE-SESSION RANGE
# =============================================================================
def get_pre_range(df, session_time, lookback_minutes=30):
    """Get high/low of the N minutes before session open."""
    loc = df.index.get_loc(session_time)
    if isinstance(loc, slice):
        loc = loc.start
    
    start = max(0, loc - lookback_minutes)
    end = loc  # exclusive — up to but not including the open bar
    
    if end - start < 10:  # need at least 10 bars
        return None, None, None
    
    window = df.iloc[start:end]
    range_high = window['High'].max()
    range_low  = window['Low'].min()
    range_size = range_high - range_low
    
    return range_high, range_low, range_size

# =============================================================================
# SIMULATE OCO TRADE
# =============================================================================
def simulate_oco(df, entry_bar_loc, buy_entry, sell_entry,
                 buy_tp, buy_sl, sell_tp, sell_sl, max_bars):
    """
    Simulate OCO bracket trade.
    
    Wait for price to trigger either buy stop or sell stop.
    Once one triggers, cancel the other and run to TP/SL/expiry.
    
    Returns dict with trade details or None if neither triggered.
    """
    total_bars = len(df)
    triggered_side = None
    trigger_bar = None
    entry_price = None
    tp_price = None
    sl_price = None
    
    # Phase 1: Wait for one side to trigger
    for i in range(entry_bar_loc, min(entry_bar_loc + max_bars, total_bars)):
        bar_high = df['High'].iloc[i]
        bar_low  = df['Low'].iloc[i]
        
        buy_hit  = bar_high >= buy_entry
        sell_hit = bar_low  <= sell_entry
        
        if buy_hit and sell_hit:
            # Both hit same bar — use open to determine which triggered first
            bar_open = df['Open'].iloc[i]
            if abs(bar_open - buy_entry) <= abs(bar_open - sell_entry):
                triggered_side = 'LONG'
            else:
                triggered_side = 'SHORT'
        elif buy_hit:
            triggered_side = 'LONG'
        elif sell_hit:
            triggered_side = 'SHORT'
        
        if triggered_side:
            trigger_bar = i
            if triggered_side == 'LONG':
                entry_price = buy_entry
                tp_price = buy_tp
                sl_price = buy_sl
            else:
                entry_price = sell_entry
                tp_price = sell_tp
                sl_price = sell_sl
            break
    
    if not triggered_side:
        return None  # neither side triggered within max_bars
    
    # Phase 2: Run triggered side to TP/SL/expiry
    remaining_bars = max_bars - (trigger_bar - entry_bar_loc)
    
    for j in range(trigger_bar + 1, min(trigger_bar + remaining_bars + 1, total_bars)):
        bar_high = df['High'].iloc[j]
        bar_low  = df['Low'].iloc[j]
        
        if triggered_side == 'LONG':
            tp_hit = bar_high >= tp_price
            sl_hit = bar_low  <= sl_price
        else:  # SHORT
            tp_hit = bar_low  <= tp_price
            sl_hit = bar_high >= sl_price
        
        if tp_hit and sl_hit:
            # Both hit — assume SL first (conservative)
            if triggered_side == 'LONG':
                pnl = sl_price - entry_price
            else:
                pnl = entry_price - sl_price
            return {
                'side': triggered_side,
                'entry': entry_price,
                'exit': sl_price,
                'pnl': pnl,
                'result': 'SL',
                'bars_held': j - trigger_bar,
                'trigger_bar': trigger_bar
            }
        
        if tp_hit:
            if triggered_side == 'LONG':
                pnl = tp_price - entry_price
            else:
                pnl = entry_price - tp_price
            return {
                'side': triggered_side,
                'entry': entry_price,
                'exit': tp_price,
                'pnl': pnl,
                'result': 'TP',
                'bars_held': j - trigger_bar,
                'trigger_bar': trigger_bar
            }
        
        if sl_hit:
            if triggered_side == 'LONG':
                pnl = sl_price - entry_price
            else:
                pnl = entry_price - sl_price
            return {
                'side': triggered_side,
                'entry': entry_price,
                'exit': sl_price,
                'pnl': pnl,
                'result': 'SL',
                'bars_held': j - trigger_bar,
                'trigger_bar': trigger_bar
            }
    
    # Expired — close at last bar's close
    last_bar = min(trigger_bar + remaining_bars, total_bars - 1)
    exit_price = df['Close'].iloc[last_bar]
    if triggered_side == 'LONG':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    
    return {
        'side': triggered_side,
        'entry': entry_price,
        'exit': exit_price,
        'pnl': pnl,
        'result': 'EXPIRED',
        'bars_held': last_bar - trigger_bar,
        'trigger_bar': trigger_bar
    }

# =============================================================================
# RUN BACKTEST
# =============================================================================
print('\n' + '=' * 80)
print('BACKTESTING')
print('=' * 80)

all_sessions = []

for session_time in sorted(london_opens.tolist() + ny_opens.tolist()):
    session_type = 'LONDON' if (session_time.hour == LONDON_OPEN_HOUR and 
                                session_time.minute == LONDON_OPEN_MINUTE) else 'NY'
    
    range_high, range_low, range_size = get_pre_range(df, session_time, PRE_RANGE_MINUTES)
    
    if range_high is None or range_size < 0.50:  # skip if range too tiny (<$0.50)
        continue
    
    # Get ATR at session open
    loc = df.index.get_loc(session_time)
    if isinstance(loc, slice):
        loc = loc.start
    
    atr_at_open = df['ATR14'].iloc[loc]
    if np.isnan(atr_at_open):
        continue
    
    all_sessions.append({
        'datetime':     session_time,
        'session':      session_type,
        'range_high':   range_high,
        'range_low':    range_low,
        'range_size':   range_size,
        'atr_at_open':  atr_at_open,
        'bar_loc':      loc,
    })

sessions_df = pd.DataFrame(all_sessions)
print(f'\nValid session opens: {len(sessions_df)}')
print(f'  London: {(sessions_df["session"] == "LONDON").sum()}')
print(f'  NY:     {(sessions_df["session"] == "NY").sum()}')

# Compute rolling ATR percentile for compression filter
sessions_df['atr_pct_rank'] = sessions_df['atr_at_open'].rolling(
    LOOKBACK_SESSIONS, min_periods=10
).apply(lambda x: (x.iloc[-1] <= x).mean() * 100, raw=False)

sessions_df['is_compressed'] = sessions_df['atr_pct_rank'] <= COMPRESSION_PCT
sessions_df = sessions_df.dropna(subset=['atr_pct_rank'])

print(f'Sessions with ATR rank: {len(sessions_df)}')
print(f'Compressed (ATR < {COMPRESSION_PCT}th pct): {sessions_df["is_compressed"].sum()} '
      f'({sessions_df["is_compressed"].mean()*100:.1f}%)')

# =============================================================================
# SIMULATE TRADES
# =============================================================================
print('\nSimulating OCO trades...')

trades = []

for _, sess in sessions_df.iterrows():
    rh = sess['range_high']
    rl = sess['range_low']
    rs = sess['range_size']
    mid = (rh + rl) / 2.0
    
    offset = rs * ENTRY_OFFSET_MULT
    
    buy_entry  = rh + offset
    sell_entry = rl - offset
    
    buy_tp  = buy_entry + rs * TP_MULT
    buy_sl  = mid  # SL at range midpoint
    sell_tp = sell_entry - rs * TP_MULT
    sell_sl = mid  # SL at range midpoint
    
    result = simulate_oco(df, int(sess['bar_loc']), buy_entry, sell_entry,
                          buy_tp, buy_sl, sell_tp, sell_sl, MAX_TRADE_BARS)
    
    if result is None:
        continue
    
    result['datetime']      = sess['datetime']
    result['session']       = sess['session']
    result['range_size']    = rs
    result['atr_at_open']   = sess['atr_at_open']
    result['atr_pct_rank']  = sess['atr_pct_rank']
    result['is_compressed'] = sess['is_compressed']
    result['rr_ratio']      = result['pnl'] / (buy_entry - mid) if (buy_entry - mid) > 0 else 0
    
    trades.append(result)

trades_df = pd.DataFrame(trades)
print(f'Total trades executed: {len(trades_df)}')

# =============================================================================
# RESULTS
# =============================================================================
def print_stats(label, tdf):
    if len(tdf) == 0:
        print(f'\n  {label}: No trades')
        return
    
    total_pnl   = tdf['pnl'].sum()
    avg_pnl     = tdf['pnl'].mean()
    win_rate    = (tdf['pnl'] > 0).mean() * 100
    tp_rate     = (tdf['result'] == 'TP').mean() * 100
    sl_rate     = (tdf['result'] == 'SL').mean() * 100
    exp_rate    = (tdf['result'] == 'EXPIRED').mean() * 100
    avg_winner  = tdf.loc[tdf['pnl'] > 0, 'pnl'].mean() if (tdf['pnl'] > 0).any() else 0
    avg_loser   = tdf.loc[tdf['pnl'] < 0, 'pnl'].mean() if (tdf['pnl'] < 0).any() else 0
    profit_factor = (tdf.loc[tdf['pnl'] > 0, 'pnl'].sum() / 
                     abs(tdf.loc[tdf['pnl'] < 0, 'pnl'].sum())) if (tdf['pnl'] < 0).any() else np.inf
    avg_bars    = tdf['bars_held'].mean()
    long_pct    = (tdf['side'] == 'LONG').mean() * 100
    
    # Max drawdown
    cum_pnl = tdf['pnl'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    
    # Sharpe (daily approximation: ~2 trades/day)
    daily_pnl = tdf.set_index('datetime')['pnl'].resample('D').sum().dropna()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
    
    print(f'\n  {label}')
    print(f'  {"─" * 50}')
    print(f'  Trades:         {len(tdf):>8}')
    print(f'  Total PnL:      ${total_pnl:>+10.2f}')
    print(f'  Avg PnL/trade:  ${avg_pnl:>+10.2f}')
    print(f'  Win rate:       {win_rate:>8.1f}%')
    print(f'  TP / SL / EXP:  {tp_rate:.0f}% / {sl_rate:.0f}% / {exp_rate:.0f}%')
    print(f'  Avg winner:     ${avg_winner:>+10.2f}')
    print(f'  Avg loser:      ${avg_loser:>+10.2f}')
    print(f'  Profit factor:  {profit_factor:>8.2f}')
    print(f'  Max drawdown:   ${max_dd:>+10.2f}')
    print(f'  Sharpe (ann):   {sharpe:>8.2f}')
    print(f'  Avg bars held:  {avg_bars:>8.1f} ({avg_bars:.0f} min)')
    print(f'  Long / Short:   {long_pct:.0f}% / {100-long_pct:.0f}%')
    print(f'  Avg range:      ${tdf["range_size"].mean():.2f}')

print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)

# All trades
print_stats('ALL TRADES', trades_df)

# Compressed only (the actual strategy)
compressed = trades_df[trades_df['is_compressed'] == True]
not_compressed = trades_df[trades_df['is_compressed'] == False]
print_stats('COMPRESSED ATR ONLY (THE STRATEGY)', compressed)
print_stats('NOT COMPRESSED (for comparison)', not_compressed)

# By session
london_trades = trades_df[trades_df['session'] == 'LONDON']
ny_trades     = trades_df[trades_df['session'] == 'NY']
print_stats('LONDON OPEN', london_trades)
print_stats('NY OPEN', ny_trades)

# Compressed by session
london_comp = compressed[compressed['session'] == 'LONDON']
ny_comp     = compressed[compressed['session'] == 'NY']
print_stats('LONDON + COMPRESSED', london_comp)
print_stats('NY + COMPRESSED', ny_comp)

# =============================================================================
# RANGE SIZE ANALYSIS
# =============================================================================
print('\n' + '=' * 80)
print('PRE-SESSION RANGE SIZE DISTRIBUTION')
print('=' * 80)

for session_type in ['LONDON', 'NY']:
    subset = sessions_df[sessions_df['session'] == session_type]
    rs = subset['range_size']
    print(f'\n  {session_type}:')
    print(f'    Mean:   ${rs.mean():.2f}')
    print(f'    Median: ${rs.median():.2f}')
    print(f'    25th:   ${rs.quantile(0.25):.2f}')
    print(f'    75th:   ${rs.quantile(0.75):.2f}')
    print(f'    90th:   ${rs.quantile(0.90):.2f}')

# =============================================================================
# ATR COMPRESSION EFFECTIVENESS
# =============================================================================
print('\n' + '=' * 80)
print('COMPRESSION FILTER EFFECTIVENESS')
print('=' * 80)

if len(compressed) > 0 and len(not_compressed) > 0:
    print(f'\n  Compressed avg PnL:     ${compressed["pnl"].mean():+.2f}  '
          f'(win rate: {(compressed["pnl"]>0).mean()*100:.1f}%)')
    print(f'  Not-compressed avg PnL: ${not_compressed["pnl"].mean():+.2f}  '
          f'(win rate: {(not_compressed["pnl"]>0).mean()*100:.1f}%)')
    
    edge = compressed['pnl'].mean() - not_compressed['pnl'].mean()
    print(f'\n  Compression edge:       ${edge:+.2f} per trade')

# =============================================================================
# MONTHLY BREAKDOWN
# =============================================================================
print('\n' + '=' * 80)
print('MONTHLY PnL (COMPRESSED ONLY)')
print('=' * 80)

if len(compressed) > 0:
    compressed_copy = compressed.copy()
    compressed_copy['month'] = compressed_copy['datetime'].dt.to_period('M')
    monthly = compressed_copy.groupby('month').agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100)
    )
    
    for period, row in monthly.iterrows():
        bar = '█' * int(max(0, row['total_pnl'] / 5))
        neg_bar = '░' * int(max(0, -row['total_pnl'] / 5))
        print(f'  {period} | {row["trades"]:3.0f} trades | '
              f'${row["total_pnl"]:>+8.2f} | '
              f'win {row["win_rate"]:4.0f}% | {bar}{neg_bar}')

# Save trade log
trades_df.to_csv('data/processed/session_breakout_trades.csv', index=False)
print(f'\nSaved trade log to data/processed/session_breakout_trades.csv')

print('\n' + '=' * 80)
print('DONE')
print('=' * 80)
