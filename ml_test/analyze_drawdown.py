"""
Analyze Account 2's worst drawdown / losing streaks
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialize failed")
    exit()

from_date = datetime.now(timezone.utc) - timedelta(days=30)
to_date = datetime.now(timezone.utc)
MAGIC_LONG = 123456
MAGIC_SHORT = 654321

deals = mt5.history_deals_get(from_date, to_date)
if deals is None or len(deals) == 0:
    print("No deals found")
    mt5.shutdown()
    exit()

df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
hedge_deals = df[(df['magic'] == MAGIC_LONG) | (df['magic'] == MAGIC_SHORT)].copy()
hedge_deals['time'] = pd.to_datetime(hedge_deals['time'], unit='s', utc=True)
exits = hedge_deals[hedge_deals['entry'] == 1].copy()

print("="*80)
print("ACCOUNT 2 - DRAWDOWN & LOSING STREAK ANALYSIS")
print("="*80)

# Calculate running P&L
exits = exits.sort_values('time').reset_index(drop=True)
exits['cumulative_pnl'] = exits['profit'].cumsum()

# Calculate drawdown
exits['peak'] = exits['cumulative_pnl'].cummax()
exits['drawdown'] = exits['cumulative_pnl'] - exits['peak']

# Find max drawdown
max_dd_idx = exits['drawdown'].idxmin()
max_dd = exits.loc[max_dd_idx, 'drawdown']
max_dd_time = exits.loc[max_dd_idx, 'time']
max_dd_peak = exits.loc[max_dd_idx, 'peak']

print(f"\nTotal Exits: {len(exits)}")
print(f"Final P&L: ${exits['cumulative_pnl'].iloc[-1]:.2f}")

print(f"\n--- MAXIMUM DRAWDOWN ---")
print(f"Max Drawdown: ${max_dd:.2f}")
print(f"Occurred at: {max_dd_time}")
print(f"Peak before drawdown: ${max_dd_peak:.2f}")

# Find the start of this drawdown
# Look backwards from max_dd_idx to find when peak was set
peak_value = exits.loc[max_dd_idx, 'peak']
peak_idx = exits[exits['cumulative_pnl'] == peak_value].index[0]
peak_time = exits.loc[peak_idx, 'time']

print(f"Drawdown started at: {peak_time}")
print(f"Drawdown duration: {max_dd_time - peak_time}")

# Analyze losing streaks (consecutive losses)
print(f"\n--- LOSING STREAKS ---")
exits['is_loss'] = exits['profit'] < 0

# Find consecutive losing streaks
streaks = []
current_streak = 0
current_streak_loss = 0
current_streak_start = None

for idx, row in exits.iterrows():
    if row['is_loss']:
        if current_streak == 0:
            current_streak_start = row['time']
        current_streak += 1
        current_streak_loss += row['profit']
    else:
        if current_streak > 0:
            streaks.append({
                'start': current_streak_start,
                'length': current_streak,
                'total_loss': current_streak_loss,
                'end': row['time']
            })
        current_streak = 0
        current_streak_loss = 0
        current_streak_start = None

# Don't forget last streak if it ends with losses
if current_streak > 0:
    streaks.append({
        'start': current_streak_start,
        'length': current_streak,
        'total_loss': current_streak_loss,
        'end': exits.iloc[-1]['time']
    })

streak_df = pd.DataFrame(streaks)

if len(streak_df) > 0:
    # Worst losing streak by total loss
    worst_streak = streak_df.loc[streak_df['total_loss'].idxmin()]
    print(f"\nWorst Losing Streak (by total loss):")
    print(f"  Start: {worst_streak['start']}")
    print(f"  End: {worst_streak['end']}")
    print(f"  Length: {worst_streak['length']} consecutive losses")
    print(f"  Total Loss: ${worst_streak['total_loss']:.2f}")
    
    # Longest losing streak
    longest_streak = streak_df.loc[streak_df['length'].idxmax()]
    print(f"\nLongest Losing Streak (by count):")
    print(f"  Start: {longest_streak['start']}")
    print(f"  End: {longest_streak['end']}")
    print(f"  Length: {longest_streak['length']} consecutive losses")
    print(f"  Total Loss: ${longest_streak['total_loss']:.2f}")
    
    # Top 10 worst streaks
    print(f"\n--- TOP 10 WORST LOSING STREAKS ---")
    top_streaks = streak_df.nsmallest(10, 'total_loss')
    print(f"{'Start Time':<25} {'Length':<10} {'Total Loss':<12}")
    print("-"*50)
    for _, s in top_streaks.iterrows():
        print(f"{str(s['start']):<25} {s['length']:<10} ${s['total_loss']:>10.2f}")

# Hourly analysis of losses
print(f"\n--- LOSSES BY HOUR ---")
exits['hour'] = exits['time'].dt.hour
hourly_loss = exits[exits['is_loss']].groupby('hour').agg({
    'profit': ['sum', 'count', 'mean']
}).reset_index()
hourly_loss.columns = ['hour', 'total_loss', 'count', 'avg_loss']
hourly_loss = hourly_loss.sort_values('total_loss')

print(f"{'Hour':<8} {'Total Loss':<12} {'Count':<10} {'Avg Loss':<12}")
print("-"*45)
for _, row in hourly_loss.head(10).iterrows():
    print(f"{int(row['hour']):>4}     ${row['total_loss']:>10.2f}   {int(row['count']):>6}     ${row['avg_loss']:>10.2f}")

# Rolling drawdown periods
print(f"\n--- EQUITY CURVE SUMMARY ---")
print(f"Starting P&L: $0.00")
print(f"Peak P&L: ${exits['peak'].max():.2f}")
print(f"Final P&L: ${exits['cumulative_pnl'].iloc[-1]:.2f}")
print(f"Max Drawdown: ${max_dd:.2f} ({max_dd/exits['peak'].max()*100:.1f}% of peak)")

# Show equity curve around max drawdown
print(f"\n--- TRADES AROUND MAX DRAWDOWN ---")
dd_start = max(0, peak_idx - 5)
dd_end = min(len(exits), max_dd_idx + 10)
dd_trades = exits.iloc[dd_start:dd_end][['time', 'profit', 'cumulative_pnl', 'peak', 'drawdown', 'magic']]
dd_trades['side'] = dd_trades['magic'].apply(lambda x: 'LONG' if x == MAGIC_LONG else 'SHORT')
print(dd_trades.to_string(index=False))

mt5.shutdown()
