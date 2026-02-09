"""
Calculate HEDGE PAIR win rate (not individual position win rate)
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd

mt5.initialize()
from_date = datetime.now(timezone.utc) - timedelta(days=30)
to_date = datetime.now(timezone.utc)
MAGIC_LONG = 123456
MAGIC_SHORT = 654321

deals = mt5.history_deals_get(from_date, to_date)
df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
hedge_deals = df[(df['magic'] == MAGIC_LONG) | (df['magic'] == MAGIC_SHORT)].copy()
hedge_deals['time'] = pd.to_datetime(hedge_deals['time'], unit='s')
exits = hedge_deals[hedge_deals['entry'] == 1]

long_exits = exits[exits['magic'] == MAGIC_LONG].copy().reset_index(drop=True)
short_exits = exits[exits['magic'] == MAGIC_SHORT].copy().reset_index(drop=True)

print(f'Long exits: {len(long_exits)}')
print(f'Short exits: {len(short_exits)}')

# Convert time to numeric for fast comparison
long_exits['time_sec'] = long_exits['time'].astype('int64') // 10**9
short_exits['time_sec'] = short_exits['time'].astype('int64') // 10**9

# Match pairs by time (within 60 seconds) - optimized
hedge_pairs = []
short_exits_list = short_exits.to_dict('records')
used_indices = set()

print("Matching hedge pairs...")
for i, long_row in long_exits.iterrows():
    if i % 500 == 0:
        print(f"  Processing {i}/{len(long_exits)}...")
    
    long_time_sec = long_row['time_sec']
    long_profit = long_row['profit']
    long_time = long_row['time']
    
    for j, short_row in enumerate(short_exits_list):
        if j in used_indices:
            continue
        
        time_diff = abs(short_row['time_sec'] - long_time_sec)
        if time_diff <= 60:
            short_profit = short_row['profit']
            pair_profit = long_profit + short_profit
            
            hedge_pairs.append({
                'time': long_time,
                'long_profit': long_profit,
                'short_profit': short_profit,
                'pair_profit': pair_profit,
                'result': 'WIN' if pair_profit > 0 else ('LOSS' if pair_profit < 0 else 'BE')
            })
            used_indices.add(j)
            break

print(f'\n=== HEDGE PAIR ANALYSIS ===')
print(f'Matched pairs: {len(hedge_pairs)}')

if len(hedge_pairs) > 0:
    pair_df = pd.DataFrame(hedge_pairs)
    wins = len(pair_df[pair_df['result'] == 'WIN'])
    losses = len(pair_df[pair_df['result'] == 'LOSS'])
    be = len(pair_df[pair_df['result'] == 'BE'])
    
    print(f'Winning pairs (net profit > 0): {wins}')
    print(f'Losing pairs (net profit < 0): {losses}')
    print(f'Breakeven pairs: {be}')
    print(f'\n*** HEDGE WIN RATE: {wins/len(pair_df)*100:.1f}% ***')
    
    # Average profits
    avg_pair = pair_df['pair_profit'].mean()
    avg_win = pair_df[pair_df['result'] == 'WIN']['pair_profit'].mean() if wins > 0 else 0
    avg_loss = pair_df[pair_df['result'] == 'LOSS']['pair_profit'].mean() if losses > 0 else 0
    
    print(f'\nAverage profit per pair: ${avg_pair:.2f}')
    print(f'Average winning pair: ${avg_win:.2f}')
    print(f'Average losing pair: ${avg_loss:.2f}')
    
    # Profit factor
    total_wins = pair_df[pair_df['result'] == 'WIN']['pair_profit'].sum()
    total_losses = abs(pair_df[pair_df['result'] == 'LOSS']['pair_profit'].sum())
    pf = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f'Profit Factor: {pf:.2f}')
    
    # Daily breakdown
    pair_df['date'] = pair_df['time'].dt.date
    print(f'\n=== DAILY HEDGE PAIR BREAKDOWN ===')
    for date in sorted(pair_df['date'].unique()):
        day_pairs = pair_df[pair_df['date'] == date]
        day_wins = len(day_pairs[day_pairs['result'] == 'WIN'])
        day_losses = len(day_pairs[day_pairs['result'] == 'LOSS'])
        day_wr = day_wins / len(day_pairs) * 100 if len(day_pairs) > 0 else 0
        day_pnl = day_pairs['pair_profit'].sum()
        print(f'{date} | Pairs: {len(day_pairs):>4} | Wins: {day_wins:>4} | Losses: {day_losses:>4} | WR: {day_wr:>5.1f}% | P&L: ${day_pnl:>8.2f}')
    
    # Sample pairs
    print(f'\n=== SAMPLE HEDGE PAIRS (last 20) ===')
    print(pair_df.tail(20).to_string(index=False))

mt5.shutdown()
