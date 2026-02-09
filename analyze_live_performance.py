"""
Analyze actual live trading performance from MT5 history
Accounts for spread, slippage, and commissions
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialize failed")
    exit()

# Get deals history for last 30 days
from_date = datetime.now(timezone.utc) - timedelta(days=30)
to_date = datetime.now(timezone.utc)

# Get all deals for hedge trader magic numbers
MAGIC_LONG = 123456
MAGIC_SHORT = 654321

deals = mt5.history_deals_get(from_date, to_date)
if deals is None or len(deals) == 0:
    print("No deals found")
    mt5.shutdown()
    exit()

# Convert to DataFrame
df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())

# Filter for our hedge trades
hedge_deals = df[(df['magic'] == MAGIC_LONG) | (df['magic'] == MAGIC_SHORT)].copy()

if len(hedge_deals) == 0:
    print("No hedge trades found with magic numbers", MAGIC_LONG, MAGIC_SHORT)
    print("\nAll unique magic numbers in history:")
    print(df['magic'].unique())
    mt5.shutdown()
    exit()

print("="*80)
print("LIVE HEDGE TRADING PERFORMANCE ANALYSIS")
print("="*80)

# Convert time to readable format
hedge_deals['time'] = pd.to_datetime(hedge_deals['time'], unit='s')

print(f"\nTotal deals: {len(hedge_deals)}")
print(f"Date range: {hedge_deals['time'].min()} to {hedge_deals['time'].max()}")

# Separate entries (DEAL_ENTRY_IN = 0) and exits (DEAL_ENTRY_OUT = 1)
entries = hedge_deals[hedge_deals['entry'] == 0]
exits = hedge_deals[hedge_deals['entry'] == 1]

print(f"\nEntries: {len(entries)}")
print(f"Exits: {len(exits)}")

# ============================================================================
# SPREAD ANALYSIS - Calculate spread from LONG/SHORT entry pairs
# ============================================================================
print(f"\n--- SPREAD ANALYSIS ---")

# Get LONG and SHORT entries
long_entries = entries[entries['magic'] == MAGIC_LONG].copy()
short_entries = entries[entries['magic'] == MAGIC_SHORT].copy()

# Match pairs by time (within 2 seconds of each other)
spreads = []
for _, long_row in long_entries.iterrows():
    long_time = long_row['time']
    long_price = long_row['price']  # ASK price
    
    # Find matching SHORT entry within 2 seconds
    time_diff = abs((short_entries['time'] - long_time).dt.total_seconds())
    matching = short_entries[time_diff <= 2]
    
    if len(matching) > 0:
        short_price = matching.iloc[0]['price']  # BID price
        spread = long_price - short_price  # ASK - BID = spread
        spreads.append({
            'time': long_time,
            'long_price': long_price,
            'short_price': short_price,
            'spread': spread
        })

if len(spreads) > 0:
    spread_df = pd.DataFrame(spreads)
    print(f"Matched hedge pairs: {len(spread_df)}")
    print(f"Average spread: ${spread_df['spread'].mean():.2f}")
    print(f"Min spread: ${spread_df['spread'].min():.2f}")
    print(f"Max spread: ${spread_df['spread'].max():.2f}")
    print(f"Median spread: ${spread_df['spread'].median():.2f}")
    
    # Show highest spread trades
    print(f"\n--- HIGHEST SPREAD TRADES (top 10) ---")
    top_spreads = spread_df.nlargest(10, 'spread')
    for _, row in top_spreads.iterrows():
        print(f"  {row['time']} | LONG: {row['long_price']:.2f} | SHORT: {row['short_price']:.2f} | Spread: ${row['spread']:.2f}")
    
    # Spread distribution
    print(f"\n--- SPREAD DISTRIBUTION ---")
    bins = [0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, float('inf')]
    labels = ['$0-0.25', '$0.25-0.50', '$0.50-0.75', '$0.75-1.00', '$1.00-1.50', '$1.50-2.00', '$2.00+']
    spread_df['spread_bin'] = pd.cut(spread_df['spread'], bins=bins, labels=labels)
    print(spread_df['spread_bin'].value_counts().sort_index().to_string())
else:
    print("Could not match any hedge pairs for spread analysis")

# ============================================================================

# Calculate P&L
total_profit = hedge_deals['profit'].sum()
total_commission = hedge_deals['commission'].sum()
total_swap = hedge_deals['swap'].sum()
net_pnl = total_profit + total_commission + total_swap

print(f"\n--- P&L BREAKDOWN ---")
print(f"Gross Profit: ${total_profit:.2f}")
print(f"Commission: ${total_commission:.2f}")
print(f"Swap: ${total_swap:.2f}")
print(f"NET P&L: ${net_pnl:.2f}")

# Analyze by LONG vs SHORT
long_deals = hedge_deals[hedge_deals['magic'] == MAGIC_LONG]
short_deals = hedge_deals[hedge_deals['magic'] == MAGIC_SHORT]

long_profit = long_deals['profit'].sum() + long_deals['commission'].sum()
short_profit = short_deals['profit'].sum() + short_deals['commission'].sum()

print(f"\n--- BY SIDE ---")
print(f"LONG trades P&L: ${long_profit:.2f}")
print(f"SHORT trades P&L: ${short_profit:.2f}")

# Count wins vs losses
winning_deals = exits[exits['profit'] > 0]
losing_deals = exits[exits['profit'] < 0]
breakeven_deals = exits[exits['profit'] == 0]

print(f"\n--- INDIVIDUAL POSITION BREAKDOWN (not hedge-adjusted) ---")
print(f"Winning exits: {len(winning_deals)}")
print(f"Losing exits: {len(losing_deals)}")
print(f"Breakeven exits: {len(breakeven_deals)}")

# ============================================================================
# HEDGE PAIR WIN RATE - Based on net P&L per pair
# ============================================================================
print(f"\n--- HEDGE PAIR WIN RATE (Actual Performance) ---")

# Match LONG and SHORT exits by time (within 60 seconds = same pair)
long_exits = exits[exits['magic'] == MAGIC_LONG].copy()
short_exits = exits[exits['magic'] == MAGIC_SHORT].copy()

hedge_pairs = []
used_short_indices = set()

for _, long_row in long_exits.iterrows():
    long_time = long_row['time']
    long_profit = long_row['profit']
    
    # Find matching SHORT exit within 60 seconds
    for idx, short_row in short_exits.iterrows():
        if idx in used_short_indices:
            continue
        
        time_diff = abs((short_row['time'] - long_time).total_seconds())
        if time_diff <= 60:
            short_profit = short_row['profit']
            pair_profit = long_profit + short_profit
            
            hedge_pairs.append({
                'time': long_time,
                'long_profit': long_profit,
                'short_profit': short_profit,
                'pair_profit': pair_profit,
                'result': 'WIN' if pair_profit > 0 else ('LOSS' if pair_profit < 0 else 'BREAKEVEN')
            })
            used_short_indices.add(idx)
            break

if len(hedge_pairs) > 0:
    pair_df = pd.DataFrame(hedge_pairs)
    
    pair_wins = len(pair_df[pair_df['result'] == 'WIN'])
    pair_losses = len(pair_df[pair_df['result'] == 'LOSS'])
    pair_breakeven = len(pair_df[pair_df['result'] == 'BREAKEVEN'])
    
    hedge_win_rate = pair_wins / len(pair_df) * 100
    
    print(f"Total hedge pairs matched: {len(pair_df)}")
    print(f"Winning pairs (net profit > 0): {pair_wins}")
    print(f"Losing pairs (net profit < 0): {pair_losses}")
    print(f"Breakeven pairs: {pair_breakeven}")
    print(f"HEDGE WIN RATE: {hedge_win_rate:.1f}%")
    
    # Average profit per pair
    avg_pair_profit = pair_df['pair_profit'].mean()
    avg_winning_pair = pair_df[pair_df['result'] == 'WIN']['pair_profit'].mean() if pair_wins > 0 else 0
    avg_losing_pair = pair_df[pair_df['result'] == 'LOSS']['pair_profit'].mean() if pair_losses > 0 else 0
    
    print(f"\nAverage profit per hedge pair: ${avg_pair_profit:.2f}")
    print(f"Average winning pair profit: ${avg_winning_pair:.2f}")
    print(f"Average losing pair loss: ${avg_losing_pair:.2f}")
    
    # Profit factor
    total_pair_wins = pair_df[pair_df['result'] == 'WIN']['pair_profit'].sum()
    total_pair_losses = abs(pair_df[pair_df['result'] == 'LOSS']['pair_profit'].sum())
    profit_factor = total_pair_wins / total_pair_losses if total_pair_losses > 0 else float('inf')
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Show sample pairs
    print(f"\n--- SAMPLE HEDGE PAIRS (last 15) ---")
    print(pair_df.tail(15).to_string(index=False))
else:
    print("Could not match hedge pairs")

if len(exits) > 0:
    win_rate = len(winning_deals) / len(exits) * 100
    print(f"Win Rate: {win_rate:.1f}%")

# Average win vs average loss
if len(winning_deals) > 0:
    avg_win = winning_deals['profit'].mean()
    print(f"Average Win: ${avg_win:.2f}")
    
if len(losing_deals) > 0:
    avg_loss = losing_deals['profit'].mean()
    print(f"Average Loss: ${avg_loss:.2f}")
    
# Expected value per trade
if len(exits) > 0:
    ev = net_pnl / len(exits)
    print(f"\nExpected Value per exit: ${ev:.2f}")

# Analyze slippage by comparing planned vs actual
print(f"\n--- SAMPLE TRADES (last 20 exits) ---")
recent_exits = exits.tail(20)[['time', 'type', 'price', 'volume', 'profit', 'commission', 'magic']]
recent_exits['side'] = recent_exits['magic'].apply(lambda x: 'LONG' if x == MAGIC_LONG else 'SHORT')
print(recent_exits.to_string(index=False))

# Daily breakdown
hedge_deals['date'] = hedge_deals['time'].dt.date
daily_pnl = hedge_deals.groupby('date').agg({
    'profit': 'sum',
    'commission': 'sum',
    'swap': 'sum'
}).reset_index()
daily_pnl['net'] = daily_pnl['profit'] + daily_pnl['commission'] + daily_pnl['swap']

print(f"\n--- DAILY P&L ---")
print(daily_pnl.to_string(index=False))

# Detailed daily analysis
print(f"\n--- DETAILED DAILY ANALYSIS ---")
for date in sorted(hedge_deals['date'].unique()):
    day_deals = hedge_deals[hedge_deals['date'] == date]
    day_entries = day_deals[day_deals['entry'] == 0]
    day_exits = day_deals[day_deals['entry'] == 1]
    
    day_wins = day_exits[day_exits['profit'] > 0]
    day_losses = day_exits[day_exits['profit'] < 0]
    
    day_pnl = day_deals['profit'].sum()
    day_wr = len(day_wins) / len(day_exits) * 100 if len(day_exits) > 0 else 0
    
    # Calculate EV per trade for this day
    day_ev = day_pnl / len(day_exits) if len(day_exits) > 0 else 0
    
    print(f"{date} | Trades: {len(day_exits):>4} | Wins: {len(day_wins):>4} | Losses: {len(day_losses):>4} | WR: {day_wr:>5.1f}% | P&L: ${day_pnl:>8.2f} | EV/trade: ${day_ev:>5.2f}")

# Compare worst days
print(f"\n--- WORST DAYS COMPARISON ---")
worst_days = daily_pnl.nsmallest(3, 'net')
best_days = daily_pnl.nlargest(3, 'net')

print("Worst 3 days:")
for _, row in worst_days.iterrows():
    date = row['date']
    day_deals = hedge_deals[hedge_deals['date'] == date]
    day_exits = day_deals[day_deals['entry'] == 1]
    day_wins = day_exits[day_exits['profit'] > 0]
    day_wr = len(day_wins) / len(day_exits) * 100 if len(day_exits) > 0 else 0
    print(f"  {date}: ${row['net']:>8.2f} | {len(day_exits)} trades | {day_wr:.1f}% WR")

print("\nBest 3 days:")
for _, row in best_days.iterrows():
    date = row['date']
    day_deals = hedge_deals[hedge_deals['date'] == date]
    day_exits = day_deals[day_deals['entry'] == 1]
    day_wins = day_exits[day_exits['profit'] > 0]
    day_wr = len(day_wins) / len(day_exits) * 100 if len(day_exits) > 0 else 0
    print(f"  {date}: ${row['net']:>8.2f} | {len(day_exits)} trades | {day_wr:.1f}% WR")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total Hedge Pairs: ~{len(entries)//2}")
print(f"Net P&L: ${net_pnl:.2f}")
print(f"Commission Impact: ${total_commission:.2f}")

# Calculate what % of trades would need to be winners to break even
# Given spread + slippage costs
if len(exits) > 0 and len(losing_deals) > 0:
    avg_loss_size = abs(losing_deals['profit'].mean())
    avg_win_size = winning_deals['profit'].mean() if len(winning_deals) > 0 else 0
    
    if avg_win_size > 0 and avg_loss_size > 0:
        breakeven_wr = avg_loss_size / (avg_win_size + avg_loss_size) * 100
        print(f"\nBreakeven Win Rate needed: {breakeven_wr:.1f}%")
        print(f"Your actual Win Rate: {win_rate:.1f}%")
        
        if win_rate > breakeven_wr:
            print("✓ You ARE profitable with current spread/slippage")
        else:
            print("✗ You are NOT profitable - need higher win rate or better fills")

mt5.shutdown()
