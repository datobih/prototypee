import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

mt5.initialize()

deals = mt5.history_deals_get(datetime(2025, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc))
df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())

df = df[(df['magic'].isin([123456, 654321])) & (df['entry'] == 1)]
df = df.sort_values('time').reset_index(drop=True)

symbol = df['symbol'].iloc[0]
print(f'Symbol: {symbol}')
print(f'Total trades: {len(df)}')
print(f'Final P&L: ${df["profit"].sum():.2f}')
print()

# Calculate spread cost
# For 0.05 lots on gold, spread cost depends on pip value
# Typical spread = 25-35 pips, each pip = $0.50 for 0.05 lots
total_trades = len(df)

spread_per_trade_low = 1.00   # $1.00 per round trip
spread_per_trade_avg = 1.50   # $1.50 per round trip
spread_per_trade_high = 2.00  # $2.00 per round trip

total_spread_low = total_trades * spread_per_trade_low
total_spread_avg = total_trades * spread_per_trade_avg
total_spread_high = total_trades * spread_per_trade_high

final_pnl = df['profit'].sum()

print('--- SPREAD IMPACT ANALYSIS ---')
print(f'If spread was $1.00/trade: Total spread cost = ${total_spread_low:.2f}')
print(f'  P&L with 0 spread = ${final_pnl:.2f} + ${total_spread_low:.2f} = ${final_pnl + total_spread_low:.2f}')
print()
print(f'If spread was $1.50/trade: Total spread cost = ${total_spread_avg:.2f}')
print(f'  P&L with 0 spread = ${final_pnl:.2f} + ${total_spread_avg:.2f} = ${final_pnl + total_spread_avg:.2f}')
print()
print(f'If spread was $2.00/trade: Total spread cost = ${total_spread_high:.2f}')
print(f'  P&L with 0 spread = ${final_pnl:.2f} + ${total_spread_high:.2f} = ${final_pnl + total_spread_high:.2f}')
print()

# The REAL issue - win rate and double stops
print('--- THE REAL ISSUE ---')
losses = df[df['profit'] < 0]
wins = df[df['profit'] > 0]
print(f'Winning trades: {len(wins)} (avg: ${wins["profit"].mean():.2f})')
print(f'Losing trades: {len(losses)} (avg: ${losses["profit"].mean():.2f})')
print(f'Win rate: {len(wins)/len(df)*100:.1f}%')
print()

# Double stops (both LONG and SHORT losing within same minute)
df['side'] = df['magic'].map({123456: 'LONG', 654321: 'SHORT'})
df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
losses = df[df['profit'] < 0]  # re-filter with datetime
both_lose = losses.groupby(losses['time'].dt.floor('min')).size()
double_stops = (both_lose >= 2).sum()
print(f'Minutes with 2+ losses (double-stops): {double_stops}')
print()

# Calculate double-stop impact
# When both stops hit, you lose ~$25 x 2 = $50 instead of gaining net $2.50
print('--- DOUBLE-STOP IMPACT ---')
double_stop_loss = double_stops * 25 * 2  # Both sides hit stop
normal_win = (len(df)/2 - double_stops) * 2.50  # Normal hedge profit
print(f'Estimated double-stop losses: ${double_stop_loss:.2f}')
print(f'If no double-stops, approx gain: ${normal_win:.2f}')

mt5.shutdown()
