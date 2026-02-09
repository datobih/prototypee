"""
Analyze market conditions during Account 1's trading period
Date range: 2026-02-06 08:19:02 to 2026-02-06 20:33:11 UTC
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialize failed")
    exit()

SYMBOL = 'XAUUSDc'  # Account 1 symbol

# Account 1 trading period
start_time = datetime(2026, 2, 6, 8, 0, tzinfo=timezone.utc)
end_time = datetime(2026, 2, 6, 21, 0, tzinfo=timezone.utc)

# Get 1-minute data for the trading period + some context before
context_start = start_time - timedelta(hours=24)  # 24 hours before for comparison
rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, context_start, end_time)

if rates is None or len(rates) == 0:
    print(f"Failed to get data for {SYMBOL}")
    mt5.shutdown()
    exit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
df['hour'] = df['time'].dt.hour

print("="*80)
print(f"MARKET CONDITIONS ANALYSIS - {SYMBOL}")
print(f"Account 1 Trading Period: 2026-02-06 08:19 to 20:33 UTC")
print("="*80)

# Calculate volatility metrics
df['range'] = df['high'] - df['low']
df['body'] = abs(df['close'] - df['open'])
df['atr_10'] = df['range'].rolling(10).mean()
df['atr_20'] = df['range'].rolling(20).mean()
df['atr_60'] = df['range'].rolling(60).mean()  # 1-hour ATR

# Price movement
df['returns'] = df['close'].pct_change() * 100
df['volatility_10'] = df['returns'].rolling(10).std()
df['volatility_60'] = df['returns'].rolling(60).std()

# Trend strength
df['ema_8'] = df['close'].ewm(span=8).mean()
df['ema_21'] = df['close'].ewm(span=21).mean()
df['trend'] = np.where(df['ema_8'] > df['ema_21'], 1, -1)
df['trend_change'] = df['trend'].diff().abs()

# Split into periods
trading_period = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
before_period = df[(df['time'] >= context_start) & (df['time'] < start_time)]

print(f"\nData loaded: {len(df)} bars total")
print(f"Trading period bars: {len(trading_period)}")
print(f"Before period bars: {len(before_period)}")

# Compare volatility
print("\n" + "="*80)
print("VOLATILITY COMPARISON")
print("="*80)

metrics = ['range', 'atr_10', 'atr_20', 'volatility_10']
print(f"\n{'Metric':<20} {'Before (24h)':<15} {'During Trading':<15} {'Difference':<15}")
print("-"*65)

for metric in metrics:
    before_val = before_period[metric].mean()
    during_val = trading_period[metric].mean()
    diff_pct = ((during_val - before_val) / before_val) * 100 if before_val != 0 else 0
    print(f"{metric:<20} {before_val:>14.4f} {during_val:>14.4f} {diff_pct:>+13.1f}%")

# Hourly breakdown during trading
print("\n" + "="*80)
print("HOURLY BREAKDOWN DURING TRADING PERIOD")
print("="*80)

hourly = trading_period.groupby('hour').agg({
    'range': 'mean',
    'atr_10': 'mean',
    'volatility_10': 'mean',
    'trend_change': 'sum',
    'close': ['first', 'last']
}).reset_index()

hourly.columns = ['hour', 'avg_range', 'avg_atr10', 'avg_vol', 'trend_changes', 'open_price', 'close_price']
hourly['hour_move'] = hourly['close_price'] - hourly['open_price']
hourly['hour_move_pct'] = (hourly['hour_move'] / hourly['open_price']) * 100

print(f"\n{'Hour':<6} {'Avg Range':<12} {'Avg ATR10':<12} {'Volatility':<12} {'Trend Chg':<12} {'Hour Move':<12}")
print("-"*70)
for _, row in hourly.iterrows():
    print(f"{int(row['hour']):>4}   {row['avg_range']:>10.2f}   {row['avg_atr10']:>10.2f}   {row['avg_vol']:>10.4f}   {int(row['trend_changes']):>10}   {row['hour_move']:>+10.2f}")

# Identify choppy periods (high trend changes = choppy)
print("\n" + "="*80)
print("CHOPPINESS ANALYSIS")
print("="*80)

# Rolling trend changes (measure of choppiness)
trading_period = trading_period.copy()
trading_period['trend_changes_10'] = trading_period['trend_change'].rolling(10).sum()

choppy_bars = trading_period[trading_period['trend_changes_10'] >= 3]
smooth_bars = trading_period[trading_period['trend_changes_10'] < 3]

print(f"\nChoppy bars (3+ trend changes in 10 bars): {len(choppy_bars)} ({len(choppy_bars)/len(trading_period)*100:.1f}%)")
print(f"Smooth bars (< 3 trend changes in 10 bars): {len(smooth_bars)} ({len(smooth_bars)/len(trading_period)*100:.1f}%)")

# Price range analysis
print("\n" + "="*80)
print("PRICE RANGE ANALYSIS")
print("="*80)

total_high = trading_period['high'].max()
total_low = trading_period['low'].min()
total_range = total_high - total_low
avg_range = trading_period['range'].mean()

print(f"\nTrading period high: {total_high:.2f}")
print(f"Trading period low: {total_low:.2f}")
print(f"Total range: ${total_range:.2f}")
print(f"Average 1-min range: ${avg_range:.2f}")
print(f"Range/ATR ratio: {total_range / (avg_range * len(trading_period)):.2f}")

# Compare to Account 2's profitable days
print("\n" + "="*80)
print("COMPARISON: Account 1 Day vs Account 2 Best Days")
print("="*80)

# Get data for a profitable day (Jan 30)
profitable_start = datetime(2026, 1, 30, 8, 0, tzinfo=timezone.utc)
profitable_end = datetime(2026, 1, 30, 21, 0, tzinfo=timezone.utc)
profitable_rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, profitable_start, profitable_end)

if profitable_rates is not None and len(profitable_rates) > 0:
    prof_df = pd.DataFrame(profitable_rates)
    prof_df['time'] = pd.to_datetime(prof_df['time'], unit='s', utc=True)
    prof_df['range'] = prof_df['high'] - prof_df['low']
    prof_df['returns'] = prof_df['close'].pct_change() * 100
    prof_df['volatility_10'] = prof_df['returns'].rolling(10).std()
    prof_df['ema_8'] = prof_df['close'].ewm(span=8).mean()
    prof_df['ema_21'] = prof_df['close'].ewm(span=21).mean()
    prof_df['trend'] = np.where(prof_df['ema_8'] > prof_df['ema_21'], 1, -1)
    prof_df['trend_change'] = prof_df['trend'].diff().abs()
    
    print(f"\n{'Metric':<25} {'Feb 6 (Loss)':<15} {'Jan 30 (Profit)':<15}")
    print("-"*55)
    print(f"{'Avg Range':<25} ${trading_period['range'].mean():>12.2f} ${prof_df['range'].mean():>12.2f}")
    print(f"{'Volatility (10-bar std)':<25} {trading_period['volatility_10'].mean():>12.4f}  {prof_df['volatility_10'].mean():>12.4f}")
    print(f"{'Trend Changes':<25} {trading_period['trend_change'].sum():>12.0f}  {prof_df['trend_change'].sum():>12.0f}")
    print(f"{'Total Price Range':<25} ${(trading_period['high'].max() - trading_period['low'].min()):>12.2f} ${(prof_df['high'].max() - prof_df['low'].min()):>12.2f}")

# Identify specific problem periods
print("\n" + "="*80)
print("PROBLEM PERIODS - High Volatility + High Choppiness")
print("="*80)

trading_period['problem_score'] = (
    (trading_period['range'] > trading_period['range'].quantile(0.75)).astype(int) +
    (trading_period['trend_changes_10'] >= 3).astype(int)
)

problem_periods = trading_period[trading_period['problem_score'] >= 2]
print(f"\nHigh-risk bars (high range + choppy): {len(problem_periods)} ({len(problem_periods)/len(trading_period)*100:.1f}%)")

if len(problem_periods) > 0:
    print("\nSample problem periods:")
    print(problem_periods[['time', 'range', 'trend_changes_10', 'close']].head(20).to_string(index=False))

# Summary
print("\n" + "="*80)
print("SUMMARY - WHY ACCOUNT 1 LOST ON FEB 6")
print("="*80)

avg_range_trading = trading_period['range'].mean()
avg_vol_trading = trading_period['volatility_10'].mean()
choppy_pct = len(choppy_bars) / len(trading_period) * 100

print(f"""
Key Findings:
1. Average 1-min range: ${avg_range_trading:.2f}
2. Choppiness: {choppy_pct:.1f}% of bars were choppy (frequent trend changes)
3. Total trend changes: {int(trading_period['trend_change'].sum())}
4. Total price swing: ${total_range:.2f}

Implications for Hedging:
- With $2.50 stop and ${avg_range_trading:.2f} average range, stops can be hit in ~{2.5/avg_range_trading:.0f} bars
- High choppiness ({choppy_pct:.1f}%) means price whipsaws through both stops frequently
- This explains why BOTH sides of hedges got stopped out (double losses)

Recommendations:
1. Skip trading when choppiness > 40%
2. Widen stops to at least 2x average range (${avg_range_trading*2:.2f})
3. Add ATR filter - skip when ATR > ${trading_period['atr_10'].quantile(0.75):.2f}
""")

mt5.shutdown()
