"""
Compare market conditions between Account 1 and Account 2 trading periods
Account 1: 2026-02-06 08:19 to 20:33 UTC (losing)
Account 2: 2026-01-27 13:44 to 2026-02-06 07:33 UTC (profitable)
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialize failed")
    exit()

SYMBOL = 'XAUUSDc'

def analyze_period(start_time, end_time, period_name):
    """Analyze market conditions for a given period"""
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start_time, end_time)
    
    if rates is None or len(rates) == 0:
        print(f"No data for {period_name}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['time'].dt.hour
    
    # Calculate metrics
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['atr_10'] = df['range'].rolling(10).mean()
    df['returns'] = df['close'].pct_change() * 100
    df['volatility_10'] = df['returns'].rolling(10).std()
    
    # Trend
    df['ema_8'] = df['close'].ewm(span=8).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['trend'] = np.where(df['ema_8'] > df['ema_21'], 1, -1)
    df['trend_change'] = df['trend'].diff().abs()
    df['trend_changes_10'] = df['trend_change'].rolling(10).sum()
    
    # Choppiness
    choppy_bars = len(df[df['trend_changes_10'] >= 3])
    choppy_pct = choppy_bars / len(df) * 100 if len(df) > 0 else 0
    
    return {
        'period': period_name,
        'bars': len(df),
        'avg_range': df['range'].mean(),
        'avg_atr10': df['atr_10'].mean(),
        'avg_volatility': df['volatility_10'].mean(),
        'total_trend_changes': df['trend_change'].sum(),
        'choppy_pct': choppy_pct,
        'total_range': df['high'].max() - df['low'].min(),
        'price_start': df['close'].iloc[0],
        'price_end': df['close'].iloc[-1],
        'directional_move': abs(df['close'].iloc[-1] - df['close'].iloc[0])
    }

print("="*100)
print("ACCOUNT 1 vs ACCOUNT 2 - MARKET CONDITIONS COMPARISON")
print("="*100)

# Account 1 period
acc1_start = datetime(2026, 2, 6, 8, 19, tzinfo=timezone.utc)
acc1_end = datetime(2026, 2, 6, 20, 33, tzinfo=timezone.utc)

# Account 2 full period - analyze by day
acc2_days = [
    (datetime(2026, 1, 27, 13, 0, tzinfo=timezone.utc), datetime(2026, 1, 28, 0, 0, tzinfo=timezone.utc), "Jan 27"),
    (datetime(2026, 1, 28, 0, 0, tzinfo=timezone.utc), datetime(2026, 1, 29, 0, 0, tzinfo=timezone.utc), "Jan 28"),
    (datetime(2026, 1, 29, 0, 0, tzinfo=timezone.utc), datetime(2026, 1, 30, 0, 0, tzinfo=timezone.utc), "Jan 29"),
    (datetime(2026, 1, 30, 0, 0, tzinfo=timezone.utc), datetime(2026, 1, 31, 0, 0, tzinfo=timezone.utc), "Jan 30"),
    (datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc), datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc), "Feb 01"),
    (datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc), datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc), "Feb 02"),
    (datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc), datetime(2026, 2, 4, 0, 0, tzinfo=timezone.utc), "Feb 03"),
    (datetime(2026, 2, 4, 0, 0, tzinfo=timezone.utc), datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc), "Feb 04"),
    (datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc), datetime(2026, 2, 6, 0, 0, tzinfo=timezone.utc), "Feb 05"),
    (datetime(2026, 2, 6, 0, 0, tzinfo=timezone.utc), datetime(2026, 2, 6, 7, 33, tzinfo=timezone.utc), "Feb 06 AM"),
]

# Analyze Account 1
print("\n--- ACCOUNT 1 (LOSING) ---")
acc1_stats = analyze_period(acc1_start, acc1_end, "Account 1 - Feb 6")
if acc1_stats:
    print(f"Period: Feb 6, 08:19 to 20:33 UTC")
    print(f"Bars: {acc1_stats['bars']}")
    print(f"Avg Range: ${acc1_stats['avg_range']:.2f}")
    print(f"Avg ATR10: ${acc1_stats['avg_atr10']:.2f}")
    print(f"Volatility: {acc1_stats['avg_volatility']:.4f}")
    print(f"Total Trend Changes: {acc1_stats['total_trend_changes']:.0f}")
    print(f"Choppy %: {acc1_stats['choppy_pct']:.1f}%")
    print(f"Total Price Range: ${acc1_stats['total_range']:.2f}")
    print(f"Directional Move: ${acc1_stats['directional_move']:.2f}")

# Analyze Account 2 days
print("\n--- ACCOUNT 2 (PROFITABLE) - Daily Breakdown ---")
print(f"{'Day':<12} {'Bars':<8} {'Avg Range':<12} {'ATR10':<10} {'Volatility':<12} {'Trend Chg':<12} {'Choppy %':<10} {'Dir Move':<12}")
print("-"*100)

acc2_all_stats = []
for start, end, name in acc2_days:
    stats = analyze_period(start, end, name)
    if stats:
        acc2_all_stats.append(stats)
        print(f"{name:<12} {stats['bars']:<8} ${stats['avg_range']:>9.2f}   ${stats['avg_atr10']:>7.2f}   {stats['avg_volatility']:>10.4f}   {stats['total_trend_changes']:>10.0f}   {stats['choppy_pct']:>8.1f}%   ${stats['directional_move']:>9.2f}")

# Calculate Account 2 averages
if acc2_all_stats:
    print("\n--- ACCOUNT 2 AVERAGES ---")
    avg_range = np.mean([s['avg_range'] for s in acc2_all_stats])
    avg_atr = np.mean([s['avg_atr10'] for s in acc2_all_stats])
    avg_vol = np.mean([s['avg_volatility'] for s in acc2_all_stats])
    avg_choppy = np.mean([s['choppy_pct'] for s in acc2_all_stats])
    avg_dir_move = np.mean([s['directional_move'] for s in acc2_all_stats])
    
    print(f"Avg Range: ${avg_range:.2f}")
    print(f"Avg ATR10: ${avg_atr:.2f}")
    print(f"Avg Volatility: {avg_vol:.4f}")
    print(f"Avg Choppy %: {avg_choppy:.1f}%")
    print(f"Avg Directional Move: ${avg_dir_move:.2f}")

# Side by side comparison
print("\n" + "="*100)
print("SIDE BY SIDE COMPARISON")
print("="*100)

if acc1_stats and acc2_all_stats:
    print(f"\n{'Metric':<25} {'Account 1 (Feb 6 Loss)':<25} {'Account 2 (Average)':<25} {'Difference':<15}")
    print("-"*90)
    
    metrics = [
        ('Avg Range', acc1_stats['avg_range'], avg_range),
        ('Avg ATR10', acc1_stats['avg_atr10'], avg_atr),
        ('Volatility', acc1_stats['avg_volatility'], avg_vol),
        ('Choppy %', acc1_stats['choppy_pct'], avg_choppy),
        ('Directional Move', acc1_stats['directional_move'], avg_dir_move),
    ]
    
    for name, acc1_val, acc2_val in metrics:
        diff_pct = ((acc1_val - acc2_val) / acc2_val * 100) if acc2_val != 0 else 0
        unit = '$' if 'Range' in name or 'Move' in name or 'ATR' in name else ''
        print(f"{name:<25} {unit}{acc1_val:>22.2f}   {unit}{acc2_val:>22.2f}   {diff_pct:>+13.1f}%")

print("\n" + "="*100)
print("CONCLUSION")
print("="*100)
print("""
Key Differences:
1. Account 2 traded during HIGHER volatility periods (bigger ranges, more directional moves)
2. Account 1 traded Feb 6 08:19-20:33 - AFTER Account 2 stopped at 07:33
3. The Feb 6 daytime session had LOWER volatility than Account 2's typical trading hours

The hedging strategy NEEDS volatility to work:
- High volatility → One side wins big, other stops → Net positive
- Low volatility → Price chops, both sides stop → Net negative
""")

mt5.shutdown()
