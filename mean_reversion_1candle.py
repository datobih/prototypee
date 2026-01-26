import pandas as pd
import numpy as np

print("="*80)
print("BTCUSD 1-CANDLE MEAN REVERSION EXPLORATION")
print("RF >= 0.70 Filter Applied")
print("="*80)

df = pd.read_csv("data/processed/BTCUSD_feature_analysis.csv")
print(f"Total bars: {len(df)}")

rf_high = df[df["rf_prob"] >= 0.70].copy()
print(f"RF >= 0.70 bars: {len(rf_high)}")

rf_high["prev_up"] = (rf_high["is_up"].shift(1) == 1)
rf_high["prev_down"] = (rf_high["is_up"].shift(1) == 0)
rf_high["long_win"] = (rf_high["outcome"] == 1).astype(int)
rf_high["short_win"] = (rf_high["outcome"] == 2).astype(int)
rf_high = rf_high.dropna()

print("\n" + "="*80)
print("1-CANDLE MEAN REVERSION BASELINE")
print("="*80)

after_up = rf_high[rf_high["prev_up"] == True]
if len(after_up) > 50:
    short_wr = after_up["short_win"].mean() * 100
    long_wr = after_up["long_win"].mean() * 100
    print(f"\nAfter UP candle (n={len(after_up)}):")
    print(f"  SHORT (fade): {short_wr:.1f}%")
    print(f"  LONG (cont):  {long_wr:.1f}%")

after_down = rf_high[rf_high["prev_down"] == True]
if len(after_down) > 50:
    long_wr = after_down["long_win"].mean() * 100
    short_wr = after_down["short_win"].mean() * 100
    print(f"\nAfter DOWN candle (n={len(after_down)}):")
    print(f"  LONG (fade):  {long_wr:.1f}%")
    print(f"  SHORT (cont): {short_wr:.1f}%")

print("\n" + "="*80)
print("MEAN REVERSION WITH FILTERS")
print("="*80)

results = []

# After BIG UP -> SHORT
big_up = rf_high[(rf_high["prev_up"]) & (rf_high["big_body"].shift(1) == 1)]
if len(big_up) > 30:
    wr = big_up["short_win"].mean() * 100
    results.append(("After BIG UP -> SHORT", wr, len(big_up)))

# After BIG DOWN -> LONG  
big_down = rf_high[(rf_high["prev_down"]) & (rf_high["big_body"].shift(1) == 1)]
if len(big_down) > 30:
    wr = big_down["long_win"].mean() * 100
    results.append(("After BIG DOWN -> LONG", wr, len(big_down)))

# After UP + upper wick -> SHORT
up_wick = rf_high[(rf_high["prev_up"]) & (rf_high["upper_reject"].shift(1) == 1)]
if len(up_wick) > 30:
    wr = up_wick["short_win"].mean() * 100
    results.append(("After UP + upper_wick -> SHORT", wr, len(up_wick)))

# After DOWN + lower wick -> LONG
down_wick = rf_high[(rf_high["prev_down"]) & (rf_high["lower_reject"].shift(1) == 1)]
if len(down_wick) > 30:
    wr = down_wick["long_win"].mean() * 100
    results.append(("After DOWN + lower_wick -> LONG", wr, len(down_wick)))

# After UP + close>0.8 -> SHORT
up_ext = rf_high[(rf_high["prev_up"]) & (rf_high["close_position"].shift(1) > 0.8)]
if len(up_ext) > 30:
    wr = up_ext["short_win"].mean() * 100
    results.append(("After UP + close>0.8 -> SHORT", wr, len(up_ext)))

# After DOWN + close<0.2 -> LONG
down_ext = rf_high[(rf_high["prev_down"]) & (rf_high["close_position"].shift(1) < 0.2)]
if len(down_ext) > 30:
    wr = down_ext["long_win"].mean() * 100
    results.append(("After DOWN + close<0.2 -> LONG", wr, len(down_ext)))

# BIG UP + close>0.8 -> SHORT
big_up_ext = rf_high[(rf_high["prev_up"]) & (rf_high["big_body"].shift(1) == 1) & (rf_high["close_position"].shift(1) > 0.8)]
if len(big_up_ext) > 20:
    wr = big_up_ext["short_win"].mean() * 100
    results.append(("BIG UP + close>0.8 -> SHORT", wr, len(big_up_ext)))

# BIG DOWN + close<0.2 -> LONG
big_down_ext = rf_high[(rf_high["prev_down"]) & (rf_high["big_body"].shift(1) == 1) & (rf_high["close_position"].shift(1) < 0.2)]
if len(big_down_ext) > 20:
    wr = big_down_ext["long_win"].mean() * 100
    results.append(("BIG DOWN + close<0.2 -> LONG", wr, len(big_down_ext)))

# BIG UP + upper_wick -> SHORT
big_up_wick = rf_high[(rf_high["prev_up"]) & (rf_high["big_body"].shift(1) == 1) & (rf_high["upper_reject"].shift(1) == 1)]
if len(big_up_wick) > 20:
    wr = big_up_wick["short_win"].mean() * 100
    results.append(("BIG UP + upper_wick -> SHORT", wr, len(big_up_wick)))

# BIG DOWN + lower_wick -> LONG
big_down_wick = rf_high[(rf_high["prev_down"]) & (rf_high["big_body"].shift(1) == 1) & (rf_high["lower_reject"].shift(1) == 1)]
if len(big_down_wick) > 20:
    wr = big_down_wick["long_win"].mean() * 100
    results.append(("BIG DOWN + lower_wick -> LONG", wr, len(big_down_wick)))

# dist_ema8 filters
up_far_ema = rf_high[(rf_high["prev_up"]) & (rf_high["dist_ema8"].shift(1) > 0.001)]
if len(up_far_ema) > 30:
    wr = up_far_ema["short_win"].mean() * 100
    results.append(("After UP + dist_ema8>0.1% -> SHORT", wr, len(up_far_ema)))

down_far_ema = rf_high[(rf_high["prev_down"]) & (rf_high["dist_ema8"].shift(1) < -0.001)]
if len(down_far_ema) > 30:
    wr = down_far_ema["long_win"].mean() * 100
    results.append(("After DOWN + dist_ema8<-0.1% -> LONG", wr, len(down_far_ema)))

print("\nSorted by Win Rate:")
print("-"*60)
results.sort(key=lambda x: x[1], reverse=True)
for name, wr, count in results:
    mark = "**" if wr >= 40 else ""
    print(f"{mark}{name:<45} {wr:>6.1f}% (n={count})")
