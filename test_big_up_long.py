import pandas as pd
df = pd.read_csv("data/processed/BTCUSD_feature_analysis.csv")
rf = df[df["rf_prob"] >= 0.70].copy()
rf["prev_up"] = rf["is_up"].shift(1) == 1
rf["long_win"] = (rf["outcome"] == 1).astype(int)
rf["short_win"] = (rf["outcome"] == 2).astype(int)

print("="*60)
print("AFTER BIG UP CANDLE - LONG vs SHORT")
print("="*60)

big_up = rf[(rf["prev_up"]) & (rf["big_body"].shift(1) == 1)]
long_wr = big_up["long_win"].mean()*100
short_wr = big_up["short_win"].mean()*100
print(f"\nAfter BIG UP (n={len(big_up)}):")
print(f"  LONG (continue):    {long_wr:.1f}%")
print(f"  SHORT (fade):       {short_wr:.1f}%")

big_up_ext = rf[(rf["prev_up"]) & (rf["big_body"].shift(1) == 1) & (rf["close_position"].shift(1) > 0.8)]
long_wr = big_up_ext["long_win"].mean()*100
short_wr = big_up_ext["short_win"].mean()*100
print(f"\nAfter BIG UP + close>0.8 (n={len(big_up_ext)}):")
print(f"  LONG (continue):    {long_wr:.1f}%")
print(f"  SHORT (fade):       {short_wr:.1f}%")

big_up_wick = rf[(rf["prev_up"]) & (rf["big_body"].shift(1) == 1) & (rf["upper_reject"].shift(1) == 1)]
long_wr = big_up_wick["long_win"].mean()*100
short_wr = big_up_wick["short_win"].mean()*100
print(f"\nAfter BIG UP + upper_wick (n={len(big_up_wick)}):")
print(f"  LONG (continue):    {long_wr:.1f}%")
print(f"  SHORT (fade):       {short_wr:.1f}%")

rf75 = df[df["rf_prob"] >= 0.75].copy()
rf75["prev_up"] = rf75["is_up"].shift(1) == 1
rf75["long_win"] = (rf75["outcome"] == 1).astype(int)
rf75["short_win"] = (rf75["outcome"] == 2).astype(int)
big_up_75 = rf75[(rf75["prev_up"]) & (rf75["big_body"].shift(1) == 1)]
long_wr = big_up_75["long_win"].mean()*100
short_wr = big_up_75["short_win"].mean()*100
print(f"\nRF>=0.75 + BIG UP (n={len(big_up_75)}):")
print(f"  LONG (continue):    {long_wr:.1f}%")
print(f"  SHORT (fade):       {short_wr:.1f}%")

df["hour"] = pd.to_datetime(df["Datetime"]).dt.hour
df["ny_overlap"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)
rf = df[df["rf_prob"] >= 0.70].copy()
rf["prev_up"] = rf["is_up"].shift(1) == 1
rf["long_win"] = (rf["outcome"] == 1).astype(int)
big_up_ny = rf[(rf["prev_up"]) & (rf["big_body"].shift(1) == 1) & (rf["ny_overlap"].shift(1) == 1)]
long_wr = big_up_ny["long_win"].mean()*100
print(f"\nBIG UP + NY (n={len(big_up_ny)}):")
print(f"  LONG (continue):    {long_wr:.1f}%")

