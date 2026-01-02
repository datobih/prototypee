import pandas as pd

df = pd.read_csv("data/processed/XAUUSD1_feature_analysis.csv", index_col=0, parse_dates=True)

df["hour"] = df.index.hour
df["session"] = df["hour"].apply(lambda h: "NY_OVERLAP" if 13 <= h < 17 else "OTHER")
df["month"] = df.index.month

aug_sep = df[(df["month"] >= 8) & (df["month"] <= 9)]

print("="*60)
print("AUGUST-SEPTEMBER ANALYSIS")
print("="*60)
print(f"Total bars in Aug-Sep: {len(aug_sep)}")
print(f"NY_OVERLAP bars: {(aug_sep['session'] == 'NY_OVERLAP').sum()}")

movement = aug_sep[aug_sep["outcome"] != 0]
ny_movement = movement[movement["session"] == "NY_OVERLAP"]
print(f"NY_OVERLAP movement bars: {len(ny_movement)}")

pattern = ((ny_movement["up_count_3"] == 3) & (ny_movement["big_body"] == 1) & (ny_movement["close_position"] > 0.7))
rf_pass = ny_movement["rf_prob"] >= 0.70

print()
print("CONDITION BREAKDOWN:")
print("-"*60)
print(f"1. Pattern PASS, RF FAIL: {(pattern & ~rf_pass).sum()}")
print(f"2. RF PASS, Pattern FAIL: {(~pattern & rf_pass).sum()}")
print(f"3. BOTH PASS (Rule 4):    {(pattern & rf_pass).sum()}")
print(f"4. BOTH FAIL:             {(~pattern & ~rf_pass).sum()}")

print()
print("PATTERN SUB-CONDITIONS:")
print(f"  up_count_3 == 3:        {(ny_movement['up_count_3'] == 3).sum()}")
print(f"  big_body == 1:          {(ny_movement['big_body'] == 1).sum()}")
print(f"  close_position > 0.7:   {(ny_movement['close_position'] > 0.7).sum()}")
print(f"  All three combined:     {pattern.sum()}")

print()
print("RF PROBABILITY DISTRIBUTION:")
for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
    count = (ny_movement["rf_prob"] >= thresh).sum()
    print(f"  RF >= {thresh}: {count}")
