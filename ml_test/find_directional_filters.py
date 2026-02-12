"""
Analyze directional bias in high-probability RF setups.
Goal: Find features that separate LONG wins (outcome=1) from SHORT wins (outcome=2)
when the RF model is already fairly confident (RF >= 0.75).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading data...")
df = pd.read_csv('data/processed/XAUUSD1_feature_analysis.csv')
print(f"Total rows: {len(df)}")

# Filter for high probability RF setups
RF_THRESHOLD = 0.75
subset = df[df['rf_prob'] >= RF_THRESHOLD].copy()
print(f"\nHigh Probability Setups (RF >= {RF_THRESHOLD}): {len(subset)}")

# Define targets
# 1 = Long Win, 2 = Short Win
# We ignore 0 (Loss) for directional bias analysis, or treat them as noise?
# Let's focus on separating the winners first.
winners = subset[subset['outcome'].isin([1, 2])].copy()
print(f"Winning Setups (Outcome 1 or 2): {len(winners)}")
print(f"  Long Wins (1): {(winners['outcome'] == 1).sum()} ({((winners['outcome'] == 1).sum() / len(winners) * 100):.1f}%)")
print(f"  Short Wins (2): {(winners['outcome'] == 2).sum()} ({((winners['outcome'] == 2).sum() / len(winners) * 100):.1f}%)")

# Features to analyze (numeric only)
# Exclude non-feature columns
exclude_cols = ['Datetime', 'outcome', 'session', 'rf_prob', 'lr_prob', 'entry', 'stop', 'target', 'close_position_val', 'high_10', 'low_10']
feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

print(f"\nAnalyzing {len(feature_cols)} features for directional bias...")

results = []

for feat in feature_cols:
    try:
        longs = winners[winners['outcome'] == 1][feat]
        shorts = winners[winners['outcome'] == 2][feat]
        
        # Calculate means
        mean_long = longs.mean()
        mean_short = shorts.mean()
        
        # Calculate standardized difference (Cohen's d approximation)
        pooled_std = np.sqrt((longs.std()**2 + shorts.std()**2) / 2)
        if pooled_std == 0:
            continue
            
        diff_std = (mean_long - mean_short) / pooled_std
        
        # Calculate separation score (simple Area Under Curve proxy)
        # How well does this feature split the two classes?
        # We can simulate a threshold classifier
        
        results.append({
            'Feature': feat,
            'Long Mean': mean_long,
            'Short Mean': mean_short,
            'Diff Std': diff_std,
            'Abs Diff Std': abs(diff_std)
        })
    except Exception as e:
        # print(f"Error analyzing {feat}: {e}")
        pass

# Convert to DataFrame
res_df = pd.DataFrame(results)
res_df = res_df.sort_values('Abs Diff Std', ascending=False)

print("\n" + "="*80)
print("TOP DIRECTIONAL FEATURES (Separation between LONG and SHORT wins)")
print("="*80)
print(f"{'Feature':<30} {'Long Mean':<12} {'Short Mean':<12} {'Separation (Std Dev)'}")
print("-" * 80)

top_features = res_df.head(20)
for _, row in top_features.iterrows():
    print(f"{row['Feature']:<30} {row['Long Mean']:<12.4f} {row['Short Mean']:<12.4f} {row['Diff Std']:.4f}")

# Generate Candidate Rules
print("\n" + "="*80)
print("GENERATING CANDIDATE DIRECTIONAL RULES")
print("="*80)

best_rules = []

for _, row in top_features.iterrows():
    feat = row['Feature']
    # If Long Mean > Short Mean, we look for High Values -> Long
    # If Long Mean < Short Mean, we look for Low Values -> Long
    
    direction = "High -> Long" if row['Long Mean'] > row['Short Mean'] else "Low -> Long"
    
    # Simple threshold optimization
    # Test percentiles as thresholds
    thresholds = winners[feat].quantile(np.linspace(0.1, 0.9, 9)).values
    
    for t in thresholds:
        if direction == "High -> Long":
            # Rule: Feature > Threshold implies Long
            mask = winners[feat] > t
            if mask.sum() < 20: continue # Skip if too few samples
            long_rate = (winners[mask]['outcome'] == 1).mean()
            if long_rate > 0.60: # Informative?
                best_rules.append({
                    'Feature': feat,
                    'Condition': '>',
                    'Threshold': t,
                    'Predicted': 'LONG',
                    'Win Rate': long_rate,
                    'Count': mask.sum()
                })
        else:
            # Rule: Feature < Threshold implies Long
            mask = winners[feat] < t
            if mask.sum() < 20: continue
            long_rate = (winners[mask]['outcome'] == 1).mean()
            if long_rate > 0.60:
                best_rules.append({
                    'Feature': feat,
                    'Condition': '<',
                    'Threshold': t,
                    'Predicted': 'LONG',
                    'Win Rate': long_rate,
                    'Count': mask.sum()
                })
                
        # Also check for Short prediction
        if direction == "High -> Long": # Low values implies Short
             mask = winners[feat] < t
             if mask.sum() < 20: continue
             short_rate = (winners[mask]['outcome'] == 2).mean()
             if short_rate > 0.60:
                 best_rules.append({
                    'Feature': feat,
                    'Condition': '<',
                    'Threshold': t,
                    'Predicted': 'SHORT',
                    'Win Rate': short_rate,
                    'Count': mask.sum()
                })
        else: # High values implies Short
             mask = winners[feat] > t
             if mask.sum() < 20: continue
             short_rate = (winners[mask]['outcome'] == 2).mean()
             if short_rate > 0.60:
                 best_rules.append({
                    'Feature': feat,
                    'Condition': '>',
                    'Threshold': t,
                    'Predicted': 'SHORT',
                    'Win Rate': short_rate,
                    'Count': mask.sum()
                })

# Deduplicate and sort rules
if best_rules:
    rules_df = pd.DataFrame(best_rules)
    rules_df = rules_df.sort_values('Win Rate', ascending=False)
    
    print(f"{'Rule':<50} {'Bias':<8} {'Prob':<8} {'Count'}")
    print("-" * 80)
    for _, row in rules_df.head(30).iterrows():
        rule_str = f"{row['Feature']} {row['Condition']} {row['Threshold']:.4f}"
        print(f"{rule_str:<50} {row['Predicted']:<8} {row['Win Rate']*100:.1f}%   {row['Count']}")
else:
    print("No strong directional rules found (threshold > 60%).")
