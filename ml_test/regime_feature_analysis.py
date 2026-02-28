import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# REGIME FEATURE ANALYSIS — Fully Automated Discovery
# No hand-picking: loads ALL numeric columns from CSV and filters them down
# Pipeline:
#   1. Load all numeric columns (exclude price/label/meta columns)
#   2. Variance filter — drop near-zero variance features
#   3. Correlation clustering — group r > CORR_THRESHOLD, keep one per cluster
#   4. Iterative VIF elimination — drop highest VIF until all < VIF_MAX
#   5. PCA check — confirm independent dimensions match feature count
#   6. Output final feature set
# ============================================================================

DATA_PATH       = 'data/processed/tsfresh_features_cache.parquet'
STD_MIN         = 0.01   # Drop features with std below this
CORR_THRESHOLD  = 0.80   # Features correlated above this are in the same cluster
VIF_MAX         = 5.0    # Iteratively drop highest VIF until all below this

# Columns to always exclude (non-features)
EXCLUDE_COLS = {
    'Open', 'High', 'Low', 'Close',
    'outcome', 'rf_prob', 'p_long', 'p_short', 'p_move', 'edge',
    'TickVol', 'Vol', 'Spread', 'Date', 'Time',
}

print('='*80)
print('REGIME FEATURE ANALYSIS — AUTOMATED DISCOVERY')
print('='*80)

# ============================================================================
# 1. LOAD ALL NUMERIC FEATURES
# ============================================================================
print('\nLoading data...')
import os
if not os.path.exists(DATA_PATH):
    print(f'ERROR: Cache not found: {DATA_PATH}')
    print('Run engineer_features.py first.')
    import sys; sys.exit(1)
df = pd.read_parquet(DATA_PATH)
print(f'Loaded {len(df)} bars, {len(df.columns)} total columns')

# Keep only numeric columns not in the exclusion list
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
candidates = [c for c in numeric_cols if c not in EXCLUDE_COLS]
print(f'Candidate features after exclusions: {len(candidates)}')

X = df[candidates].dropna()
print(f'Clean samples: {len(X)}')

# Clip outliers at 1/99 percentile
for col in candidates:
    lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = X[col].clip(lo, hi)

# ============================================================================
# 2. VARIANCE FILTER
# ============================================================================
print('\n' + '='*80)
print('2. VARIANCE FILTER')
print('='*80)

stds = X.std()
low_var = stds[stds < STD_MIN].index.tolist()
print(f'Dropped (std < {STD_MIN}): {len(low_var)} features')
if low_var:
    print(f'  {low_var}')

survivors = [f for f in candidates if f not in low_var]
print(f'Survivors after variance filter: {len(survivors)}')

# ============================================================================
# 3. CORRELATION CLUSTERING
# Pick one representative per cluster of highly correlated features
# Representative = feature with lowest average correlation to ALL other features
# (most independent from the rest of the set)
# ============================================================================
print('\n' + '='*80)
print(f'3. CORRELATION CLUSTERING (threshold={CORR_THRESHOLD})')
print('='*80)

corr_abs = X[survivors].corr().abs()

# Build clusters using single-linkage: if any two features in a cluster
# have |r| > threshold, they belong to the same cluster
remaining = list(survivors)
clusters = []
visited = set()

for feat in remaining:
    if feat in visited:
        continue
    # Find all features correlated with this one above threshold
    cluster = [feat]
    visited.add(feat)
    for other in remaining:
        if other in visited:
            continue
        # Check if 'other' is correlated with ANY member of the current cluster
        if any(corr_abs.loc[m, other] >= CORR_THRESHOLD for m in cluster):
            cluster.append(other)
            visited.add(other)
    clusters.append(cluster)

print(f'\nFound {len(clusters)} clusters from {len(survivors)} features:')

selected = []
for i, cluster in enumerate(clusters):
    if len(cluster) == 1:
        winner = cluster[0]
        selected.append(winner)
        continue

    # Winner = lowest average absolute correlation to ALL survivors (most independent)
    avg_corr_to_all = corr_abs.loc[cluster, survivors].mean(axis=1)
    winner = avg_corr_to_all.idxmin()
    selected.append(winner)

    print(f'\n  Cluster {i+1} ({len(cluster)} features):')
    for f in sorted(cluster):
        marker = ' <-- WINNER' if f == winner else ''
        print(f'    {f:<40} avg_corr={avg_corr_to_all[f]:.3f}{marker}')

print(f'\nSelected {len(selected)} features (one per cluster)')

# ============================================================================
# 4. ITERATIVE VIF ELIMINATION
# ============================================================================
print('\n' + '='*80)
print(f'4. ITERATIVE VIF ELIMINATION (max VIF={VIF_MAX})')
print('='*80)

scaler = StandardScaler()
current = list(selected)
iteration = 0

while True:
    iteration += 1
    X_scaled = scaler.fit_transform(X[current])
    X_df = pd.DataFrame(X_scaled, columns=current)

    vifs = {col: variance_inflation_factor(X_df.values, i)
            for i, col in enumerate(current)}

    max_feat = max(vifs, key=vifs.get)
    max_vif = vifs[max_feat]

    if max_vif <= VIF_MAX:
        print(f'Iteration {iteration}: All VIF <= {VIF_MAX}. Done.')
        break

    print(f'Iteration {iteration}: Drop {max_feat} (VIF={max_vif:.2f})')
    current.remove(max_feat)

    if len(current) < 2:
        print('Too few features remaining — stopping.')
        break

final_features = current

print(f'\nFinal VIF scores:')
print(f'{"Feature":<40} {"VIF":>8}')
print('-'*50)
X_scaled_final = scaler.fit_transform(X[final_features])
X_df_final = pd.DataFrame(X_scaled_final, columns=final_features)
for i, col in enumerate(final_features):
    vif = variance_inflation_factor(X_df_final.values, i)
    flag = ' * moderate' if vif > 2 else ''
    print(f'{col:<40} {vif:>8.2f}{flag}')

# ============================================================================
# 5. PCA EXPLAINED VARIANCE CHECK
# ============================================================================
print('\n' + '='*80)
print('5. PCA EXPLAINED VARIANCE CHECK')
print('='*80)
print('Confirms how many truly independent dimensions remain')

pca = PCA(random_state=42)
pca.fit(X_scaled_final)
cumvar = np.cumsum(pca.explained_variance_ratio_)

print(f'\n{"PC":<6} {"Var%":>8} {"Cumulative%":>12}')
print('-'*30)
n_90 = None
for i, (v, c) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
    print(f'PC{i+1:<4} {v*100:>7.1f}% {c*100:>11.1f}%')
    if n_90 is None and c >= 0.90:
        n_90 = i + 1

print(f'\n  {n_90} PCs explain 90% of variance across {len(final_features)} features')
if n_90 == len(final_features):
    print('  All features contribute independent information.')
else:
    print(f'  {len(final_features) - n_90} feature(s) may still be partially redundant.')

# ============================================================================
# 6. FINAL OUTPUT
# ============================================================================
print('\n' + '='*80)
print('6. FINAL RECOMMENDED FEATURE SET FOR REGIME CLUSTERING')
print('='*80)

print(f'\nStarted with: {len(candidates)} features')
print(f'After variance filter: {len(survivors)}')
print(f'After correlation clustering: {len(selected)}')
print(f'After VIF elimination: {len(final_features)}')

print(f'\nFinal {len(final_features)} features:')
for i, f in enumerate(final_features, 1):
    X_scaled_final = scaler.fit_transform(X[final_features])
    X_df_final = pd.DataFrame(X_scaled_final, columns=final_features)
    vif = variance_inflation_factor(X_df_final.values, final_features.index(f))
    print(f'  {i:2d}. {f:<40} VIF={vif:.2f}')

print(f'\nCopy this into regime_detection.py -> REGIME_FEATURES:')
print(f'REGIME_FEATURES = {final_features}')

print('\n' + '='*80)
print('FEATURE ANALYSIS COMPLETE')
print('='*80)
