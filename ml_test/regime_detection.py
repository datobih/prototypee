import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# REGIME DETECTION — Data-driven approach
# Step 1: BayesianGaussianMixture (DPGMM) — learns k from data automatically
# Step 2: GaussianHMM with effective k — adds temporal memory/smoothing
# Features: validated low-multicollinearity subset (from regime_feature_analysis.py)
#
# Prerequisites:
#   1. engineer_features.py  — builds tsfresh_features_cache.parquet
#   2. regime_feature_analysis.py — selects REGIME_FEATURES from the cache
# ============================================================================

DATA_PATH = 'data/processed/tsfresh_features_cache.parquet'
DPGMM_MAX_K = 5         # Upper bound — DPGMM will prune unused components
WEIGHT_THRESHOLD = 0.05 # Components with weight < 5% are considered inactive

REGIME_FEATURES = ['TickVol__benford_correlation', 'hl_range__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2', 'TickVol__fft_coefficient__attr_"abs"__coeff_12', 'hl_range__number_crossing_m__m_1', 'TickVol__fft_coefficient__attr_"abs"__coeff_13', 'hl_range__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"', 'Close__fft_coefficient__attr_"abs"__coeff_9', 'Close__fft_coefficient__attr_"abs"__coeff_6', 'hl_range__fft_coefficient__attr_"abs"__coeff_9', 'Close__fft_coefficient__attr_"abs"__coeff_5', 'hl_range__fft_coefficient__attr_"abs"__coeff_7', 'TickVol__fft_coefficient__attr_"abs"__coeff_14', 'hl_range__fft_coefficient__attr_"abs"__coeff_10', 'Close__variance_larger_than_standard_deviation', 'Close__fft_coefficient__attr_"abs"__coeff_3', 'TickVol__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)', 'Close__fft_coefficient__attr_"abs"__coeff_4', 'Close__fft_coefficient__attr_"abs"__coeff_7', 'hl_range__fft_coefficient__attr_"abs"__coeff_12', 'TickVol__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"', 'hl_range__fft_coefficient__attr_"abs"__coeff_14', 'Close__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"', 'hl_range__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)', 'hl_range__fft_coefficient__attr_"abs"__coeff_11', 'Close__spkt_welch_density__coeff_5', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0', 'hl_range__fft_coefficient__attr_"abs"__coeff_13', 'Close__cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)', 'hl_range__fft_coefficient__attr_"abs"__coeff_8', 'hl_range__spkt_welch_density__coeff_8', 'hl_range__max_langevin_fixed_point__m_3__r_30', 'TickVol__fft_coefficient__attr_"abs"__coeff_15', 'Close__fft_coefficient__attr_"abs"__coeff_15', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6', 'Close__max_langevin_fixed_point__m_3__r_30', 'hl_range__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"', 'hl_range__fft_coefficient__attr_"abs"__coeff_15', 'Close__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4', 'Close__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"', 'Close__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6', 'Close__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2', 'TickVol__max_langevin_fixed_point__m_3__r_30', 'Close__friedrich_coefficients__coeff_3__m_3__r_30', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8', 'hl_range__friedrich_coefficients__coeff_3__m_3__r_30', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4', 'Close__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"', 'hl_range__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8', 'hl_range__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)', 'TickVol__autocorrelation__lag_1', 'hl_range__ar_coefficient__coeff_0__k_10', 'TickVol__has_duplicate', 'TickVol__lempel_ziv_complexity__bins_100', 'Close__fourier_entropy__bins_3', 'Close__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6', 'Close__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2', 'hl_range__index_mass_quantile__q_0.8', 'hl_range__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0', 'TickVol__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2', 'TickVol__longest_strike_below_mean', 'TickVol__index_mass_quantile__q_0.9', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0', 'TickVol__fft_aggregated__aggtype_"skew"', 'TickVol__ar_coefficient__coeff_0__k_10', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2', 'hl_range__percentage_of_reoccurring_values_to_all_values', 'Close__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4', 'TickVol__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6', 'hl_range__energy_ratio_by_chunks__num_segments_10__segment_focus_9', 'TickVol__fft_coefficient__attr_"real"__coeff_13', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6', 'TickVol__autocorrelation__lag_4', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_0', 'TickVol__autocorrelation__lag_2', 'hl_range__variance_larger_than_standard_deviation', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_1', 'hl_range__partial_autocorrelation__lag_4', 'TickVol__linear_trend__attr_"pvalue"', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2', 'hl_range__index_mass_quantile__q_0.9', 'hl_range__autocorrelation__lag_6', 'Close__ar_coefficient__coeff_7__k_10', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8', 'Close__lempel_ziv_complexity__bins_2', 'hl_range__fft_coefficient__attr_"real"__coeff_12', 'hl_range__ar_coefficient__coeff_6__k_10', 'TickVol__sum_of_reoccurring_data_points', 'TickVol__lempel_ziv_complexity__bins_3', 'Close__agg_autocorrelation__f_agg_"mean"__maxlag_40', 'TickVol__ar_coefficient__coeff_1__k_10', 'hl_range__fft_coefficient__attr_"real"__coeff_13', 'TickVol__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_4', 'TickVol__lempel_ziv_complexity__bins_5', 'TickVol__symmetry_looking__r_0.1', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_2', 'hl_range__longest_strike_above_mean', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4', 'TickVol__lempel_ziv_complexity__bins_2', 'TickVol__friedrich_coefficients__coeff_3__m_3__r_30', 'Close__number_cwt_peaks__n_5', 'TickVol__autocorrelation__lag_7', 'TickVol__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0', 'Close__partial_autocorrelation__lag_8', 'hl_range__permutation_entropy__dimension_5__tau_1']

import pickle, os

print('='*80)
print('REGIME DETECTION — DPGMM (data-driven k) + HMM (temporal smoothing)')
print('='*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print('\nLoading feature data...')
if not os.path.exists(DATA_PATH):
    print(f'ERROR: Cache not found: {DATA_PATH}')
    print('Run engineer_features.py first.')
    exit(1)
df = pd.read_parquet(DATA_PATH)
print(f'Loaded {len(df)} bars')

missing = [f for f in REGIME_FEATURES if f not in df.columns]
if missing:
    print(f'ERROR: Missing features: {missing}')
    print('Run engineer_features.py then regime_feature_analysis.py first.')
    exit(1)

# ============================================================================
# FEATURE PREPARATION
# ============================================================================
X = df[REGIME_FEATURES].dropna().copy()
for col in REGIME_FEATURES:
    lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = X[col].clip(lo, hi)

print(f'Clean samples: {len(X)} | Features: {len(REGIME_FEATURES)}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# STEP 1: BAYESIAN GAUSSIAN MIXTURE (DPGMM)
# Learns the effective number of regimes from data
# ============================================================================
print(f'\n--- STEP 1: BAYESIAN GAUSSIAN MIXTURE (max k={DPGMM_MAX_K}) ---')
print('Fitting... (this may take ~30s)')

dpgmm = BayesianGaussianMixture(
    n_components=DPGMM_MAX_K,
    covariance_type='diag',
    weight_concentration_prior=1.0 / DPGMM_MAX_K,  # sparse prior — encourages pruning
    max_iter=500,
    n_init=3,
    random_state=42,
)
dpgmm.fit(X_scaled)

# Find effective k: components with weight above threshold
weights = dpgmm.weights_
active_mask = weights >= WEIGHT_THRESHOLD
effective_k = active_mask.sum()
active_indices = np.where(active_mask)[0]

print(f'\nComponent weights (threshold={WEIGHT_THRESHOLD}):')
for i, w in enumerate(weights):
    status = 'ACTIVE' if w >= WEIGHT_THRESHOLD else 'pruned'
    print(f'  Component {i:2d}: weight={w:.4f}  [{status}]')

print(f'\nEffective number of regimes: {effective_k}')
print(f'Active components: {active_indices.tolist()}')

# Remap DPGMM labels to 0..effective_k-1
dpgmm_labels_raw = dpgmm.predict(X_scaled)
label_remap = {old: new for new, old in enumerate(active_indices)}
# Bars assigned to pruned components get nearest active component
dpgmm_probs = dpgmm.predict_proba(X_scaled)
active_probs = dpgmm_probs[:, active_indices]
dpgmm_labels = active_probs.argmax(axis=1)  # 0..effective_k-1

# ============================================================================
# STEP 2: GAUSSIAN HMM (temporal smoothing)
# Uses effective_k from DPGMM, adds transition memory
# ============================================================================
print(f'\n--- STEP 2: GAUSSIAN HMM (k={effective_k}, temporal smoothing) ---')
print('Fitting...')

hmm = GaussianHMM(
    n_components=effective_k,
    covariance_type='diag',
    n_iter=500,
    init_params='',   # disable auto-init so warm-start is used
    random_state=42,
)

# Warm-start HMM means/covars from DPGMM (better initialisation)
hmm.startprob_ = np.ones(effective_k) / effective_k
diag_val = 0.99
off_diag = (1.0 - diag_val) / max(effective_k - 1, 1)
transmat = np.full((effective_k, effective_k), off_diag)
np.fill_diagonal(transmat, diag_val)  # strong persistence prior — forces longer regimes
hmm.transmat_ = transmat
hmm.means_ = dpgmm.means_[active_indices]
hmm.covars_ = dpgmm.covariances_[active_indices]

hmm.fit(X_scaled)
hmm_labels = hmm.predict(X_scaled)

# Transition matrix
print(f'\nHMM Transition Matrix (rows=from, cols=to):')
print(f'{"":>12}' + ''.join(f'  R{j:<3}' for j in range(effective_k)))
for i in range(effective_k):
    row = ''.join(f'{hmm.transmat_[i,j]:6.3f}' for j in range(effective_k))
    print(f'  Regime {i:<3}  {row}')

# Average regime duration (bars)
print(f'\nExpected regime duration (bars):')
for i in range(effective_k):
    p_stay = hmm.transmat_[i, i]
    avg_duration = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float('inf')
    print(f'  Regime {i}: p_stay={p_stay:.3f}  avg_duration={avg_duration:.1f} bars')

# ============================================================================
# REGIME CHARACTERIZATION
# ============================================================================
print(f'\n--- REGIME CHARACTERIZATION ---')
# Show first 5 features in summary table to keep output readable
DISPLAY_FEATURES = REGIME_FEATURES[:5]
col_header = '  '.join(f'{f.split("__")[0][:14]:>14}' for f in DISPLAY_FEATURES)
print(f'{"Regime":<10} {"Count":>8} {"Pct":>6}  {col_header}')
print('-' * (10 + 8 + 6 + 4 + len(DISPLAY_FEATURES) * 16))

regime_names = {}
regime_stats = {}

for r in range(effective_k):
    mask = hmm_labels == r
    subset = X.iloc[mask]
    count = mask.sum()
    pct = count / len(X) * 100
    means = subset[REGIME_FEATURES].mean()
    regime_stats[r] = {'count': count, 'pct': pct, 'means': means}

    vals = '  '.join(f'{means[f]:>14.3f}' for f in DISPLAY_FEATURES)
    print(f'Regime {r:<4} {count:>8} {pct:>5.1f}%  {vals}')

    # Auto-name based on relative size (largest = most active, smallest = quietest)
    regime_names[r] = f'REGIME_{r}'

# Rename by relative count: largest regime = DOMINANT, smallest = RARE
counts = {r: regime_stats[r]['count'] for r in range(effective_k)}
sorted_r = sorted(counts, key=counts.get, reverse=True)
for rank, r in enumerate(sorted_r):
    if rank == 0:
        regime_names[r] = 'DOMINANT'
    elif rank == len(sorted_r) - 1:
        regime_names[r] = 'RARE'
    else:
        regime_names[r] = f'REGIME_{rank}'

print(f'\n--- AUTO-NAMED REGIMES ---')
for r, name in regime_names.items():
    print(f'  Regime {r} -> {name:<20}  count={regime_stats[r]["count"]:>6}  pct={regime_stats[r]["pct"]:>5.1f}%')

# ============================================================================
# STRATEGY PERFORMANCE BY REGIME
# ============================================================================
X['regime'] = hmm_labels
X['regime_name'] = X['regime'].map(regime_names)
df_regimes = df.loc[X.index].copy()
df_regimes['regime'] = hmm_labels
df_regimes['regime_name'] = X['regime_name'].values

if 'outcome' in df_regimes.columns:
    print(f'\n--- BASE OUTCOME RATES BY REGIME ---')
    print(f'{"Regime":<22} {"Count":>8} {"Move%":>7} {"Long%":>7} {"Short%":>7}')
    print('-'*55)
    for r in range(effective_k):
        sub = df_regimes[df_regimes['regime'] == r]
        if len(sub) == 0:
            continue
        move_pct = (sub['outcome'] != 0).mean() * 100
        long_pct = (sub['outcome'] == 1).mean() * 100
        short_pct = (sub['outcome'] == 2).mean() * 100
        print(f'{regime_names[r]:<22} {len(sub):>8} {move_pct:>6.1f}% {long_pct:>6.1f}% {short_pct:>6.1f}%')

if 'p_move' in df_regimes.columns:
    print(f'\n--- RF SIGNAL QUALITY BY REGIME (p_move >= 0.70) ---')
    print(f'{"Regime":<22} {"Signals":>8} {"WR%":>7} {"Long_WR%":>9} {"Short_WR%":>10}')
    print('-'*62)
    for r in range(effective_k):
        sub = df_regimes[(df_regimes['regime'] == r) & (df_regimes['p_move'] >= 0.70)]
        if len(sub) < 5:
            continue
        wr = (sub['outcome'] != 0).mean() * 100
        long_wr = (sub['outcome'] == 1).mean() * 100
        short_wr = (sub['outcome'] == 2).mean() * 100
        print(f'{regime_names[r]:<22} {len(sub):>8} {wr:>6.1f}% {long_wr:>8.1f}% {short_wr:>9.1f}%')

if 'p_move' in df_regimes.columns:
    print(f'\n--- DIRECTIONAL ENTRY QUALITY BY REGIME ---')
    print(f'(p_move>=0.70, RF direction only)')
    print(f'{"Regime":<22} {"LONG_n":>7} {"LONG_WR%":>9} {"SHORT_n":>8} {"SHORT_WR%":>10}')
    print('-'*62)
    for r in range(effective_k):
        rm = df_regimes['regime'] == r
        pm = df_regimes['p_move'] >= 0.70
        lm = rm & pm & (df_regimes['p_long'] > df_regimes['p_short'])
        sm = rm & pm & (df_regimes['p_short'] > df_regimes['p_long'])
        ls, ss = df_regimes[lm], df_regimes[sm]
        lwr = (ls['outcome'] == 1).mean() * 100 if len(ls) > 0 else 0
        swr = (ss['outcome'] == 2).mean() * 100 if len(ss) > 0 else 0
        print(f'{regime_names[r]:<22} {len(ls):>7} {lwr:>8.1f}% {len(ss):>8} {swr:>9.1f}%')

# ============================================================================
# SAVE
# ============================================================================
print(f'\n--- SAVING ---')
out_path = 'data/processed/XAUUSD1_regimes.csv'
save_cols = ['regime', 'regime_name', 'outcome', 'p_move', 'p_long', 'p_short']
save_cols = [c for c in save_cols if c in df_regimes.columns] + REGIME_FEATURES
df_regimes[save_cols].to_csv(out_path)
print(f'Saved: {out_path}')

os.makedirs('models', exist_ok=True)
with open('models/regime_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/regime_dpgmm.pkl', 'wb') as f:
    pickle.dump(dpgmm, f)
with open('models/regime_hmm.pkl', 'wb') as f:
    pickle.dump(hmm, f)
with open('models/regime_names.pkl', 'wb') as f:
    pickle.dump(regime_names, f)
with open('models/regime_feature_names.txt', 'w') as f:
    f.write('\n'.join(REGIME_FEATURES))
print(f'Saved: models/regime_scaler.pkl, regime_dpgmm.pkl, regime_hmm.pkl')
print(f'Effective regimes discovered: {effective_k}')

print('\n' + '='*80)
print('REGIME DETECTION COMPLETE')
print('='*80)
