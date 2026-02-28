import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
import os
import warnings
warnings.filterwarnings('ignore')

N_JOBS = max(1, (os.cpu_count() or 1) - 1)  # leave one core free, minimum 1

# ============================================================================
# TSFRESH FEATURE DISCOVERY FOR RANDOM FOREST
# Pipeline:
#   1. Load raw OHLCV, compute outcome labels
#   2. Build rolling windows (Convert Close to stationary Returns)
#   3. tsfresh.extract_features() on Returns, Volume proxy, High-Low range
#   4. tsfresh.select_features() (FDR controlled selection)
#   5. Variance filter -> Correlation clustering
#   6. Output final Random-Forest-ready feature list
# ============================================================================

RAW_DATA_PATH  = 'data/raw/XAUUSD1.csv'
WINDOW_SIZE    = 30       # bars per rolling window (matches HORIZON)
SUBSAMPLE_N    = 20000    # bars to use (keeps runtime manageable)
HORIZON        = 30       # outcome labeling horizon
TARGET         = 5.0      # dollar TP
STOP           = 1.5      # dollar SL
CORR_THRESHOLD = 0.80     # correlation clustering threshold
STD_MIN        = 0.01     # min std for variance filter

# tsfresh FC parameters: EfficientFCParameters = ~800 features, MinimalFCParameters = ~10
# Use Efficient for thorough discovery, Minimal for a quick test run
FC_PARAMS = EfficientFCParameters()

if __name__ == '__main__':
    print('='*80)
    print('TSFRESH FEATURE DISCOVERY FOR REGIME DETECTION')
    print('='*80)

    # ============================================================================
    # 1. LOAD RAW DATA + LABEL OUTCOMES
    # ============================================================================
    print('\nLoading raw OHLCV data...')
    df = pd.read_csv(RAW_DATA_PATH, sep='\t',
                     names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df = df[['Open','High','Low','Close','TickVol']].copy()
    print(f'Loaded {len(df)} bars')

    mid = len(df) // 2
    df_sub = df.iloc[mid - SUBSAMPLE_N//2 : mid + SUBSAMPLE_N//2].copy().reset_index()
    print(f'Using subsample: {len(df_sub)} bars from index {mid - SUBSAMPLE_N//2}')

    print(f'Labeling outcomes (horizon={HORIZON}, TP=${TARGET}, SL=${STOP})...')
    closes = df_sub['Close'].values
    highs  = df_sub['High'].values
    lows   = df_sub['Low'].values
    n      = len(df_sub)
    outcomes = np.zeros(n, dtype=int)

    for i in range(n - HORIZON):
        entry = closes[i]
        long_hit = short_hit = False
        for j in range(i+1, i+HORIZON+1):
            if not long_hit and highs[j] >= entry + TARGET:
                long_hit = True
                break
            if lows[j] <= entry - STOP:
                break
        if not long_hit:
            for j in range(i+1, i+HORIZON+1):
                if lows[j] <= entry - TARGET:
                    short_hit = True
                    break
                if highs[j] >= entry + STOP:
                    break
        outcomes[i] = 1 if long_hit else (2 if short_hit else 0)

    df_sub['outcome'] = outcomes
    df_sub = df_sub.iloc[:n - HORIZON].copy()
    print(f'Labeled {len(df_sub)} bars | Move rate: {(df_sub["outcome"] != 0).mean()*100:.1f}%')

    # ============================================================================
    # 2. BUILD TSFRESH INPUT FORMAT (Stationary)
    # ============================================================================
    print(f'\nBuilding rolling windows (window={WINDOW_SIZE})...')
    df_sub['hl_range'] = df_sub['High'] - df_sub['Low']
    
    # CRITICAL FIX for Tree Models: Convert raw Close to stationary Returns
    # This prevents the Random Forest from failing if the asset price reaches new all-time highs
    df_sub['Close_Ret'] = df_sub['Close'].pct_change().fillna(0.0) 
    
    series_cols = ['Close_Ret', 'hl_range', 'TickVol']

    records = []
    y_labels = {}
    step = 5
    window_ids = range(0, len(df_sub) - WINDOW_SIZE, step)

    for wid, start in enumerate(window_ids):
        window = df_sub.iloc[start:start + WINDOW_SIZE]
        for t, (_, row) in enumerate(window.iterrows()):
            for col in series_cols:
                records.append({'id': wid, 'time': t, 'kind': col, 'value': row[col]})
        end_idx = start + WINDOW_SIZE
        y_labels[wid] = int(df_sub.iloc[end_idx]['outcome'] != 0)

    ts_df = pd.DataFrame(records)
    y_series = pd.Series(y_labels)
    print(f'Windows: {len(window_ids)} | Tradeable: {y_series.sum()} ({y_series.mean()*100:.1f}%)')

    # ============================================================================
    # 3. TSFRESH FEATURE EXTRACTION
    # ============================================================================
    print(f'\nExtracting tsfresh features (this may take several minutes)...')
    print(f'Using EfficientFCParameters (~800 features × {len(series_cols)} series)')

    X_tsfresh = extract_features(
        ts_df,
        column_id='id',
        column_sort='time',
        column_kind='kind',
        column_value='value',
        default_fc_parameters=FC_PARAMS,
        n_jobs=N_JOBS,
        show_warnings=False,
        disable_progressbar=False,
    )

    impute(X_tsfresh)
    print(f'Extracted {X_tsfresh.shape[1]} features from {X_tsfresh.shape[0]} windows')

    # ============================================================================
    # 4. TSFRESH FEATURE SELECTION (FDR-controlled)
    # ============================================================================
    print(f'\nSelecting relevant features (FDR-controlled statistical tests)...')
    y_aligned = y_series.loc[X_tsfresh.index]
    X_selected = select_features(X_tsfresh, y_aligned, fdr_level=0.05, n_jobs=N_JOBS)
    print(f'Selected {X_selected.shape[1]} / {X_tsfresh.shape[1]} features')

    if X_selected.shape[1] == 0:
        print('ERROR: No features selected. Try relaxing fdr_level to 0.10')
        exit(1)

    selected_names = X_selected.columns.tolist()
    print(f'\nTop selected features (first 30):')
    for f in selected_names[:30]:
        print(f'  {f}')

    # ============================================================================
    # 5. VARIANCE FILTER
    # ============================================================================
    print('\n' + '='*80)
    print('5. VARIANCE FILTER')
    print('='*80)
    stds = X_selected.std()
    low_var = stds[stds < STD_MIN].index.tolist()
    print(f'Dropped (std < {STD_MIN}): {len(low_var)}')
    survivors = [f for f in selected_names if f not in low_var]
    print(f'Survivors: {len(survivors)}')
    X_surv = X_selected[survivors]

    # ============================================================================
    # 6. CORRELATION CLUSTERING
    # ============================================================================
    print('\n' + '='*80)
    print(f'6. CORRELATION CLUSTERING (threshold={CORR_THRESHOLD})')
    print('='*80)
    corr_abs = X_surv.corr().abs()
    remaining = list(survivors)
    clusters = []
    visited = set()

    for feat in remaining:
        if feat in visited:
            continue
        cluster = [feat]
        visited.add(feat)
        for other in remaining:
            if other in visited:
                continue
            if any(corr_abs.loc[m, other] >= CORR_THRESHOLD for m in cluster):
                cluster.append(other)
                visited.add(other)
        clusters.append(cluster)

    print(f'Found {len(clusters)} clusters from {len(survivors)} features')
    cluster_selected = []
    for cluster in clusters:
        if len(cluster) == 1:
            cluster_selected.append(cluster[0])
            continue
        avg_corr = corr_abs.loc[cluster, survivors].mean(axis=1)
        winner = avg_corr.idxmin()
        cluster_selected.append(winner)
    print(f'After clustering: {len(cluster_selected)} features')

    # ============================================================================
    # 7. FINAL OUTPUT
    # ============================================================================
    print('\n' + '='*80)
    print('7. FINAL RESULTS')
    print('='*80)
    print(f'\nPipeline summary:')
    print(f'  tsfresh extracted:          {X_tsfresh.shape[1]:>6} features')
    print(f'  After FDR selection:        {len(selected_names):>6} features')
    print(f'  After variance filter:      {len(survivors):>6} features')
    print(f'  After correlation cluster:  {len(cluster_selected):>6} features')
    
    final_features = cluster_selected

    print(f'\nFinal {len(final_features)} features for regime_detection.py:')
    for i, f in enumerate(final_features, 1):
        print(f'  {i:2d}. {f}')

    print(f'\nCopy this into regime_detection.py -> REGIME_FEATURES:')
    print(f'REGIME_FEATURES = {final_features}')

    with open('ml_test/tsfresh_regime_features.txt', 'w') as f:
        f.write('# tsfresh-discovered regime features (Optimized for Random Forest)\n')
        f.write('# Pipeline: extract -> FDR select -> variance -> corr cluster\n\n')
        f.write(f'REGIME_FEATURES = {final_features}\n')
    print(f'\nSaved to: ml_test/tsfresh_regime_features.txt')

    print('\n' + '='*80)
    print('TSFRESH FEATURE DISCOVERY COMPLETE')
    print('='*80)
