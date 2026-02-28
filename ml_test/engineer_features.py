"""
engineer_features.py
====================
Standalone feature engineering script.

Loads raw XAUUSD1 OHLCV, runs tsfresh rolling-window extraction (window=30,
stride=5), and saves the full feature matrix to a parquet cache.

Run this ONCE before hedging_strategy_strict.py or regime_detection.py.
Subsequent scripts load the parquet directly — no re-extraction needed.

Output: data/processed/tsfresh_features_cache.parquet
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
RAW_DATA   = 'data/raw/XAUUSD1.csv'
CACHE_PATH = 'data/processed/tsfresh_features_cache.parquet'
WINDOW_SIZE = 30
STRIDE      = 1
N_JOBS      = max(1, (os.cpu_count() or 1) - 1)

# Discovered feature names from tsfresh_regime_features.txt.
# tsfresh's from_columns() parses these back into FC params automatically —
# no manual parameter specification needed.
DISCOVERED_FEATURES = ['TickVol__benford_correlation', 'hl_range__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2', 'TickVol__fft_coefficient__attr_"abs"__coeff_12', 'hl_range__number_crossing_m__m_1', 'TickVol__fft_coefficient__attr_"abs"__coeff_13', 'hl_range__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"', 'Close__fft_coefficient__attr_"abs"__coeff_9', 'Close__fft_coefficient__attr_"abs"__coeff_6', 'hl_range__fft_coefficient__attr_"abs"__coeff_9', 'Close__fft_coefficient__attr_"abs"__coeff_5', 'hl_range__fft_coefficient__attr_"abs"__coeff_7', 'TickVol__fft_coefficient__attr_"abs"__coeff_14', 'hl_range__fft_coefficient__attr_"abs"__coeff_10', 'Close__variance_larger_than_standard_deviation', 'Close__fft_coefficient__attr_"abs"__coeff_3', 'TickVol__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)', 'Close__fft_coefficient__attr_"abs"__coeff_4', 'Close__fft_coefficient__attr_"abs"__coeff_7', 'hl_range__fft_coefficient__attr_"abs"__coeff_12', 'TickVol__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"', 'hl_range__fft_coefficient__attr_"abs"__coeff_14', 'Close__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"', 'hl_range__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)', 'hl_range__fft_coefficient__attr_"abs"__coeff_11', 'Close__spkt_welch_density__coeff_5', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0', 'hl_range__fft_coefficient__attr_"abs"__coeff_13', 'Close__cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)', 'hl_range__fft_coefficient__attr_"abs"__coeff_8', 'hl_range__spkt_welch_density__coeff_8', 'hl_range__max_langevin_fixed_point__m_3__r_30', 'TickVol__fft_coefficient__attr_"abs"__coeff_15', 'Close__fft_coefficient__attr_"abs"__coeff_15', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6', 'Close__max_langevin_fixed_point__m_3__r_30', 'hl_range__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"', 'hl_range__fft_coefficient__attr_"abs"__coeff_15', 'Close__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4', 'Close__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"', 'Close__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6', 'Close__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2', 'TickVol__max_langevin_fixed_point__m_3__r_30', 'Close__friedrich_coefficients__coeff_3__m_3__r_30', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8', 'hl_range__friedrich_coefficients__coeff_3__m_3__r_30', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0', 'TickVol__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4', 'Close__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"', 'hl_range__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8', 'hl_range__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)', 'TickVol__autocorrelation__lag_1', 'hl_range__ar_coefficient__coeff_0__k_10', 'TickVol__has_duplicate', 'TickVol__lempel_ziv_complexity__bins_100', 'Close__fourier_entropy__bins_3', 'Close__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6', 'Close__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2', 'hl_range__index_mass_quantile__q_0.8', 'hl_range__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0', 'TickVol__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2', 'TickVol__longest_strike_below_mean', 'TickVol__index_mass_quantile__q_0.9', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0', 'TickVol__fft_aggregated__aggtype_"skew"', 'TickVol__ar_coefficient__coeff_0__k_10', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2', 'hl_range__percentage_of_reoccurring_values_to_all_values', 'Close__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4', 'TickVol__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6', 'hl_range__energy_ratio_by_chunks__num_segments_10__segment_focus_9', 'TickVol__fft_coefficient__attr_"real"__coeff_13', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6', 'TickVol__autocorrelation__lag_4', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_0', 'TickVol__autocorrelation__lag_2', 'hl_range__variance_larger_than_standard_deviation', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_1', 'hl_range__partial_autocorrelation__lag_4', 'TickVol__linear_trend__attr_"pvalue"', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2', 'hl_range__index_mass_quantile__q_0.9', 'hl_range__autocorrelation__lag_6', 'Close__ar_coefficient__coeff_7__k_10', 'TickVol__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8', 'Close__lempel_ziv_complexity__bins_2', 'hl_range__fft_coefficient__attr_"real"__coeff_12', 'hl_range__ar_coefficient__coeff_6__k_10', 'TickVol__sum_of_reoccurring_data_points', 'TickVol__lempel_ziv_complexity__bins_3', 'Close__agg_autocorrelation__f_agg_"mean"__maxlag_40', 'TickVol__ar_coefficient__coeff_1__k_10', 'hl_range__fft_coefficient__attr_"real"__coeff_13', 'TickVol__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_4', 'TickVol__lempel_ziv_complexity__bins_5', 'TickVol__symmetry_looking__r_0.1', 'TickVol__energy_ratio_by_chunks__num_segments_10__segment_focus_2', 'hl_range__longest_strike_above_mean', 'hl_range__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4', 'TickVol__lempel_ziv_complexity__bins_2', 'TickVol__friedrich_coefficients__coeff_3__m_3__r_30', 'Close__number_cwt_peaks__n_5', 'TickVol__autocorrelation__lag_7', 'TickVol__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0', 'Close__partial_autocorrelation__lag_8', 'hl_range__permutation_entropy__dimension_5__tau_1']


def build_tsfresh_input(df):
    """Build tsfresh long-format DataFrame from OHLCV using vectorised numpy."""
    df = df.copy()
    df['hl_range'] = df['High'] - df['Low']

    df_reset = df.reset_index()
    arr_close = df_reset['Close'].to_numpy(dtype=float)
    arr_hl    = df_reset['hl_range'].to_numpy(dtype=float)
    arr_tv    = df_reset['TickVol'].to_numpy(dtype=float)
    arr_dt    = df_reset['Datetime'].to_numpy()

    indices = np.arange(WINDOW_SIZE - 1, len(df), STRIDE)
    n_wins  = len(indices)

    t_vals   = np.tile(np.arange(WINDOW_SIZE), n_wins)
    id_vals  = np.repeat(np.arange(n_wins), WINDOW_SIZE)
    starts   = indices - WINDOW_SIZE + 1
    bar_idx  = (starts[:, None] + np.arange(WINDOW_SIZE)).ravel()
    datetimes = arr_dt[indices]

    ts_df = pd.concat([
        pd.DataFrame({'id': id_vals, 'time': t_vals, 'kind': 'Close',    'value': arr_close[bar_idx]}),
        pd.DataFrame({'id': id_vals, 'time': t_vals, 'kind': 'hl_range', 'value': arr_hl[bar_idx]}),
        pd.DataFrame({'id': id_vals, 'time': t_vals, 'kind': 'TickVol',  'value': arr_tv[bar_idx]}),
    ], ignore_index=True)

    return ts_df, datetimes


if __name__ == '__main__':
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction.settings import from_columns

    if os.path.exists(CACHE_PATH):
        print(f'Cache already exists: {CACHE_PATH}')
        print('Delete it to force re-extraction.')
        import sys; sys.exit(0)

    # Derive FC params directly from the discovered feature names.
    # from_columns() parses the tsfresh column name format back into
    # the exact {calculator: [param_dicts]} structure tsfresh expects.
    fc_params = from_columns(DISCOVERED_FEATURES)
    print(f'Derived {sum(len(v) if v else 1 for v in fc_params.values())} '
          f'calculator configs across {len(fc_params)} calculators')

    print('=' * 70)
    print('TSFRESH FEATURE ENGINEERING')
    print('=' * 70)

    print(f'\nLoading {RAW_DATA}...')
    df = pd.read_csv(RAW_DATA, sep='\t',
                     names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'TickVol']].copy()
    print(f'Loaded {len(df):,} bars')

    print(f'\nBuilding rolling windows (window={WINDOW_SIZE}, stride={STRIDE})...')
    ts_df, datetimes = build_tsfresh_input(df)
    n_wins = len(datetimes)
    print(f'Built {n_wins:,} windows ({len(ts_df):,} rows)')

    print(f'\nRunning tsfresh extraction (n_jobs={N_JOBS})...')
    print('This will take ~20-60 min. Cached after first run.\n')

    X = extract_features(
        ts_df,
        column_id='id',
        column_sort='time',
        column_kind='kind',
        column_value='value',
        kind_to_fc_parameters=fc_params,
        n_jobs=N_JOBS,
        show_warnings=False,
        disable_progressbar=False,
    )
    impute(X)
    print(f'\nExtracted {X.shape[1]} features across {X.shape[0]} windows')

    X.index = pd.DatetimeIndex(datetimes)
    X.index.name = 'Datetime'

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    X.to_parquet(CACHE_PATH)
    print(f'\nSaved: {CACHE_PATH}')
    print('Done. Run hedging_strategy_strict.py or regime_detection.py next.')
