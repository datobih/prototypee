import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import sys, os

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
TARGET = 2        # Target profit in dollars
STOP = 0.5         # Stop loss in dollars
HORIZON = 30       # Number of bars to look ahead for outcome
RF_THRESHOLD = 0.70  # Random Forest probability threshold for hedge entry

RAW_DATA    = 'data/raw/XAUUSD1.csv'
WINDOW_SIZE = 30  # Needs to match tsfresh discovery

# The 21 Features discovered by tsfresh FDR and Correlation Clustering
REGIME_FEATURES = ['hl_range__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6', 'hl_range__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2', 'hl_range__max_langevin_fixed_point__m_3__r_30', 'hl_range__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0', 'hl_range__range_count__max_1__min_-1', 'hl_range__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2', 'hl_range__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"', 'hl_range__fft_coefficient__attr_"abs"__coeff_7', 'hl_range__fft_coefficient__attr_"abs"__coeff_3', 'hl_range__number_crossing_m__m_1', 'hl_range__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"', 'TickVol__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2', 'TickVol__fft_coefficient__attr_"abs"__coeff_6', 'TickVol__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)', 'hl_range__autocorrelation__lag_9', 'hl_range__fft_coefficient__attr_"abs"__coeff_14', 'hl_range__variance_larger_than_standard_deviation', 'hl_range__fft_coefficient__attr_"abs"__coeff_9', 'hl_range__fft_coefficient__attr_"abs"__coeff_6', 'TickVol__spkt_welch_density__coeff_2', 'TickVol__fft_coefficient__attr_"abs"__coeff_7']

def label_outcomes(df, horizon=15, target=5.0, stop=2.5):
    """Label outcomes using fixed dollar SL/TP (not percentage-based)"""
    df = df.copy()
    outcomes = []
    
    for i in range(len(df) - horizon):
        if i % 20000 == 0:
            print(f'  Labeling {i}/{len(df)-horizon}...')
        
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]
        
        # Check LONG (fixed dollar targets)
        long_hit = False
        for h, l in zip(future['High'], future['Low']):
            if h >= entry + target:  # TP hit
                long_hit = True
                break
            if l <= entry - stop:  # SL hit
                break
        
        # Check SHORT (fixed dollar targets)
        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry - target:  # TP hit
                    short_hit = True
                    break
                if h >= entry + stop:  # SL hit
                    break
        
        outcomes.append(1 if long_hit else (2 if short_hit else 0))
    
    df = df.iloc[:len(outcomes)].copy()
    df['outcome'] = outcomes
    return df

# =============================================================================
# LOAD RAW OHLC + TSFRESH FEATURES
# =============================================================================

if __name__ == '__main__':
    print('='*80)
    print('XAUUSD 1MIN HEDGE STRATEGY (TSFRESH FEATURES)')
    print('='*80)

    print('\nLoading raw OHLC data...')
    df = pd.read_csv(RAW_DATA, sep='\t',
                     names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df = df[['Open','High','Low','Close', 'TickVol']].copy()
    print(f'Loaded {len(df):,} bars')

    print(f'\nEngineering specific tsfresh input columns...')
    df['hl_range'] = df['High'] - df['Low']
    df['Close_Ret'] = df['Close'].pct_change().fillna(0.0)

    # Build windows to extract ONLY the 21 regime features
    print(f'\nBuilding rolling feature windows ({WINDOW_SIZE} bars)...')
    records = []
    step = 5  # We stride by 5 to create samples without computing every single bar perfectly
    window_ids = range(0, len(df) - WINDOW_SIZE, step)

    # We only need enough data to test the model; let's limit to 20k samples to keep execution fast
    SUBSAMPLE_N = 20000
    mid = len(df) // 2
    df_sub = df.iloc[mid - SUBSAMPLE_N//2 : mid + SUBSAMPLE_N//2].copy().reset_index()
    window_ids = range(0, len(df_sub) - WINDOW_SIZE, step)

    series_cols = ['Close_Ret', 'hl_range', 'TickVol']

    for wid, start in enumerate(window_ids):
        window = df_sub.iloc[start:start + WINDOW_SIZE]
        for t, (_, row) in enumerate(window.iterrows()):
            for col in series_cols:
                records.append({'id': wid, 'time': t, 'kind': col, 'value': row[col]})

    ts_df = pd.DataFrame(records)

    print(f'\nDynamic tsfresh extraction ({len(window_ids)} samples)...')
    ts_feat = extract_features(
        ts_df,
        column_id='id',
        column_sort='time',
        column_kind='kind',
        column_value='value',
        kind_to_fc_parameters=None, # In a prod environment, we'd pass a specific dictionary to calculate ONLY REGIME_FEATURES
        n_jobs=max(1, (os.cpu_count() or 1) - 1),
        show_warnings=False
    )

    impute(ts_feat)

    # Strictly isolate the exact 21 features we want to train the model on
    ts_feat = ts_feat[[col for col in REGIME_FEATURES if col in ts_feat.columns]]

    # Map extracted features back to the main dataframe
    df_final = []
    for wid, start in enumerate(window_ids):
        end_idx = start + WINDOW_SIZE - 1
        # Grab the OHLC row that corresponds to the END of the rolling window
        row = df_sub.iloc[end_idx].to_dict()
        
        # Grab the tsfresh features calculated for this exact window
        ts_row = ts_feat.loc[wid].to_dict()
        
        # Merge
        row.update(ts_row)
        df_final.append(row)

    df = pd.DataFrame(df_final)
    df.set_index('Datetime', inplace=True)

    print(f'Ready: {len(df):,} bars with {ts_feat.shape[1]} optimal regime features + OHLC')

    all_features = list(ts_feat.columns)

    print(f'\nLabeling outcomes ({HORIZON} bars, ${TARGET} target, ${STOP} stop)...')
    df = label_outcomes(df, HORIZON, TARGET, STOP)

    # Split 60:40
    split = int(len(df) * 0.6)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    print(f'\nTrain set: {len(train):,} bars (60%)')
    print(f'Test set:  {len(test):,} bars (40%)')
    print(f'Base success rate: {(test["outcome"] != 0).sum() / len(test) * 100:.1f}%')

    # =============================================================================
    # TRAIN MODELS (tsfresh features only)
    # =============================================================================
    print('\n' + '='*80)
    print(f'TRAINING MODELS ({len(all_features)} tsfresh features)')
    print('='*80)

    train_safe = train['outcome'] != 0
    safe_trades = test['outcome'] != 0

    X_train = train[all_features].fillna(0)
    y_train = train_safe.astype(int)
    X_test = test[all_features].fillna(0)
    y_test = safe_trades.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f'\nTraining on {len(X_train):,} samples with {len(all_features)} features')
    print(f'Testing on {len(X_test):,} samples')

    # 1. Logistic Regression
    print('\n--- LOGISTIC REGRESSION ---')
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)
    print(f'AUC-ROC: {lr_auc:.4f}')

    feature_importance = sorted(zip(all_features, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print(f'\nTop 10 features by coefficient:')
    for feat, coef in feature_importance[:10]:
        print(f'  {feat:<55} {coef:>8.4f}')

    print(f'\nFiltered trading results by probability threshold:')
    print(f'{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count"}')
    print('-'*60)
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        filtered = test[lr_probs >= threshold]
        if len(filtered) > 0:
            success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
            frequency = len(filtered) / len(test) * 100
            print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d}')

    # 2. Random Forest
    print('\n--- RANDOM FOREST ---')
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)
    print(f'AUC-ROC: {rf_auc:.4f}')

    feature_importance_rf = sorted(zip(all_features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f'\nTop 15 features by importance:')
    for feat, importance in feature_importance_rf[:15]:
        print(f'  {feat:<55} {importance:>8.4f}')

    print(f'\nFiltered trading results by probability threshold:')
    print(f'{"Threshold":<12} {"Success %":<12} {"Frequency %":<12} {"Count"}')
    print('-'*60)
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        filtered = test[rf_probs >= threshold]
        if len(filtered) > 0:
            success_rate = (filtered['outcome'] != 0).sum() / len(filtered) * 100
            frequency = len(filtered) / len(test) * 100
            print(f'{threshold:<12.2f} {success_rate:>11.1f} {frequency:>11.2f} {len(filtered):>11d}')

    # Highlight RF >= 0.75 specifically
    print(f'\n*** RF >= 0.75 RESULTS ***')
    filtered_75 = test[rf_probs >= 0.75]
    if len(filtered_75) > 0:
        success_75 = (filtered_75['outcome'] != 0).sum() / len(filtered_75) * 100
        print(f'Success Rate: {success_75:.1f}%')
        print(f'Trade Count: {len(filtered_75)}')
        print(f'Frequency: {len(filtered_75) / len(test) * 100:.2f}%')

    # Save probabilities to test set for further analysis
    test['lr_prob'] = lr_probs
    test['rf_prob'] = rf_probs

    # Save models
    print('\n' + '='*80)
    print('SAVING MODELS')
    print('='*80)

    import pickle
    os.makedirs('models', exist_ok=True)

    with open('models/hedge_ts_rf.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print('Saved: models/hedge_ts_rf.pkl')

    with open('models/hedge_ts_lr.pkl', 'wb') as f:
        pickle.dump(lr, f)
    print('Saved: models/hedge_ts_lr.pkl')

    with open('models/hedge_ts_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print('Saved: models/hedge_ts_scaler.pkl')

    with open('models/hedge_ts_features.txt', 'w') as f:
        f.write('\n'.join(all_features))
    print(f'Saved: models/hedge_ts_features.txt ({len(all_features)} features)')

    # ============================================================================
    # HEDGING STRATEGY - FORWARD TEST
    # ============================================================================
    print('\n' + '='*80)
    print(f'HEDGING STRATEGY - FORWARD TEST (RF >= {RF_THRESHOLD})')
    print('='*80)
    print(f'Logic: Take BOTH LONG and SHORT when RF >= {RF_THRESHOLD}')
    print('       Cancel whichever side hits stop loss first')
    print('       Keep the surviving trade until target/stop')

    # Prepare test set with datetime and session info
    test_hedge = test.reset_index()
    test_hedge['hour'] = test_hedge['Datetime'].dt.hour
    
    def get_session(hour):
        if 0 <= hour < 8:
            return 'ASIAN'
        elif 8 <= hour < 13:
            return 'LONDON'
        elif 13 <= hour < 17:
            return 'NY_OVERLAP'
        elif 17 <= hour < 22:
            return 'NY'
        else:
            return 'LATE'

    test_hedge['session'] = test_hedge['hour'].apply(get_session)

    # Filter for high probability setups (RF >= RF_THRESHOLD)
    hedge_setups = test_hedge[test_hedge['rf_prob'] >= RF_THRESHOLD].copy()

    print(f'\nTotal test bars: {len(test_hedge)}')
    print(f'High probability setups (RF >= {RF_THRESHOLD}): {len(hedge_setups)} ({len(hedge_setups)/len(test_hedge)*100:.2f}%)')

    # Bar-by-bar OHLC hedge simulation
    # Phase 1: Both sides open — walk bars until one side's SL is hit
    # Phase 2: Surviving side continues — walk remaining bars until TP/SL/timeout
    hedge_results = []
    highs = test_hedge['High'].values
    lows  = test_hedge['Low'].values
    closes = test_hedge['Close'].values
    n_bars = len(test_hedge)

    for idx in hedge_setups.index:
        entry = test_hedge.at[idx, 'Close']
        long_tp  = entry + TARGET
        long_sl  = entry - STOP
        short_tp = entry - TARGET
        short_sl = entry + STOP

        surviving_side = None
        result = 'TIMEOUT'
        pnl = 0.0
        cancel_bar = None

        # --- PHASE 1: both sides open, find which SL is hit first ---
        end_phase1 = min(idx + HORIZON + 1, n_bars)
        for j in range(idx + 1, end_phase1):
            h, l = highs[j], lows[j]
            long_sl_hit  = (l <= long_sl)    # price dropped to long SL
            short_sl_hit = (h >= short_sl)   # price rose to short SL

            if long_sl_hit and short_sl_hit:
                # Both stops breached in same bar — conservatively both stopped
                surviving_side = 'BOTH_STOPPED'
                result = 'LOSS'
                pnl = -STOP - STOP
                break
            elif long_sl_hit:
                # Long stopped out, short survives
                surviving_side = 'SHORT'
                cancel_bar = j
                break
            elif short_sl_hit:
                # Short stopped out, long survives
                surviving_side = 'LONG'
                cancel_bar = j
                break

        # If neither SL was hit during Phase 1, close both at horizon (scratch)
        if surviving_side is None:
            surviving_side = 'NO_TRIGGER'
            result = 'SCRATCH'
            # Close both at last bar's close — net ~0 since hedge offsets
            last_close = closes[min(idx + HORIZON, n_bars - 1)]
            pnl = (last_close - entry) + (entry - last_close)  # = 0
            pnl = 0.0

        # --- PHASE 2: surviving side runs to TP / SL / timeout ---
        if cancel_bar is not None:
            cancelled_loss = -STOP  # cancelled side always loses STOP
            end_phase2 = min(idx + HORIZON + 1, n_bars)
            phase2_resolved = False

            for j in range(cancel_bar + 1, end_phase2):
                h, l = highs[j], lows[j]

                if surviving_side == 'LONG':
                    tp_hit = (h >= long_tp)
                    sl_hit = (l <= long_sl)
                    if tp_hit and sl_hit:
                        # Ambiguous bar — conservatively assume SL hit first
                        result = 'LOSS'
                        pnl = cancelled_loss + (-STOP)
                    elif tp_hit:
                        result = 'WIN'
                        pnl = cancelled_loss + TARGET
                    elif sl_hit:
                        result = 'LOSS'
                        pnl = cancelled_loss + (-STOP)
                    else:
                        continue

                elif surviving_side == 'SHORT':
                    tp_hit = (l <= short_tp)
                    sl_hit = (h >= short_sl)
                    if tp_hit and sl_hit:
                        result = 'LOSS'
                        pnl = cancelled_loss + (-STOP)
                    elif tp_hit:
                        result = 'WIN'
                        pnl = cancelled_loss + TARGET
                    elif sl_hit:
                        result = 'LOSS'
                        pnl = cancelled_loss + (-STOP)
                    else:
                        continue

                phase2_resolved = True
                break

            if not phase2_resolved:
                # Surviving side didn't hit TP or SL — close at market
                last_close = closes[min(idx + HORIZON, n_bars - 1)]
                if surviving_side == 'LONG':
                    surv_pnl = last_close - entry
                else:
                    surv_pnl = entry - last_close
                pnl = cancelled_loss + surv_pnl
                result = 'TIMEOUT'

        hedge_results.append({
            'Datetime': test_hedge.at[idx, 'Datetime'],
            'session': test_hedge.at[idx, 'session'],
            'rf_prob': test_hedge.at[idx, 'rf_prob'],
            'entry': entry,
            'surviving_side': surviving_side,
            'result': result,
            'pnl': round(pnl, 2),
        })

    # Convert to DataFrame
    hedge_df = pd.DataFrame(hedge_results)

    if len(hedge_df) > 0:
        print(f'\nTotal hedged setups: {len(hedge_df)}')
        
        # Performance metrics using actual simulated pnl
        wins = (hedge_df['result'] == 'WIN').sum()
        losses = (hedge_df['result'] == 'LOSS').sum()
        timeouts = (hedge_df['result'] == 'TIMEOUT').sum()
        scratches = (hedge_df['result'] == 'SCRATCH').sum()
        total_pnl = hedge_df['pnl'].sum()
        avg_pnl = total_pnl / len(hedge_df)
        gross_profit = hedge_df[hedge_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(hedge_df[hedge_df['pnl'] < 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f'\n--- OVERALL PERFORMANCE ---')
        print(f'Total setups: {len(hedge_df)}')
        print(f'Wins:     {wins}')
        print(f'Losses:   {losses}')
        print(f'Timeouts: {timeouts}')
        print(f'Scratches:{scratches}')
        print(f'Win Rate: {wins / len(hedge_df) * 100:.1f}%')
        print(f'\nTotal P&L:    ${total_pnl:,.2f}')
        print(f'Avg P&L:      ${avg_pnl:.4f} per trade')
        print(f'Profit Factor: {pf:.2f}')
        print(f'RR: ${TARGET} TP / ${STOP} SL = {TARGET/STOP:.1f}:1')
        
        # Breakdown by surviving side
        print(f'\n--- SURVIVING SIDE BREAKDOWN ---')
        for side in ['LONG', 'SHORT', 'BOTH_STOPPED', 'NO_TRIGGER']:
            side_trades = hedge_df[hedge_df['surviving_side'] == side]
            if len(side_trades) > 0:
                count = len(side_trades)
                pct = count / len(hedge_df) * 100
                side_pnl = side_trades['pnl'].sum()
                print(f'{side:<15} {count:>6} ({pct:>5.1f}%)  P&L: ${side_pnl:>10,.2f}')
        
        # Session breakdown
        print(f'\n--- SESSION BREAKDOWN ---')
        for sess in ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']:
            sess_data = hedge_df[hedge_df['session'] == sess]
            if len(sess_data) > 0:
                sess_wins = (sess_data['result'] == 'WIN').sum()
                sess_wr = sess_wins / len(sess_data) * 100
                sess_pnl = sess_data['pnl'].sum()
                print(f'{sess:<12} {len(sess_data):>5} setups  WR: {sess_wr:>5.1f}%  P&L: ${sess_pnl:>10,.2f}')
        
        # Save results
        hedge_df['Datetime'] = hedge_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
        hedge_df['rf_prob'] = hedge_df['rf_prob'].round(3)
        hedge_df['entry'] = hedge_df['entry'].round(2)
        hedge_df.to_csv('data/processed/HEDGE_strategy.csv', index=False)
        print(f'\nSaved: data/processed/HEDGE_strategy.csv')
        
        # Show sample trades
        print('\n--- SAMPLE HEDGE SETUPS (first 20) ---')
        print(hedge_df.head(20).to_string(index=False))
    else:
        print('\nNo hedge setups generated.')

    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)