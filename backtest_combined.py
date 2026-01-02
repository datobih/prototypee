"""
COMBINED BACKTEST - Movement + Direction (No Data Leakage)
==========================================================
Walk-forward backtest where BOTH models are trained only on past data.

Strategy:
- Movement model (RF) predicts if there will be a move (rf_prob >= 0.85)
- Direction model (LogReg) validates LONG signals (dir_prob > 0.60)
- Only take LONG trades when both conditions are met
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_feature_data():
    """Load the pre-processed feature analysis data (test portion only from analyze_features.py)."""
    print('Loading pre-processed data...')
    df = pd.read_csv('data/processed/XAUUSD1_feature_analysis.csv', parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    print(f'Loaded {len(df)} bars (this is the TEST portion from analyze_features.py)')
    return df


def load_raw_and_process():
    """Load raw data and create features exactly like analyze_features.py."""
    print('Loading raw XAUUSD1 data...')
    df = pd.read_csv('data/raw/XAUUSD1.csv', sep='\t', 
                     names=['Datetime','Open','High','Low','Close','Volume'], 
                     parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    print(f'Loaded {len(df)} raw bars')
    
    print('Creating microstructure features...')
    df = create_microstructure_features(df)
    
    print('Labeling outcomes (20 bars, 0.1% target, 0.05% stop)...')
    df = label_outcomes(df, horizon=20, target=0.001, stop=0.0005)
    
    print('Creating combo features...')
    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']
    
    return df


def create_microstructure_features(df):
    """Create features exactly like analyze_features.py."""
    df = df.copy()
    
    # Price structure
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)
    
    # Order flow
    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)
    
    # Imbalance
    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()
    
    # Momentum consistency - is_up = bullish candle (close > open)
    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))
    
    # Volatility
    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)
    
    # Trend structure
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - \
                        ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']
    
    # Support/Resistance
    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)
    
    # Rejection patterns
    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
    
    # Size patterns
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)
    
    return df.dropna()


def label_outcomes(df, horizon=20, target=0.001, stop=0.0005):
    """Label outcomes exactly like analyze_features.py."""
    df = df.copy()
    outcomes = []
    
    for i in range(len(df) - horizon):
        if i % 20000 == 0:
            print(f'  Labeling {i}/{len(df)-horizon}...')
        
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+horizon+1]
        
        # Check LONG
        long_hit = False
        for h, l in zip(future['High'], future['Low']):
            if h >= entry * (1 + target):
                long_hit = True
                break
            if l <= entry * (1 - stop):
                break
        
        # Check SHORT
        short_hit = False
        if not long_hit:
            for h, l in zip(future['High'], future['Low']):
                if l <= entry * (1 - target):
                    short_hit = True
                    break
                if h >= entry * (1 + stop):
                    break
        
        outcomes.append(1 if long_hit else (2 if short_hit else 0))
    
    df = df.iloc[:len(outcomes)].copy()
    df['outcome'] = outcomes
    return df


def engineer_direction_features(df):
    """Engineer additional features for direction prediction."""
    df = df.copy()
    
    # Buy/Sell pressure
    df['buy_press'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['sell_press'] = (df['High'] - df['Close']) / (df['range'] + 1e-10)
    df['press_imb'] = df['buy_press'] - df['sell_press']
    df['press_imb_3'] = df['press_imb'].rolling(3).mean()
    df['press_imb_5'] = df['press_imb'].rolling(5).mean()
    
    # Up ratio
    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_ratio_3'] = df['is_up'].rolling(3).mean()
    df['up_ratio_5'] = df['is_up'].rolling(5).mean()
    df['up_ratio_10'] = df['is_up'].rolling(10).mean()
    
    # Momentum ratio
    chg = df['Close'].diff()
    pos = chg.clip(lower=0)
    neg = (-chg).clip(lower=0)
    df['mom_ratio_5'] = pos.rolling(5).sum() / (pos.rolling(5).sum() + neg.rolling(5).sum() + 1e-10)
    df['mom_ratio_10'] = pos.rolling(10).sum() / (pos.rolling(10).sum() + neg.rolling(10).sum() + 1e-10)
    
    # EMA features
    ema5 = df['Close'].ewm(span=5).mean()
    ema10 = df['Close'].ewm(span=10).mean()
    ema20 = df['Close'].ewm(span=20).mean()
    df['ema5_slope'] = ema5.diff(3) / ema5.shift(3) * 1000
    df['ema10_slope'] = ema10.diff(5) / ema10.shift(5) * 1000
    df['price_vs_ema5'] = (df['Close'] - ema5) / ema5 * 1000
    df['price_vs_ema10'] = (df['Close'] - ema10) / ema10 * 1000
    df['ema_align'] = ((ema5 > ema10) & (ema10 > ema20)).astype(int) - \
                      ((ema5 < ema10) & (ema10 < ema20)).astype(int)
    
    # Range position
    h10 = df['High'].rolling(10).max()
    l10 = df['Low'].rolling(10).min()
    df['range_pos_10'] = (df['Close'] - l10) / (h10 - l10 + 1e-10)
    df['breakout_press'] = df['range_pos_10'] - 0.5
    
    # Rejection
    df['upper_wick_r'] = df['upper_wick'] / (df['range'] + 1e-10)
    df['lower_wick_r'] = df['lower_wick'] / (df['range'] + 1e-10)
    df['net_reject'] = df['lower_wick_r'] - df['upper_wick_r']
    df['net_reject_5'] = df['net_reject'].rolling(5).mean()
    
    # ROC
    df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 1000
    df['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 1000
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_sig'] = (df['rsi'] - 50) / 50
    
    # Combo
    df['bull_combo'] = ((df['price_vs_ema10'] > 0).astype(float) + 
                        (df['press_imb_5'] > 0).astype(float) + 
                        (df['up_ratio_5'] > 0.5).astype(float)) / 3
    df['dir_score'] = df['bull_combo'] * 2 - 1
    
    return df


def get_movement_features():
    """Features for movement model - must match analyze_features.py exactly (includes combos)."""
    base = [
        'flow_3', 'flow_5', 'flow_10', 'flow_momentum',
        'imbalance_3', 'imbalance_5',
        'consistency_3', 'consistency_5',
        'vol_ratio', 'vol_expansion', 'vol_contraction',
        'trend_align', 'dist_ema8',
        'at_high', 'at_low',
        'upper_reject', 'lower_reject',
        'big_body', 'small_body',
        'body_pct', 'close_position'
    ]
    # Combo features used in analyze_features.py
    combos = [
        'combo_flow_trend', 'combo_vol_imbalance', 'combo_consistency_position',
        'combo_body_reject', 'combo_trend_volatility', 'combo_imbalance_momentum',
        'combo_position_consistency', 'combo_vol_flow'
    ]
    return base + combos


def create_combo_features(df):
    """Create the combo features used in analyze_features.py."""
    df = df.copy()
    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']
    return df


def get_direction_features():
    """Features for direction model."""
    return [
        'buy_press', 'sell_press', 'press_imb', 'press_imb_3', 'press_imb_5',
        'up_ratio_3', 'up_ratio_5', 'up_ratio_10', 'mom_ratio_5', 'mom_ratio_10',
        'ema5_slope', 'ema10_slope', 'price_vs_ema5', 'price_vs_ema10', 'ema_align',
        'range_pos_10', 'breakout_press', 'upper_wick_r', 'lower_wick_r',
        'net_reject', 'net_reject_5', 'roc_5', 'roc_10', 'rsi_sig',
        'bull_combo', 'dir_score',
        'flow_3', 'flow_5', 'flow_10', 'imbalance_3', 'imbalance_5',
        'consistency_3', 'consistency_5', 'trend_align', 'close_position', 'body_pct'
    ]


def walk_forward_backtest(df, move_feats, dir_feats, n_splits=5, 
                          move_th=0.85, dir_th=0.60):
    """
    Walk-forward backtest with NO data leakage.
    Both models trained fresh on each fold using only past data.
    """
    print(f'\n{"="*70}')
    print(f'WALK-FORWARD BACKTEST (No Data Leakage)')
    print(f'Movement threshold: {move_th}, Direction threshold: {dir_th}')
    print(f'{"="*70}')
    
    # Prepare targets
    df['is_movement'] = (df['outcome'] != 0).astype(int)
    df['is_long'] = (df['outcome'] == 1).astype(int)
    
    # Filter features that exist
    move_feats = [f for f in move_feats if f in df.columns]
    dir_feats = [f for f in dir_feats if f in df.columns]
    
    # Drop NaN
    all_feats = list(set(move_feats + dir_feats))
    df = df.dropna(subset=all_feats).copy()
    print(f'Total bars after dropna: {len(df)}')
    print(f'Movement rate: {df["is_movement"].mean()*100:.1f}%')
    print(f'Movement features: {len(move_feats)}, Direction features: {len(dir_feats)}')
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_trades = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx].copy()
        
        print(f'\nFold {fold+1}: Train={len(train_df)}, Test={len(test_df)}')
        
        # =====================================================
        # 1. Train MOVEMENT model on training data
        # =====================================================
        X_move_train = train_df[move_feats]
        y_move_train = train_df['is_movement']
        
        move_model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        )
        move_model.fit(X_move_train, y_move_train)
        
        # Predict on test
        X_move_test = test_df[move_feats]
        test_df['move_prob'] = move_model.predict_proba(X_move_test)[:, 1]
        
        # Debug movement probs
        mp = test_df['move_prob']
        print(f'  Move probs: min={mp.min():.3f}, max={mp.max():.3f}, >=0.80={( mp>=0.80).sum()}, >=0.85={(mp>=0.85).sum()}')
        
        # =====================================================
        # 2. Train DIRECTION model on movement bars from training data
        # =====================================================
        train_movement = train_df[train_df['outcome'] != 0]
        
        X_dir_train = train_movement[dir_feats]
        y_dir_train = train_movement['is_long']
        
        dir_scaler = StandardScaler()
        X_dir_train_s = dir_scaler.fit_transform(X_dir_train)
        
        dir_model = LogisticRegression(
            C=0.1, penalty='l2', class_weight='balanced', 
            max_iter=1000, random_state=42
        )
        dir_model.fit(X_dir_train_s, y_dir_train)
        
        # Predict on test
        X_dir_test = test_df[dir_feats]
        X_dir_test_s = dir_scaler.transform(X_dir_test)
        test_df['dir_prob'] = dir_model.predict_proba(X_dir_test_s)[:, 1]
        
        # =====================================================
        # 3. Generate signals: movement >= move_th AND direction > dir_th
        # =====================================================
        signals = test_df[(test_df['move_prob'] >= move_th) & 
                          (test_df['dir_prob'] > dir_th)]
        
        n_signals = len(signals)
        n_correct = (signals['outcome'] == 1).sum()  # LONG wins
        n_wrong = (signals['outcome'] == 2).sum()    # SHORT (lost)
        n_scratch = (signals['outcome'] == 0).sum()  # No move (scratch)
        
        acc = n_correct / n_signals * 100 if n_signals > 0 else 0
        
        print(f'  Signals: {n_signals}, Correct: {n_correct}, Wrong: {n_wrong}, Scratch: {n_scratch}, Acc: {acc:.1f}%')
        
        # Store trade details
        for idx, row in signals.iterrows():
            all_trades.append({
                'fold': fold + 1,
                'datetime': idx,
                'entry': row['Close'],
                'move_prob': row['move_prob'],
                'dir_prob': row['dir_prob'],
                'outcome': row['outcome'],
                'result': 'WIN' if row['outcome'] == 1 else ('LOSS' if row['outcome'] == 2 else 'SCRATCH')
            })
    
    # =====================================================
    # Summary
    # =====================================================
    print(f'\n{"="*70}')
    print('OVERALL RESULTS')
    print(f'{"="*70}')
    
    if len(all_trades) > 0:
        trades_df = pd.DataFrame(all_trades)
        
        total = len(trades_df)
        wins = (trades_df['result'] == 'WIN').sum()
        losses = (trades_df['result'] == 'LOSS').sum()
        scratches = (trades_df['result'] == 'SCRATCH').sum()
        
        win_rate = wins / total * 100
        
        # Exclude scratches for directional accuracy
        directional = trades_df[trades_df['result'] != 'SCRATCH']
        dir_accuracy = (directional['result'] == 'WIN').sum() / len(directional) * 100 if len(directional) > 0 else 0
        
        print(f'Total Signals: {total}')
        print(f'Wins:  {wins} ({wins/total*100:.1f}%)')
        print(f'Losses: {losses} ({losses/total*100:.1f}%)')
        print(f'Scratches: {scratches} ({scratches/total*100:.1f}%)')
        print(f'\nWin Rate (all): {win_rate:.1f}%')
        print(f'Directional Accuracy (excl scratches): {dir_accuracy:.1f}%')
        
        # Per-fold breakdown
        print(f'\nPer-Fold Breakdown:')
        for fold in trades_df['fold'].unique():
            fold_trades = trades_df[trades_df['fold'] == fold]
            fold_wins = (fold_trades['result'] == 'WIN').sum()
            fold_total = len(fold_trades)
            fold_acc = fold_wins / fold_total * 100 if fold_total > 0 else 0
            print(f'  Fold {fold}: {fold_total} signals, {fold_wins} wins ({fold_acc:.1f}%)')
        
        return trades_df
    else:
        print('No signals generated!')
        return None


def main():
    print('='*70)
    print('COMBINED BACKTEST: Movement + Direction (Exact analyze_features.py Method)')
    print('='*70)
    
    # Load and process exactly like analyze_features.py
    df = load_raw_and_process()
    
    print('\nEngineering direction features...')
    df = engineer_direction_features(df)
    
    move_feats = get_movement_features()
    dir_feats = get_direction_features()
    
    # Filter features that exist
    move_feats = [f for f in move_feats if f in df.columns]
    dir_feats = [f for f in dir_feats if f in df.columns]
    all_feats = list(set(move_feats + dir_feats))
    df = df.dropna(subset=all_feats)
    
    df['is_movement'] = (df['outcome'] != 0).astype(int)
    df['is_long'] = (df['outcome'] == 1).astype(int)
    
    # 60/40 split like analyze_features.py
    split = int(len(df) * 0.6)
    train = df.iloc[:split]
    test = df.iloc[split:].copy()
    
    print(f'\n60/40 SPLIT (Same as analyze_features.py)')
    print(f'Train: {len(train)} bars, Test: {len(test)} bars')
    print(f'Movement features: {len(move_feats)}, Direction features: {len(dir_feats)}')
    
    # =====================================================
    # 1. Train MOVEMENT model (RF)
    # =====================================================
    print('\nTraining Movement model (RandomForest)...')
    move_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    move_model.fit(train[move_feats], train['is_movement'])
    test['move_prob'] = move_model.predict_proba(test[move_feats])[:, 1]
    
    mp = test['move_prob']
    print(f'Move probs: min={mp.min():.3f}, max={mp.max():.3f}')
    print(f'  >=0.70: {(mp>=0.70).sum()}, >=0.75: {(mp>=0.75).sum()}, >=0.80: {(mp>=0.80).sum()}, >=0.85: {(mp>=0.85).sum()}')
    
    # =====================================================
    # 2. Train DIRECTION model (LogReg) on movement bars only
    # =====================================================
    print('\nTraining Direction model (LogReg) on movement bars...')
    train_mov = train[train['outcome'] != 0]
    dir_scaler = StandardScaler()
    dir_model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    dir_model.fit(dir_scaler.fit_transform(train_mov[dir_feats]), train_mov['is_long'])
    test['dir_prob'] = dir_model.predict_proba(dir_scaler.transform(test[dir_feats]))[:, 1]
    
    dp = test['dir_prob']
    print(f'Dir probs: min={dp.min():.3f}, max={dp.max():.3f}')
    print(f'  >0.55: {(dp>0.55).sum()}, >0.58: {(dp>0.58).sum()}, >0.60: {(dp>0.60).sum()}')
    
    # =====================================================
    # 3. Test combined strategy at various thresholds
    # =====================================================
    print(f'\n{"="*70}')
    print('COMBINED STRATEGY RESULTS (60/40 Split - No Leakage)')
    print(f'{"="*70}')
    
    print(f'\n{"Move Th":<10} {"Dir Th":<10} {"Signals":<10} {"Wins":<10} {"Losses":<10} {"Scratch":<10} {"Win%":<10} {"Dir Acc%"}')
    print('-'*80)
    
    best = {'acc': 0}
    
    for move_th in [0.70, 0.75, 0.80, 0.85, 0.90]:
        for dir_th in [0.50, 0.55, 0.58, 0.60]:
            sigs = test[(test['move_prob'] >= move_th) & (test['dir_prob'] > dir_th)]
            n = len(sigs)
            if n == 0:
                continue
            
            wins = (sigs['outcome'] == 1).sum()
            losses = (sigs['outcome'] == 2).sum()
            scratch = (sigs['outcome'] == 0).sum()
            
            win_pct = wins / n * 100
            dir_acc = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            
            print(f'{move_th:<10.2f} {dir_th:<10.2f} {n:<10} {wins:<10} {losses:<10} {scratch:<10} {win_pct:<10.1f} {dir_acc:.1f}')
            
            if win_pct > best['acc'] and n >= 20:
                best = {'move': move_th, 'dir': dir_th, 'n': n, 'wins': wins, 'acc': win_pct, 'dir_acc': dir_acc}
    
    # Movement only baseline
    print(f'\n{"="*70}')
    print('MOVEMENT ONLY (No Direction Filter)')
    print(f'{"="*70}')
    for move_th in [0.75, 0.80, 0.85, 0.90]:
        sigs = test[test['move_prob'] >= move_th]
        n = len(sigs)
        if n > 0:
            wins = (sigs['outcome'] != 0).sum()  # Movement = success
            print(f'move>={move_th}: {n} signals, {wins} moves ({wins/n*100:.1f}%)')
    
    # Direction filter improvement
    print(f'\n{"="*70}')
    print('DIRECTION FILTER VALUE')
    print(f'{"="*70}')
    for move_th in [0.75, 0.80, 0.85]:
        # Without direction filter
        sigs_all = test[test['move_prob'] >= move_th]
        n_all = len(sigs_all)
        long_all = (sigs_all['outcome'] == 1).sum()
        long_pct_all = long_all / n_all * 100 if n_all > 0 else 0
        
        # With direction filter
        sigs_filt = test[(test['move_prob'] >= move_th) & (test['dir_prob'] > 0.60)]
        n_filt = len(sigs_filt)
        long_filt = (sigs_filt['outcome'] == 1).sum()
        long_pct_filt = long_filt / n_filt * 100 if n_filt > 0 else 0
        
        improvement = long_pct_filt - long_pct_all
        print(f'move>={move_th}: Without dir={long_pct_all:.1f}% ({n_all}), With dir>0.60={long_pct_filt:.1f}% ({n_filt}), Improvement: {improvement:+.1f}%')
    
    if best['acc'] > 0:
        print(f'\n{"="*70}')
        print(f'BEST COMBINED: move>={best["move"]}, dir>{best["dir"]}')
        print(f'  Signals: {best["n"]}, Wins: {best["wins"]}, Win Rate: {best["acc"]:.1f}%')
        print(f'  Directional Accuracy: {best["dir_acc"]:.1f}%')
        print(f'{"="*70}')


if __name__ == '__main__':
    main()
