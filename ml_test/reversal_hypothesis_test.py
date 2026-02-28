"""
Reversal Hypothesis Test
========================
Hypothesis: When RF probability >= 0.70, the NEXT candle moves in the
            OPPOSITE direction of the signal candle.

Rules:
  - Signal fires when rf_prob >= threshold
  - After a signal fires, ignore the next 5 bars for new signals (cooldown)
  - Only measure direction (Close > Open = bullish, Close < Open = bearish)
  - No TP/SL — purely directional accuracy
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
RF_THRESHOLD = 0.70
COOLDOWN = 5  # bars to skip after a signal

# ============================================================================
# LOAD DATA
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'processed', 'XAUUSD1_feature_analysis.csv')

print('='*80)
print('REVERSAL HYPOTHESIS TEST')
print('='*80)
print(f'Threshold: RF >= {RF_THRESHOLD}')
print(f'Cooldown:  {COOLDOWN} bars after each signal')
print(f'Metric:    Pure direction (no TP/SL)')
print()

df = pd.read_csv(data_path)
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f'Loaded {len(df)} bars')

# Direction of each candle
df['body'] = df['Close'] - df['Open']
df['direction'] = np.where(df['body'] > 0, 'BULL', np.where(df['body'] < 0, 'BEAR', 'DOJI'))

# Next candle direction
df['next_direction'] = df['direction'].shift(-1)
df['next_body'] = df['body'].shift(-1)

# ============================================================================
# FIND SIGNALS WITH COOLDOWN
# ============================================================================
signals = []
cooldown_until = -1

for i in range(len(df) - 1):  # -1 because we need next candle
    if i <= cooldown_until:
        continue
    
    if df.iloc[i]['rf_prob'] >= RF_THRESHOLD:
        row = df.iloc[i]
        signal_dir = row['direction']
        next_dir = row['next_direction']
        
        if signal_dir == 'DOJI':
            continue  # skip dojis
        
        # Hypothesis: next candle should be OPPOSITE
        expected_dir = 'BEAR' if signal_dir == 'BULL' else 'BULL'
        
        if next_dir == 'DOJI':
            hypothesis_correct = None  # inconclusive
        else:
            hypothesis_correct = (next_dir == expected_dir)
        
        signals.append({
            'Datetime': row['Datetime'],
            'rf_prob': round(row['rf_prob'], 3),
            'signal_dir': signal_dir,
            'signal_body': round(row['body'], 2),
            'next_dir': next_dir,
            'next_body': round(row['next_body'], 2),
            'expected_dir': expected_dir,
            'hypothesis_correct': hypothesis_correct
        })
        
        # Apply cooldown
        cooldown_until = i + COOLDOWN

signals_df = pd.DataFrame(signals)

# ============================================================================
# OVERALL RESULTS
# ============================================================================
print(f'\n{"="*80}')
print('RESULTS')
print('='*80)

# Remove inconclusive (doji next candles)
conclusive = signals_df[signals_df['hypothesis_correct'].notna()].copy()
inconclusive = signals_df[signals_df['hypothesis_correct'].isna()]

total = len(signals_df)
total_conclusive = len(conclusive)
correct = conclusive['hypothesis_correct'].sum()
incorrect = total_conclusive - correct

print(f'\nTotal signals (RF >= {RF_THRESHOLD}): {total}')
print(f'  Conclusive:    {total_conclusive}')
print(f'  Inconclusive:  {len(inconclusive)} (next candle was doji)')
print()
print(f'--- HYPOTHESIS: Next candle reverses ---')
print(f'Correct:   {correct} ({correct/total_conclusive*100:.1f}%)')
print(f'Incorrect: {incorrect} ({incorrect/total_conclusive*100:.1f}%)')

# ============================================================================
# BREAKDOWN BY SIGNAL DIRECTION
# ============================================================================
print(f'\n--- BY SIGNAL DIRECTION ---')
for sig_dir in ['BULL', 'BEAR']:
    sub = conclusive[conclusive['signal_dir'] == sig_dir]
    if len(sub) > 0:
        sub_correct = sub['hypothesis_correct'].sum()
        sub_pct = sub_correct / len(sub) * 100
        expected = 'BEAR' if sig_dir == 'BULL' else 'LONG'
        print(f'{sig_dir} signal -> expect reversal: {len(sub)} trades  '
              f'Correct: {sub_correct} ({sub_pct:.1f}%)')

# ============================================================================
# BREAKDOWN BY RF PROBABILITY BUCKET
# ============================================================================
print(f'\n--- BY RF PROBABILITY ---')
buckets = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]
for lo, hi in buckets:
    sub = conclusive[(conclusive['rf_prob'] >= lo) & (conclusive['rf_prob'] < hi)]
    if len(sub) > 0:
        sub_correct = sub['hypothesis_correct'].sum()
        sub_pct = sub_correct / len(sub) * 100
        print(f'RF [{lo:.2f}, {hi:.2f}):  {len(sub):>4} signals  '
              f'Reversal: {sub_correct:>4} ({sub_pct:>5.1f}%)  '
              f'Same dir: {len(sub)-sub_correct:>4} ({100-sub_pct:>5.1f}%)')

# ============================================================================
# SESSION BREAKDOWN
# ============================================================================
print(f'\n--- BY SESSION ---')
signals_df['hour'] = signals_df['Datetime'].dt.hour

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

signals_df['session'] = signals_df['hour'].apply(get_session)
conclusive_with_session = conclusive.copy()
conclusive_with_session['hour'] = conclusive_with_session['Datetime'].dt.hour
conclusive_with_session['session'] = conclusive_with_session['hour'].apply(get_session)

for sess in ['NY_OVERLAP', 'NY', 'LONDON', 'ASIAN', 'LATE']:
    sub = conclusive_with_session[conclusive_with_session['session'] == sess]
    if len(sub) > 0:
        sub_correct = sub['hypothesis_correct'].sum()
        sub_pct = sub_correct / len(sub) * 100
        print(f'{sess:<12} {len(sub):>4} signals  '
              f'Reversal: {sub_correct:>4} ({sub_pct:>5.1f}%)')

# ============================================================================
# COMPARE: What if we trade WITH the signal direction instead?
# ============================================================================
print(f'\n--- ALTERNATIVE: Trade WITH signal direction ---')
continuation = conclusive.copy()
continuation['continuation_correct'] = ~continuation['hypothesis_correct']
cont_correct = continuation['continuation_correct'].sum()
cont_pct = cont_correct / len(continuation) * 100
print(f'Continuation correct: {cont_correct} ({cont_pct:.1f}%)')
print(f'(This is just the inverse of reversal)')

# ============================================================================
# MAGNITUDE ANALYSIS
# ============================================================================
print(f'\n--- MAGNITUDE OF NEXT CANDLE ---')
conclusive_abs = conclusive.copy()
conclusive_abs['abs_next_body'] = conclusive_abs['next_body'].abs()

reversal_moves = conclusive_abs[conclusive_abs['hypothesis_correct'] == True]['abs_next_body']
continuation_moves = conclusive_abs[conclusive_abs['hypothesis_correct'] == False]['abs_next_body']

print(f'When reversal occurs:')
print(f'  Avg move:    ${reversal_moves.mean():.2f}')
print(f'  Median move: ${reversal_moves.median():.2f}')
print(f'When continuation occurs:')
print(f'  Avg move:    ${continuation_moves.mean():.2f}')
print(f'  Median move: ${continuation_moves.median():.2f}')

# ============================================================================
# MULTI-CANDLE LOOKAHEAD (check 2nd, 3rd candle too)
# ============================================================================
print(f'\n--- MULTI-CANDLE LOOKAHEAD ---')
print('Checking if reversal happens within N candles after signal:')

for lookahead in [1, 2, 3, 5, 10]:
    reversal_found = 0
    total_checked = 0
    
    cooldown_until = -1
    for i in range(len(df) - lookahead):
        if i <= cooldown_until:
            continue
        if df.iloc[i]['rf_prob'] >= RF_THRESHOLD:
            sig_dir = df.iloc[i]['direction']
            if sig_dir == 'DOJI':
                continue
            
            expected = 'BEAR' if sig_dir == 'BULL' else 'BULL'
            
            # Check if ANY of the next N candles are in expected direction
            found = False
            for j in range(1, lookahead + 1):
                if df.iloc[i + j]['direction'] == expected:
                    found = True
                    break
            
            total_checked += 1
            if found:
                reversal_found += 1
            
            cooldown_until = i + COOLDOWN
    
    if total_checked > 0:
        pct = reversal_found / total_checked * 100
        print(f'  Within {lookahead:>2} candle(s): {reversal_found}/{total_checked} = {pct:.1f}%')

# ============================================================================
# NO COOLDOWN COMPARISON
# ============================================================================
print(f'\n--- WITHOUT COOLDOWN (every RF >= {RF_THRESHOLD}) ---')
all_signals = df[df['rf_prob'] >= RF_THRESHOLD].copy()
all_signals = all_signals[all_signals['direction'] != 'DOJI']
all_signals = all_signals[all_signals['next_direction'] != 'DOJI']
all_signals['expected'] = np.where(all_signals['direction'] == 'BULL', 'BEAR', 'BULL')
all_signals['correct'] = all_signals['next_direction'] == all_signals['expected']
no_cd_correct = all_signals['correct'].sum()
no_cd_total = len(all_signals)
print(f'Total: {no_cd_total}  Reversal: {no_cd_correct} ({no_cd_correct/no_cd_total*100:.1f}%)')

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================
out_path = os.path.join(script_dir, '..', 'data', 'processed', 'REVERSAL_hypothesis_results.csv')
signals_df.to_csv(out_path, index=False)
print(f'\nSaved detailed results: {out_path}')

# ============================================================================
# SAMPLE SIGNALS
# ============================================================================
print(f'\n--- SAMPLE SIGNALS (first 30) ---')
print(signals_df.head(30).to_string(index=False))

print(f'\n{"="*80}')
print('HYPOTHESIS TEST COMPLETE')
print('='*80)
