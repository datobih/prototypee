import pandas as pd
import numpy as np

# ============================================================================
# MOVEMENT STRADDLE STRATEGY - Direction Agnostic
# ============================================================================
# Core Insight: We predict MOVEMENT, not direction
# Strategy: Place simultaneous LONG and SHORT bracket orders
# Win Condition: Either side hits TP before its SL
# Loss Condition: Both sides hit SL (whipsaw/chop)
# ============================================================================

def analyze_movement_straddle(data_file='data/raw/XAUUSD1.csv', 
                              results_file='data/processed/XAUUSD1_feature_analysis.csv',
                              rf_threshold=0.85,
                              stop_pct=0.0005,    # 0.05%
                              target_pct=0.001,   # 0.1%
                              horizon=20,
                              bar_minutes=1):
    
    # Load data
    df = pd.read_csv(data_file, sep='\t', 
                     names=['Datetime','Open','High','Low','Close','Volume'], 
                     parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    results = pd.read_csv(results_file, index_col=0, parse_dates=True)
    high_conf = results[results['rf_prob'] >= rf_threshold]
    
    print('='*100)
    print('MOVEMENT STRADDLE STRATEGY - Direction Agnostic')
    print('='*100)
    print()
    print('CORE LOGIC:')
    print('  - We predict MOVEMENT will happen, not which direction')
    print('  - Place BOTH long and short bracket orders simultaneously')
    print('  - WIN: Either direction hits TP first (movement captured)')
    print('  - LOSS: Both directions hit SL (choppy/ranging market)')
    print('  - TIMEOUT: Neither completes within horizon')
    print()
    print(f'Stop Loss: {stop_pct*100:.3f}% | Take Profit: {target_pct*100:.2f}% | R:R = {target_pct/stop_pct:.1f}:1')
    print(f'Horizon: {horizon} bars ({horizon*bar_minutes} minutes)')
    print(f'Entry Signal: RF probability >= {rf_threshold} (predicting movement)')
    print(f'Total signals found: {len(high_conf)}')
    print()
    
    trades = []
    
    for dt, row in high_conf.iterrows():
        if dt not in df.index:
            continue
        
        idx = df.index.get_loc(dt)
        if idx + horizon >= len(df):
            continue
        
        entry = df.loc[dt, 'Close']
        future = df.iloc[idx+1:idx+horizon+1]
        
        # Calculate bracket levels
        # LONG bracket: TP above, SL below
        long_tp = entry * (1 + target_pct)
        long_sl = entry * (1 - stop_pct)
        # SHORT bracket: TP below, SL above
        short_tp = entry * (1 - target_pct)
        short_sl = entry * (1 + stop_pct)
        
        long_result = None
        short_result = None
        long_bar = None
        short_bar = None
        first_hit = None  # Track which bracket completes first
        
        for i, (bar_dt, bar) in enumerate(future.iterrows(), 1):
            h = bar['High']
            l = bar['Low']
            o = bar['Open']
            
            # Check LONG bracket (if not already closed)
            if long_result is None:
                long_tp_hit = h >= long_tp
                long_sl_hit = l <= long_sl
                
                if long_tp_hit and long_sl_hit:
                    dist_to_tp = abs(o - long_tp)
                    dist_to_sl = abs(o - long_sl)
                    long_result = 'TP' if dist_to_tp < dist_to_sl else 'SL'
                elif long_tp_hit:
                    long_result = 'TP'
                elif long_sl_hit:
                    long_result = 'SL'
                if long_result:
                    long_bar = i
                    if first_hit is None:
                        first_hit = ('LONG', long_result, i)
            
            # Check SHORT bracket (if not already closed)
            if short_result is None:
                short_tp_hit = l <= short_tp
                short_sl_hit = h >= short_sl
                
                if short_tp_hit and short_sl_hit:
                    dist_to_tp = abs(o - short_tp)
                    dist_to_sl = abs(o - short_sl)
                    short_result = 'TP' if dist_to_tp < dist_to_sl else 'SL'
                elif short_tp_hit:
                    short_result = 'TP'
                elif short_sl_hit:
                    short_result = 'SL'
                if short_result:
                    short_bar = i
                    if first_hit is None:
                        first_hit = ('SHORT', short_result, i)
            
            # Early exit if we have a winner
            if long_result == 'TP' or short_result == 'TP':
                break
            
            # Both closed (both hit SL = loss)
            if long_result and short_result:
                break
        
        # Determine straddle outcome
        # WIN: At least one side hit TP
        # LOSS: Both sides hit SL (whipsaw)
        # TIMEOUT: Incomplete
        
        if long_result == 'TP' or short_result == 'TP':
            # Movement captured! One side won
            if long_result == 'TP' and short_result == 'SL':
                # Long won, short stopped out
                net_pct = (target_pct - stop_pct) * 100
            elif short_result == 'TP' and long_result == 'SL':
                # Short won, long stopped out
                net_pct = (target_pct - stop_pct) * 100
            elif long_result == 'TP' and short_result is None:
                # Long won, short still open (close at breakeven)
                net_pct = target_pct * 100
            elif short_result == 'TP' and long_result is None:
                # Short won, long still open (close at breakeven)
                net_pct = target_pct * 100
            elif long_result == 'TP' and short_result == 'TP':
                # Both hit TP (rare, big move then reversal)
                net_pct = 2 * target_pct * 100
            else:
                net_pct = (target_pct - stop_pct) * 100
            straddle_outcome = 'WIN'
            winner = 'LONG' if long_result == 'TP' else 'SHORT'
        elif long_result == 'SL' and short_result == 'SL':
            # Whipsaw - both stopped out
            net_pct = -2 * stop_pct * 100
            straddle_outcome = 'LOSS'
            winner = 'NONE'
        else:
            # Timeout - close remaining positions at market
            net_pct = 0
            straddle_outcome = 'TIMEOUT'
            winner = 'NONE'
        
        trades.append({
            'datetime': dt,
            'entry': entry,
            'rf_prob': row['rf_prob'],
            'sl_dist': round(entry * stop_pct, 2),
            'tp_dist': round(entry * target_pct, 2),
            'long_result': long_result or 'OPEN',
            'long_bar': long_bar,
            'short_result': short_result or 'OPEN',
            'short_bar': short_bar,
            'straddle_outcome': straddle_outcome,
            'winner': winner,
            'pnl_pct': net_pct,
            'pnl_dollar': entry * net_pct / 100
        })
    
    # Print trade log
    print('DETAILED TRADE LOG')
    print('-'*120)
    header = f"{'Date/Time':<20} {'Entry':>9} {'SL±':>6} {'TP±':>6} {'LONG':>10} {'SHORT':>10} {'Winner':>7} {'Result':>8} {'PnL%':>7}"
    print(header)
    print('-'*120)
    
    for t in trades:
        long_str = t['long_result'] + ('@' + str(t['long_bar']) if t['long_bar'] else '')
        short_str = t['short_result'] + ('@' + str(t['short_bar']) if t['short_bar'] else '')
        pnl_str = f"{t['pnl_pct']:+.3f}"
        
        print(f"{str(t['datetime'])[:16]:<20} {t['entry']:>9.2f} {t['sl_dist']:>6.2f} {t['tp_dist']:>6.2f} {long_str:>10} {short_str:>10} {t['winner']:>7} {t['straddle_outcome']:>8} {pnl_str:>7}")
    
    print('-'*120)
    
    # Summary
    wins = sum(1 for t in trades if t['straddle_outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['straddle_outcome'] == 'LOSS')
    timeouts = sum(1 for t in trades if t['straddle_outcome'] == 'TIMEOUT')
    total_pnl = sum(t['pnl_pct'] for t in trades)
    
    long_wins = sum(1 for t in trades if t['winner'] == 'LONG')
    short_wins = sum(1 for t in trades if t['winner'] == 'SHORT')
    
    print()
    print('='*60)
    print('STRADDLE PERFORMANCE SUMMARY')
    print('='*60)
    print(f"Total Trades:  {len(trades)}")
    print()
    print(f"WINS:          {wins} ({wins/len(trades)*100:.1f}%)")
    print(f"  - Long won:  {long_wins}")
    print(f"  - Short won: {short_wins}")
    print(f"LOSSES:        {losses} ({losses/len(trades)*100:.1f}%) [whipsaw/chop]")
    print(f"TIMEOUTS:      {timeouts} ({timeouts/len(trades)*100:.1f}%)")
    print()
    print(f"Total PnL:     {total_pnl:+.2f}%")
    print(f"Avg per Trade: {total_pnl/len(trades):+.4f}%")
    print()
    
    # Expected returns
    print('EXPECTED RETURNS (Movement Strategy)')
    print('-'*60)
    
    # Calculate data timespan
    first_trade = trades[0]['datetime']
    last_trade = trades[-1]['datetime']
    days = (last_trade - first_trade).days
    months = days / 30
    
    trades_per_month = len(trades) / months if months > 0 else len(trades)
    monthly_pnl = trades_per_month * (total_pnl / len(trades))
    yearly_pnl = 12 * monthly_pnl
    
    print(f"Data span:         {days} days ({months:.1f} months)")
    print(f"Trades per month:  ~{trades_per_month:.1f}")
    print(f"Monthly return:    {monthly_pnl:+.2f}%")
    print(f"Yearly return:     {yearly_pnl:+.2f}%")
    print()
    
    # Win rate needed for breakeven
    breakeven_wr = stop_pct / (target_pct + stop_pct)
    print('BREAKEVEN ANALYSIS')
    print('-'*60)
    print(f"With R:R of {target_pct/stop_pct:.1f}:1, breakeven win rate = {breakeven_wr*100:.1f}%")
    print(f"Your actual win rate: {wins/(wins+losses)*100:.1f}% (excluding timeouts)")
    edge = (wins/(wins+losses)) - breakeven_wr if (wins+losses) > 0 else 0
    print(f"Edge over breakeven: {edge*100:+.1f}%")
    print()
    
    return trades

if __name__ == '__main__':
    trades = analyze_movement_straddle()
