import pandas as pd
import os
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_mitigation_study.csv')

# Fixed Risk/Reward targets to test
TARGET_RRS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# At what RR threshold do we move Stop Loss to Break Even?
BE_TRIGGER_RR = 1.0

def simulate_fixed_tp(df, combo_name, mask):
    combo_df = df[mask].copy()
    if len(combo_df) == 0:
        return

    print(f"\n{'='*85}")
    print(f"STRATEGY EXPECTANCY: {combo_name} (Sample: {len(combo_df)} trades)")
    print(f"Active Rule: Move SL to Break Even after {BE_TRIGGER_RR}R profit")
    print(f"{'='*85}")
    print(f"{'TP Target':<12} | {'Win Rate':<10} | {'Wins':<8} | {'Break-Evens':<12} | {'Losses':<8} | {'Expectancy (R)':<15}")
    print("-" * 85)
    
    # We will assume a break immediately is a full 1R loss.
    for target in TARGET_RRS:
        # A trade is a win if its max favorable excursion >= the target
        wins = len(combo_df[combo_df['trade_rr'] >= target])
        
        # A trade is a break-even if it hits the BE trigger, but fails to reach the final TARGET
        if target > BE_TRIGGER_RR:
            break_evens = len(combo_df[(combo_df['trade_rr'] >= BE_TRIGGER_RR) & (combo_df['trade_rr'] < target)])
        else:
            break_evens = 0 # If TP is less than or equals BE trigger, we just win.
            
        losses = len(combo_df) - wins - break_evens
        
        win_rate = wins / len(combo_df) if len(combo_df) > 0 else 0
        loss_rate = losses / len(combo_df) if len(combo_df) > 0 else 0
        
        # Expectancy = (Win Rate * Reward) - (Loss Rate * 1 Risk) + (Break Even * 0 Risk)
        expectancy = (win_rate * target) - (loss_rate * 1.0)
        
        print(f"{target:>4.1f} R       | {win_rate*100:>7.1f}%   | {wins:<8} | {break_evens:<12} | {losses:<8} | {expectancy:>+6.3f} R")

def main():
    print("Loading Order Block excursion data...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Could not find data at {PROCESSED_DATA_PATH}")
        print("Please run ob_filtering.py first to generate the latest study data.")
        return
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Ensure trade_rr is available, we use that as the max excursion for our TP simulation
    if 'trade_rr' not in df.columns:
        print("Error: 'trade_rr' column not found in data. Has ob_filtering.py logic changed?")
        return

    # 1. Baseline - All Touched OBs
    simulate_fixed_tp(df, "BASELINE - All Touched OBs", df['touched'] == True)
    
    # 2. Your Highest Volume Pattern right now
    mask_high_vol = df['has_fvg'] & (df['ob_vol_rel'] > 1.2)
    simulate_fixed_tp(df, "FVG + High OB Vol (>1.2)", mask_high_vol)
    
    # 3. Your other strong combo
    mask_fvg_disp4 = df['has_fvg'] & (df['disp_atr'] >= 4.0)
    simulate_fixed_tp(df, "FVG + Disp >= 4x", mask_fvg_disp4)
    
    # 4. ELITE Inducement Setup
    mask_elite = df['has_fvg'] & df['has_inducement'] & (df['ob_vol_rel'] > 1.2)
    simulate_fixed_tp(df, "FVG + Inducement + High OB Vol", mask_elite)
    
    print("\n\nNote: This simulator assumes any trade that touches the zone but fails to hit ")
    print("the fixed Target RR will eventually hit the 1.0R Stop Loss.")

if __name__ == "__main__":
    main()
