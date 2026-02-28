import pandas as pd
import mplfinance as mpf
import os
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

# Extract DATA_PATH dynamically from ob_filtering.py to avoid running the whole script
RAW_DATA_PATH = None
with open(os.path.join(SCRIPT_DIR, 'ob_filtering.py'), 'r') as f:
    for line in f:
        if line.startswith('DATA_PATH ='):
            exec(line)
            RAW_DATA_PATH = DATA_PATH
            break

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_mitigation_study.csv')

# Configure visualization window
PRE_BARS = 30
POST_BARS = 100

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data():
    print("Loading raw price data...")
    df = pd.read_csv(RAW_DATA_PATH, sep='\t', header=0,
                     names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    
    print("Loading processed OB data...")
    ob_df = pd.read_csv(PROCESSED_DATA_PATH)
    ob_df['ob_dt'] = pd.to_datetime(ob_df['ob_dt'])
    
    return df, ob_df

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_ob(df, ob_row, idx):
    ob_dt = ob_row['ob_dt']
    ob_type = ob_row['type']
    ob_high = ob_row['ob_high']
    ob_low = ob_row['ob_low']
    reaction = ob_row['reaction']
    
    # Error handling if ob_dt not in index exactly
    if ob_dt not in df.index:
        # Find closest
        closest_idx = df.index.get_indexer([ob_dt], method='nearest')[0]
        ob_dt = df.index[closest_idx]
        
    pos = df.index.get_loc(ob_dt)
    
    bars_to_mit = ob_row.get('bars_to_mitigate', 0)
    if pd.isna(bars_to_mit):
        bars_to_mit = 0
    touch_pos = pos + int(bars_to_mit)
    
    start_pos = max(0, touch_pos - PRE_BARS)
    end_pos = min(len(df), touch_pos + POST_BARS)
    
    plot_df = df.iloc[start_pos:end_pos].copy()
    
    # Define rectangle coordinates for the OB zone
    # It covers the whole length of the plotted period from the OB out
    ob_color = 'rgba(0, 255, 0, 0.2)' if ob_type == 'bull' else 'rgba(255, 0, 0, 0.2)'
    
    # We use mplfinance to plot
    # Create horizontal lines for the zone
    hlines = dict(hlines=[ob_high, ob_low], 
                  colors=['g' if ob_type == 'bull' else 'r'] * 2, 
                  linestyle='--', linewidths=1)
                  
    # Fill between lines (the OB zone)
    # We want to fill from the OB candle onwards
    fill_where = plot_df.index >= ob_dt
    
    title = f"OB #{idx} | {ob_dt.strftime('%Y-%m-%d %H:%M')} | Type: {ob_type.upper()} | Reaction: {reaction}"
    
    print(f"Plotting: {title}")
    
    fig, axlist = mpf.plot(plot_df, type='candle', style='yahoo',
             title=title,
             hlines=hlines,
             fill_between=dict(y1=ob_low, y2=ob_high, where=fill_where, alpha=0.2, color='g' if ob_type == 'bull' else 'r'),
             returnfig=True,
             figsize=(12, 8))
             
    # Show the plot and wait for user to close it before continuing
    import matplotlib.pyplot as plt
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize Order Blocks')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of OBs to visualize')
    args = parser.parse_args()

    df, ob_df = load_data()
    
    # Let's filter for the newly discovered high probability pattern!
    # FVG + High OB Vol (>1.2)
    mask = ob_df['has_fvg'] & (ob_df['ob_vol_rel'] > 1.2)
    filtered_obs = ob_df[mask].copy()
    
    print(f"\nFound {len(filtered_obs)} OBs matching 'FVG + High OB Vol (>1.2)'")
    print(f"Bounces: {(filtered_obs['reaction'] == 'BOUNCE').sum()} | Breaks: {(filtered_obs['reaction'] == 'BREAK').sum()} | Weak: {(filtered_obs['reaction'] == 'WEAK').sum()}")
    print("-" * 50)
    
    # Sort by date (descending to see recent ones first)
    filtered_obs = filtered_obs.sort_values('ob_dt', ascending=False)
    
    count = 0
    for idx, row in filtered_obs.iterrows():
        if count >= args.limit:
            break
        plot_ob(df, row, count+1)
        count += 1
        
    print("\nDone visualizing.")

if __name__ == "__main__":
    main()
