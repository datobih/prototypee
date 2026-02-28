"""
news_reaction_backtest.py

Backtest: How does XAUUSD react to major US economic data releases?

Uses FRED API to pull historical release dates + values for key indicators,
then measures the gold price reaction at various horizons (5m, 15m, 30m, 1h, 4h).

Computes:
  - Average absolute move per indicator
  - Directional bias (does gold rally/sell on beats vs misses?)
  - Surprise magnitude vs price reaction correlation
  - Best/worst indicators for gold trading
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
from fredapi import Fred

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
FRED_API_KEY = 'bd863d7a82e2d86e045a06e5b16c2725'

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'XAUUSD1.csv')

# Key economic indicators that move gold
# Format: (series_id, name, release_hour_ET, release_minute_ET, direction)
# direction: 'inverse' = strong data → gold down, 'direct' = strong data → gold up
INDICATORS = [
    ('CPIAUCSL',   'CPI (All Urban)',           8, 30, 'inverse'),
    ('CPILFESL',   'Core CPI',                  8, 30, 'inverse'),
    ('PAYEMS',     'Non-Farm Payrolls',          8, 30, 'inverse'),
    ('UNRATE',     'Unemployment Rate',          8, 30, 'direct'),
    ('PCE',        'Personal Consumption',       8, 30, 'inverse'),
    ('PCEPILFE',   'Core PCE Price Index',       8, 30, 'inverse'),
    ('RSAFS',      'Retail Sales',               8, 30, 'inverse'),
    ('DGORDER',    'Durable Goods Orders',       8, 30, 'inverse'),
    ('UMCSENT',    'Consumer Sentiment (UMich)', 10,  0, 'inverse'),
    ('GDP',        'GDP',                        8, 30, 'inverse'),
    ('FEDFUNDS',   'Fed Funds Rate',            14,  0, 'direct'),
    ('JTSJOL',     'JOLTS Job Openings',        10,  0, 'inverse'),
    ('HOUST',      'Housing Starts',             8, 30, 'inverse'),
    ('ISMNMI',     'ISM Non-Manufacturing',     10,  0, 'inverse'),
    ('INDPRO',     'Industrial Production',      9, 15, 'inverse'),
]

# Reaction measurement horizons (in minutes)
HORIZONS = [5, 15, 30, 60, 240]

# Minimum absolute surprise (in std devs) to count as "meaningful"
MIN_SURPRISE_ZSCORE = 0.5

# =============================================================================
# LOAD GOLD PRICE DATA
# =============================================================================
print('=' * 90)
print('NEWS RELEASE PRICE REACTION ANALYSIS — XAUUSD')
print('=' * 90)

print('\nLoading XAUUSD 1-min data...')
df_gold = pd.read_csv(DATA_PATH, sep='\t',
                      names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                             'TickVol', 'Vol', 'Spread'])
df_gold['Datetime'] = pd.to_datetime(df_gold['Date'] + ' ' + df_gold['Time'])
df_gold.set_index('Datetime', inplace=True)
df_gold = df_gold[['Open', 'High', 'Low', 'Close']].copy()
df_gold.sort_index(inplace=True)

gold_start = df_gold.index[0]
gold_end   = df_gold.index[-1]
print(f'  {len(df_gold):,} bars  |  {gold_start} → {gold_end}')

# Data is assumed to be in MT4 server time (GMT+2)
# US releases at 8:30 ET = 14:30 GMT+2 (winter) / 14:30 GMT+2 (summer varies)
# For simplicity, we use GMT+2 offsets:
#   ET + 7h in winter (EST=GMT-5, server=GMT+2 → diff=7)
#   ET + 6h in summer (EDT=GMT-4, server=GMT+2 → diff=6)
# We'll try both offsets and pick the one where we see the actual move.

ET_TO_SERVER_WINTER = 7  # hours to add to ET to get server time
ET_TO_SERVER_SUMMER = 6

# =============================================================================
# PULL FRED DATA
# =============================================================================
print('\nConnecting to FRED API...')
fred = Fred(api_key=FRED_API_KEY)

def get_release_dates_and_values(series_id, series_name):
    """
    Pull a FRED series and compute release-to-release changes.
    Uses vintage dates (actual release dates) when available,
    falls back to observation dates offset by typical publication lag.
    """
    try:
        # Try to get all releases (includes actual publication dates)
        all_releases = fred.get_series_all_releases(series_id)
        
        if all_releases is not None and len(all_releases) > 0:
            # all_releases has columns: 'realtime_start', 'date', 'value'
            # realtime_start = when data was published
            # date = the observation period
            # Keep first release of each observation (initial release, most market-moving)
            all_releases = all_releases.reset_index()
            first_releases = all_releases.groupby('date').first().reset_index()
            first_releases['value'] = pd.to_numeric(first_releases['value'], errors='coerce')
            first_releases = first_releases.dropna(subset=['value'])
            first_releases['release_date'] = pd.to_datetime(first_releases['realtime_start'])
            first_releases['obs_date'] = pd.to_datetime(first_releases['date'])
            first_releases = first_releases.sort_values('release_date')
            
            # Compute surprise: change from previous release
            first_releases['prev_value'] = first_releases['value'].shift(1)
            first_releases['change'] = first_releases['value'] - first_releases['prev_value']
            first_releases['pct_change'] = first_releases['value'].pct_change()
            
            # Standardize surprise
            std = first_releases['change'].std()
            if std > 0:
                first_releases['surprise_z'] = first_releases['change'] / std
            else:
                first_releases['surprise_z'] = 0
            
            return first_releases[['release_date', 'obs_date', 'value', 
                                   'prev_value', 'change', 'pct_change', 
                                   'surprise_z']].dropna()
        
    except Exception as e:
        print(f'    Warning: all_releases failed for {series_id}: {e}')
    
    # Fallback: use regular series
    try:
        series = fred.get_series(series_id)
        sdf = series.reset_index()
        sdf.columns = ['obs_date', 'value']
        sdf['value'] = pd.to_numeric(sdf['value'], errors='coerce')
        sdf = sdf.dropna()
        sdf['release_date'] = sdf['obs_date'] + pd.DateOffset(months=1)  # rough lag
        sdf['prev_value'] = sdf['value'].shift(1)
        sdf['change'] = sdf['value'] - sdf['prev_value']
        sdf['pct_change'] = sdf['value'].pct_change()
        std = sdf['change'].std()
        sdf['surprise_z'] = sdf['change'] / std if std > 0 else 0
        return sdf.dropna()
    except Exception as e:
        print(f'    Error pulling {series_id}: {e}')
        return pd.DataFrame()


def find_price_at_time(dt, offset_minutes=0):
    """Find gold price at a specific datetime, with tolerance."""
    target = dt + pd.Timedelta(minutes=offset_minutes)
    
    # Find nearest bar within 5-minute tolerance
    idx = df_gold.index.searchsorted(target)
    if idx >= len(df_gold):
        idx = len(df_gold) - 1
    if idx < 0:
        return None, None, None
    
    nearest = df_gold.index[idx]
    if abs((nearest - target).total_seconds()) > 300:  # 5 min tolerance
        return None, None, None
    
    return df_gold['Open'].iloc[idx], df_gold['High'].iloc[idx], df_gold['Low'].iloc[idx]


def measure_reaction(release_server_time):
    """
    Measure gold price reaction at various horizons after a release.
    Returns dict of horizon → price change, or None if data missing.
    """
    base_price, _, _ = find_price_at_time(release_server_time, 0)
    if base_price is None:
        return None
    
    reactions = {'base_price': base_price}
    
    for h in HORIZONS:
        # For the reaction, get the close at base+h minutes
        target_time = release_server_time + pd.Timedelta(minutes=h)
        idx = df_gold.index.searchsorted(target_time)
        if idx >= len(df_gold) or idx < 0:
            reactions[f'{h}m_chg'] = np.nan
            reactions[f'{h}m_high'] = np.nan
            reactions[f'{h}m_low'] = np.nan
            continue
        
        # Price at horizon
        reactions[f'{h}m_chg'] = df_gold['Close'].iloc[idx] - base_price
        
        # Max excursion within horizon
        start_idx = df_gold.index.searchsorted(release_server_time)
        window = df_gold.iloc[start_idx:idx+1]
        if len(window) > 0:
            reactions[f'{h}m_high'] = window['High'].max() - base_price
            reactions[f'{h}m_low']  = window['Low'].min() - base_price
        else:
            reactions[f'{h}m_high'] = np.nan
            reactions[f'{h}m_low']  = np.nan
    
    return reactions


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
all_events = []

for series_id, name, release_hour_et, release_min_et, expected_dir in INDICATORS:
    print(f'\n  Pulling {name} ({series_id})...')
    
    releases = get_release_dates_and_values(series_id, name)
    
    if releases is None or len(releases) == 0:
        print(f'    No data')
        continue
    
    # Filter to gold data range
    releases = releases[
        (releases['release_date'] >= gold_start.strftime('%Y-%m-%d')) &
        (releases['release_date'] <= gold_end.strftime('%Y-%m-%d'))
    ]
    
    print(f'    {len(releases)} releases in gold data range')
    
    if len(releases) < 5:
        continue
    
    for _, rel in releases.iterrows():
        release_date = pd.Timestamp(rel['release_date'])
        
        # Try winter offset first, then summer
        for offset_name, offset_hours in [('winter', ET_TO_SERVER_WINTER), 
                                           ('summer', ET_TO_SERVER_SUMMER)]:
            server_time = pd.Timestamp(
                year=release_date.year,
                month=release_date.month,
                day=release_date.day,
                hour=release_hour_et + offset_hours,
                minute=release_min_et
            )
            
            reaction = measure_reaction(server_time)
            if reaction is not None:
                event = {
                    'series_id':    series_id,
                    'indicator':    name,
                    'release_date': release_date,
                    'server_time':  server_time,
                    'tz_offset':    offset_name,
                    'value':        rel['value'],
                    'prev_value':   rel['prev_value'],
                    'change':       rel['change'],
                    'pct_change':   rel['pct_change'],
                    'surprise_z':   rel['surprise_z'],
                    'expected_dir': expected_dir,
                    **reaction
                }
                all_events.append(event)
                break  # found valid time, don't try other offset

events_df = pd.DataFrame(all_events)
print(f'\n\nTotal matched events: {len(events_df)}')

if len(events_df) == 0:
    print('No events matched. Check timezone offsets or data range.')
    sys.exit(1)

# =============================================================================
# ANALYSIS
# =============================================================================
print('\n' + '=' * 90)
print('PRICE REACTION BY INDICATOR')
print('=' * 90)

summary_rows = []

for (series_id, name), group in events_df.groupby(['series_id', 'indicator']):
    if len(group) < 5:
        continue
    
    row = {
        'indicator': name,
        'series_id': series_id,
        'n_events':  len(group),
    }
    
    for h in HORIZONS:
        col = f'{h}m_chg'
        if col in group.columns:
            chg = group[col].dropna()
            row[f'{h}m_avg_abs_move'] = chg.abs().mean()
            row[f'{h}m_avg_move']     = chg.mean()
            row[f'{h}m_median_abs']   = chg.abs().median()
            row[f'{h}m_std']          = chg.std()
            
            # Max excursion
            high_col = f'{h}m_high'
            low_col  = f'{h}m_low'
            if high_col in group.columns:
                row[f'{h}m_avg_max_up']   = group[high_col].mean()
                row[f'{h}m_avg_max_down'] = group[low_col].mean()
    
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Print ranked by 30-min absolute move
print(f'\n{"Indicator":<30} {"N":>4} {"5m Abs":>8} {"15m Abs":>8} {"30m Abs":>8} {"1h Abs":>8} {"4h Abs":>8}')
print('─' * 90)

if '30m_avg_abs_move' in summary_df.columns:
    ranked = summary_df.sort_values('30m_avg_abs_move', ascending=False)
else:
    ranked = summary_df

for _, r in ranked.iterrows():
    vals = []
    for h in HORIZONS:
        col = f'{h}m_avg_abs_move'
        vals.append(f'${r.get(col, 0):.2f}' if pd.notna(r.get(col, np.nan)) else '   N/A')
    print(f'{r["indicator"]:<30} {r["n_events"]:>4} {"  ".join(vals)}')

# =============================================================================
# SURPRISE DIRECTION VS REACTION
# =============================================================================
print('\n' + '=' * 90)
print('SURPRISE DIRECTION VS GOLD REACTION (30-min)')
print('=' * 90)
print('Positive surprise = data beat expectations (higher than prev)')
print('Expected: "inverse" indicators → gold falls on positive surprise')
print()

print(f'{"Indicator":<30} {"Dir":>7} {"Beat→Au":>10} {"Miss→Au":>10} {"Correct%":>9} {"Corr":>7}')
print('─' * 90)

for (series_id, name), group in events_df.groupby(['series_id', 'indicator']):
    if len(group) < 10 or '30m_chg' not in group.columns:
        continue
    
    valid = group.dropna(subset=['surprise_z', '30m_chg'])
    if len(valid) < 10:
        continue
    
    expected_dir = valid['expected_dir'].iloc[0]
    
    beats = valid[valid['surprise_z'] > MIN_SURPRISE_ZSCORE]
    misses = valid[valid['surprise_z'] < -MIN_SURPRISE_ZSCORE]
    
    beat_avg = beats['30m_chg'].mean() if len(beats) > 2 else np.nan
    miss_avg = misses['30m_chg'].mean() if len(misses) > 2 else np.nan
    
    # Check if reaction matches expected direction
    if expected_dir == 'inverse':
        # Beat → gold should fall (negative), Miss → gold should rise (positive)
        correct = ((beats['30m_chg'] < 0).sum() + (misses['30m_chg'] > 0).sum())
        total = len(beats) + len(misses)
    else:
        # Beat → gold should rise, Miss → gold should fall
        correct = ((beats['30m_chg'] > 0).sum() + (misses['30m_chg'] < 0).sum())
        total = len(beats) + len(misses)
    
    correct_pct = (correct / total * 100) if total > 0 else np.nan
    
    # Correlation between surprise and gold move
    corr = valid['surprise_z'].corr(valid['30m_chg'])
    
    beat_str = f'${beat_avg:+.2f}' if pd.notna(beat_avg) else '  N/A'
    miss_str = f'${miss_avg:+.2f}' if pd.notna(miss_avg) else '  N/A'
    corr_str = f'{corr:+.3f}' if pd.notna(corr) else '  N/A'
    pct_str  = f'{correct_pct:.0f}%' if pd.notna(correct_pct) else 'N/A'
    
    print(f'{name:<30} {expected_dir:>7} {beat_str:>10} {miss_str:>10} {pct_str:>9} {corr_str:>7}')

# =============================================================================
# REACTION PROFILE: AVG MOVE TRAJECTORY
# =============================================================================
print('\n' + '=' * 90)
print('TOP MOVERS: AVG ABSOLUTE REACTION TRAJECTORY')
print('=' * 90)

if '30m_avg_abs_move' in summary_df.columns:
    top5 = summary_df.nlargest(5, '30m_avg_abs_move')
    
    for _, r in top5.iterrows():
        trajectory = []
        for h in HORIZONS:
            val = r.get(f'{h}m_avg_abs_move', 0)
            trajectory.append(f'{h}m: ${val:.2f}' if pd.notna(val) else f'{h}m: N/A')
        print(f'  {r["indicator"]:<28} → {"  |  ".join(trajectory)}')

# =============================================================================
# TRADEABLE EVENTS: HIGH-SURPRISE + LARGE REACTION
# =============================================================================
print('\n' + '=' * 90)
print('TRADEABLE EVENTS (|surprise| > 1σ AND |30m move| > $3)')
print('=' * 90)

if '30m_chg' in events_df.columns:
    big_events = events_df[
        (events_df['surprise_z'].abs() > 1.0) & 
        (events_df['30m_chg'].abs() > 3.0)
    ].sort_values('30m_chg', key=abs, ascending=False)
    
    print(f'\n{"Date":<22} {"Indicator":<25} {"Surprise":>8} {"5m":>7} {"15m":>7} {"30m":>7} {"1h":>7}')
    print('─' * 90)
    
    for _, e in big_events.head(30).iterrows():
        print(f'{str(e["release_date"])[:19]:<22} '
              f'{e["indicator"]:<25} '
              f'{e["surprise_z"]:>+7.2f}σ '
              f'${e.get("5m_chg", 0):>+6.2f} '
              f'${e.get("15m_chg", 0):>+6.2f} '
              f'${e.get("30m_chg", 0):>+6.2f} '
              f'${e.get("60m_chg", 0):>+6.2f}')
    
    print(f'\n  Total tradeable events: {len(big_events)} out of {len(events_df)}')

# =============================================================================
# SAVE
# =============================================================================
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
events_df.to_csv(os.path.join(output_dir, 'news_events_reactions.csv'), index=False)
summary_df.to_csv(os.path.join(output_dir, 'news_indicator_summary.csv'), index=False)

print(f'\nSaved:')
print(f'  data/processed/news_events_reactions.csv')
print(f'  data/processed/news_indicator_summary.csv')

print('\n' + '=' * 90)
print('DONE')
print('=' * 90)
